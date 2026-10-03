# Fast inference: AOTInductor

## AOTInductor

`DecisionModel` can be exported, compiled into a `.pt2` package, and loaded with
`torch._inductor.aoti_load_package`. The package returns both decision logits and action logits.
Tokenization, padding, temperature calibration and answer formatting remain the caller's job;
this does not add an `Agent` backend or change its default execution path.

The [closed PR #472](https://github.com/NandhaKishorM/laya/pull/472) documented the earlier dtype
blocker. The action head's pooled state and confidence features are computed in fp32, even
when a model's weights are explicitly bf16. Without autocast, the first action linear therefore
received an fp32 input and bf16 weights. The forward now casts the concatenated input to the
head's weight dtype **only outside autocast**. Softmax, entropy and the returned decision logits
retain their fp32 calculations. Existing fp32-parameter eager inference, including fp16/bf16
AMP, keeps its numerics; mixed weight/autocast dtypes also retain autocast's original conversion.

Export a separate evaluation copy converted to bf16, outside autocast. Declare the token
axis as `16 * Dim("tokens16", ...)` to satisfy the attention alignment guards. Rows and marker
counts can be dynamic independently. Do not assume an export captured under autocast can be
packaged outside that context: use explicit weight dtypes for this recipe.

## Reproduce offline

The assert-based check uses a tiny randomly initialized ModernBERT by default, without network
access. Pass a local checkpoint directory for a real-model measurement. Missing CUDA,
AOTInductor APIs or a C++ compiler produces an explicit `SKIP`; compilation and parity failures
on a supported installation fail the check.

```bash
python scripts/check_aoti.py --output-dir /tmp/laya-aoti-smoke
HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/laya-aoti/cache \
  python scripts/check_aoti.py --model /path/to/local/multilingual \
  --output-dir /tmp/laya-aoti
```

The output directory holds `decision.pt2`, `results.json`, `inputs.pt` and `eager.pt`. The JSON
includes full `nvidia-smi` snapshots, export/package/load times, package bytes, latency and
maximum absolute logit/probability deltas for both heads. Binary artifacts and compiler caches
are deliberately kept out of the repository.

## Measured on RTX 4070 Ti SUPER

2026-10-03, Linux x86-64, Python 3.12.13, torch 2.11.0+cu130, transformers 5.17.0,
NVIDIA driver 615.71.09, 16,376 MiB VRAM. Checkpoint: `convaiinnovations/laya`,
`multilingual` subdirectory at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
The baseline is main `fa9a2a7`. Full machine snapshots and unrounded measurements are in
[`aoti_multilingual_rtx4070.json`](https://github.com/NandhaKishorM/laya/blob/main/benchmarks/results/aoti_multilingual_rtx4070.json).

| Measurement | Before | After |
|---|---:|---:|
| Export | 5.00 s | 3.89 s |
| Packaging | failed after 16.59 s | 60.02 s |
| Package size | no artifact | 645,729,621 bytes (615.82 MiB) |
| Load in the already initialized process | unavailable | 0.447 s |
| Load in a fresh process with an empty cache | unavailable | 4.867 s |

The reproduced failure is at packaging, after successful export:
`mat1 and mat2 must have the same dtype, but got Float and BFloat16`.
Each run used a separate initially empty Inductor cache; AMP compilation ran before packaging,
so the package timing is not a fresh-interpreter cold-start measurement.
The fresh-process check loaded only the package and saved inputs (no checkpoint), with
`torch.compile` and packaging entry points blocked. It reproduced both heads' answers.
PyTorch still compiled a small CPU AVX capability probe into the initially empty cache;
no model graph or CUDA kernel was compiled. A blanket "no compiler activity at load" claim
would therefore be inaccurate on this runtime.

The same saved inputs contain seven decision rows across three batches, with real tokenized
billing/refund text, choice/score/noul questions, padding and unequal valid marker counts.
Each latency is the median of five groups of 30 forwards after ten warmups, using synchronized
wall-clock timing. `torch.compile(dynamic=True)` uses the default mode. No explicit CUDA graphs,
tokenization, export or compilation are included in these latency numbers.

| Rows × tokens × marker slots | AMP eager before → after (ms) | AMP compiled before → after (ms) | Explicit bf16 eager (ms) | Explicit bf16 compiled (ms) | AOTI bf16 (ms) |
|---|---:|---:|---:|---:|---:|
| 2 × 128 × 3 | 15.1173 → 14.1200 | 6.7117 → 6.9076 | 13.9635 | 4.8835 | 2.2674 |
| 1 × 256 × 3 | 15.4687 → 14.4156 | 6.7274 → 5.4772 | 14.7994 | 4.7250 | 2.4154 |
| 4 × 160 × 5 | 14.8452 → 14.5384 | 7.3872 → 5.5934 | 14.7115 | 5.1393 | 3.6018 |

AMP here means fp32 parameters with bf16 autocast. Explicit bf16 means weights and the residual
stream are bf16, without autocast; use those columns for the closest execution comparison to
the package. Explicit bf16 eager/compiled and AOTI were unavailable before this fix.

| Comparison, maximum over all seven rows | Decision logits | Decision probabilities | Action logits | Action probabilities |
|---|---:|---:|---:|---:|
| Existing eager before vs after, fp32 and fp16/bf16 AMP | 0 | 0 | 0 | 0 |
| AOTI vs explicit bf16 eager | 0.5625 | 0.00659859 | 0 | 0 |
| AOTI vs existing bf16 AMP eager | 0.4375 | 0.00769910 | 8.0 | 0 |

Both decision and action argmax agree on **7/7 rows** for both AOTI comparisons. Probabilities
are raw softmax outputs, without calibration. The action distribution is saturated on these
inputs, so its zero probability delta does not imply identical underlying logits against AMP.
The explicit bf16 compiled forward also differs from explicit bf16 eager (maximum decision
logit delta 0.5, probability delta 0.00659859). Reduced-precision compilation is not bit-exact.

The GPU was shared with desktop applications and other Python jobs; CPU compilation was also
shared, and clocks were not locked. These are observed timings, not an isolated speedup claim.
`nvidia-smi` snapshots bracketing the runs:

| Run / snapshot | GPU utilization | Used VRAM | Power | Temperature / state |
|---|---:|---:|---:|---|
| Before / start | 28% | 2,817 MiB | 12 W | 33°C / P8 |
| Before / end | 10% | 8,560 MiB | 23 W | 36°C / P3 |
| After / start | 0% | 2,634 MiB | 12 W | 33°C / P8 |
| After / end | 73% | 6,476 MiB | 160 W | 42°C / P2 |

## Remaining limits

- The artifact is specific to this checkpoint, precision, PyTorch/runtime stack and GPU target;
  portability to other hardware or PyTorch versions was not tested.
- This check exports rows 1–8, tokens 32–512 in multiples of 16, and marker slots 2–8. It exercises
  three shapes, including shapes different from the export example, rather than every point
  in those ranges. One valid option can use padded marker slots; a true one-slot tensor takes
  the separate single-option branch and needs a separate export. Arbitrary token lengths and
  a production bucketing/dispatch layer are outside this change.
- Seven rows verify the packaging regression, not broad checkpoint accuracy or calibrated
  confidence parity. Casting an export copy to bf16 differs from retaining fp32 weights under
  autocast; the existing eager path itself remains unchanged.
- The script requires a CUDA-capable PyTorch build and local compiler toolchain to create the
  artifact. The `.pt2` embeds the model and CUDA kernels; no hosted service is involved.
