# Compile defaults measurements (2026-10-03)

Source: PR bodies #718 and #576 (`gh pr view 718 --repo NandhaKishorM/laya --json body`
and the same command for 576). The requested `docs/fast-backends.md` is now
`docs/compile-and-fast-path.md`. Base: fa9a2a7, branch feat/compile-defaults.

Hardware: RTX 4070 Ti SUPER 16 GiB, torch 2.11.0+cu130, driver 615.71.09.
Interpreter: `/home/ckl/projects/S/laya/.venv/bin/python`.
Checkpoint: locally cached English `convaiinnovations/laya` snapshot
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; no network downloads.

Each `.command` contains the exact benchmark invocation; the matching `.log` contains
stdout/stderr, all 30 request timings, Dynamo/Inductor cache counters, allocator memory,
and full `nvidia-smi` snapshots before load and after requests. No caches were cleared:
each cold experiment starts with a new directory, and its restart uses the same directory
in a separate process. Three passes over the same ten varying request shapes assert
identical decoded answers within each process. Request 8 is the first single-row request.
Memory columns are PyTorch peak allocated/reserved MiB, not total device usage.

This is a shared desktop GPU. Baseline began at 755 MiB occupied. Other worktrees started
AOTInductor and GPU parity jobs during the automatic-warmup experiments; their processes
are visible in the snapshots. Contended timings are observations, not isolated speedup
estimates. The checkpoint's existing out-of-range temperature warning is retained in logs.
Disk-cache reuse is verified by FX cache counters, independently of timing.

## Item 1: automatic warm-up

`compile=True` warms after runtime/device initialization. `compile_warmup=False` preserves
lazy startup; `agent.warmup()` remains callable. Eager and TileLang paths do not auto-warm.
The constructor regression uses a tiny CPU model with mocked checkpoint I/O; the existing
Dynamo eager-backend check verifies the actual warm-up builds two graphs and later shapes
add none. API defaults are pinned in `tests/test_hooks_api.py`.

Validation commands and full output are in `warmup-validation.log` (all exit codes recorded).
Documentation dependencies were checked with:

```
uv pip install --python /home/ckl/projects/S/laya/.venv/bin/python -r requirements-docs.txt
```

Output: `Using Python 3.12.13 environment at: /home/ckl/projects/S/laya/.venv`;
`Checked 3 packages in 2ms`. Strict docs builds must have no `griffe:` lines.

| Run | Load s | Request 1 ms | Request 8 ms | Peak alloc/reserved MiB | FX hits/misses |
|---|---:|---:|---:|---:|---:|
| baseline-cold | 3.078 | 50154.96 | 40468.15 | 1680.0 / 1820.0 | 0 / 2 |
| baseline-restart | 3.124 | 23281.17 | 18366.73 | 1671.0 / 1794.0 | 2 / 0 |
| warmup-cold | 108.498 | 26.61 | 10.91 | 1685.6 / 1820.0 | 0 / 2 |
| warmup-restart | 61.314 | 27.42 | 10.69 | 1671.0 / 1796.0 | 2 / 0 |

Baseline-cold ran before code edits; use `--no-warmup` to reproduce its lazy behavior on this tree.
Warm-up shifts cost into load; it does not eliminate compilation or all possible guard specializations.
The two warm-up runs were contended. Cross-process reuse yielded two FX hits and zero misses.

## Item 2: opt-in persistent Laya cache

`compile_cache=True` selects `$XDG_CACHE_HOME/laya/torchinductor` or
`~/.cache/laya/torchinductor` only if `TORCHINDUCTOR_CACHE_DIR` is absent.
The default is false, existing settings win, and non-compiled/TileLang loads do nothing.
The choice is explicitly process-wide; PyTorch owns cache compatibility/invalidation.

The `cache-verified-*` experiments unset `TORCHINDUCTOR_CACHE_DIR`, set an isolated persistent
XDG root, assert the selected path, and test reuse in separate processes (see `.command` files).
The earlier `cache-cold`, `cache-restart`, and `cache-quiet-*` logs are diagnostic attempts,
NOT measurements of the Laya cache: an early Dynamo counters import populated the default
Inductor environment variable before the opt-in ran. This was caught by auditing the actual
printed path. The benchmark now imports counters after load, asserts the path, and the agent
configures the cache before model construction can import Dynamo. A regression pins that order. Full GPU state is attached to each run, not inferred from
allocator counters. Tests cover opt-in behavior, XDG/fallback paths, explicit overrides,
repeated setup, and constructor forwarding. `cache-verified-validation.log` records every gate and core suite after the correction.
The final ordering assertion is rerun with `python tests/test_compile.py`
(`cache-order-tests.log`, exit 0).

| Run | Load s | Request 1 ms | Request 8 ms | Peak alloc/reserved MiB | FX hits/misses |
|---|---:|---:|---:|---:|---:|
| cache-verified-cold | 111.643 | 22.86 | 8.69 | 1685.6 / 1820.0 | 0 / 2 |
| cache-verified-restart | 51.292 | 24.43 | 10.84 | 1671.0 / 1796.0 | 2 / 0 |

Both verified processes selected `/home/ckl/.cache/laya-compile-defaults-verified-20261003/laya/torchinductor`.
The second process hit both FX graphs, reducing load from 111.64 s to 51.29 s.
The cache location adds no meaningful GPU memory requirement; both peaks match the earlier
warm-up experiments. These final runs had no other compute jobs in their GPU snapshots.
Reboot/container persistence was not exercised; that depends on retaining the directory.
