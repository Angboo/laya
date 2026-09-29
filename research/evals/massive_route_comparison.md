# Swedish routing: MASSIVE full-test comparison

This report compares the `laya.lang` detector on all 51 locale files in the pinned
`mteb/amazon_massive_intent` test split (dataset revision
`940fd47a81eaa7f2cc7b129674d945d618ac38c2`). Each locale contributes 2,974 examples,
for 151,674 examples total. The comparison is model-free: it evaluates the same text
against the baseline `main` detector and the proposed detector without loading either
checkpoint.

## Method

The expected route is `english` for the `en` locale and `multilingual` for every other
locale. The script calls the repository's real `Router._route` for both detectors, so
all other routing rules stay the same, including the configured `english` default when
Latin text is undecided. Route accuracy measures only that checkpoint choice. Exact
named-language accuracy is reported separately: a detector can safely choose
`multilingual` while leaving the specific language undecided. Results for every locale,
including guess counts, are in
[`../results/massive_route_comparison_swedish.json`](../results/massive_route_comparison_swedish.json).

Reproduce from the repository root, passing a baseline and candidate checkout:

```sh
python research/evals/massive_route_comparison.py \
  --before-module /path/to/baseline/laya/lang.py \
  --after-module laya/lang.py \
  --out research/results/massive_route_comparison_swedish.json
```

The report records the source Git revisions and pinned dataset revision. The script
downloads the split from Hugging Face on first run and needs `huggingface_hub` installed.

## Results

| Measure | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Overall checkpoint route accuracy | 77.428% | 77.429% | +0.001 pp |
| Exact named-language accuracy | 8.837% | 8.841% | +0.004 pp |
| Swedish checkpoint route accuracy | 86.954% | 86.954% | 0.000 pp |
| Swedish named-language accuracy | 0.000% | 30.296% | +30.296 pp |
| Danish checkpoint route accuracy | 53.430% | 53.430% | 0.000 pp |
| Norwegian Bokmål route accuracy | 49.630% | 49.630% | 0.000 pp |

Checkpoint route accuracy changes only for `it` (+0.034 pp) and `pt` (+0.034 pp); no
locale decreased. The overall route gain is two examples out of 151,674. Named-language
accuracy also changes slightly for `de`, `it`, `pt`, and `ro`, reflecting shared evidence
rules rather than locale-specific language work. For the 51 locale rows and exact
before/after counts, use the JSON artifact rather than rounded percentages here.

## Interpretation and limits

MASSIVE measures intent classification utterances, not customer-support transcripts.
It supplies a broad per-locale routing check, not evidence that Swedish support
customers are served well end to end. On the pinned split, the Swedish change improves
named-language detection but barely changes which checkpoint is selected; the
configured default already sends most ambiguous examples to English.

Swedish, Danish, and Norwegian share short words. In the candidate results, four
Norwegian examples are named `sv`; all still route to the same multilingual checkpoint
as before. The detector does not claim that checkpoint-routing gains prove perfect
language identification. Broader Swedish support data and a focused false-positive
review remain useful follow-up work.
