"""Summarise `step4_gate_d_results.json` into the canonical 3-seed x 2-loss panel and a verdict.

Reads the Gate D results file (as emitted by the Kaggle Gate D kernel) and prints/writes:

- the per-cell table (accuracy / ECE / Brier / per-type accuracy / wall clock);
- `soft-ce - rlcd` paired accuracy deltas at seeds 0/1/2, plus mean/min/max and sign consistency;
- mean and sample std of accuracy per loss, and mean/sample std of the paired deltas;
- the same summaries for ECE and Brier, explicitly labelled as calibration-scope numbers;
- one evidence verdict from the fixed rule below.

The verdict thresholds are stated in the output, not hidden. No p-values: n=3 training seeds is
not enough for significance, and the Coordinator's rule asks for direction, magnitude and sign
consistency instead.

    python research/scripts/step4_benchmark/analyze_gate_d.py --results step4_gate_d_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List, Optional

SEEDS = (0, 1, 2)
LOSSES = ("rlcd", "soft-ce")
# A soft-CE edge only counts as "useful" if it clears this in the mean and on every seed. Stated
# here so the verdict is reproducible rather than eyeballed.
MATERIAL_ACCURACY_DELTA = 0.005
# A per-type or calibration regression this large or larger counts as "material" for the verdict.
MATERIAL_REGRESSION = 0.010


def cell(results: Dict[str, Any], loss: str, seed: int) -> Optional[Dict[str, Any]]:
    entry = results.get("%s_seed%d" % (loss, seed))
    if not isinstance(entry, dict) or "accuracy" not in entry:
        return None
    return entry


def paired(results: Dict[str, Any], metric: str) -> Dict[int, Optional[float]]:
    out: Dict[int, Optional[float]] = {}
    for seed in SEEDS:
        a, b = cell(results, "soft-ce", seed), cell(results, "rlcd", seed)
        if a is None or b is None:
            out[seed] = None
        else:
            out[seed] = round(float(a[metric]) - float(b[metric]), 4)
    return out


def summarise(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="step4_gate_d_results.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    print("== per-cell ==")
    header = "%-8s %-4s %8s %8s %8s %8s %10s" % ("loss", "seed", "acc", "ece", "brier", "wall_s", "finite")
    print(header)
    for loss in LOSSES:
        for seed in SEEDS:
            c = cell(results, loss, seed)
            if c is None:
                print("%-8s %-4d %8s" % (loss, seed, "MISSING"))
                continue
            print("%-8s %-4d %8.4f %8.4f %8.4f %10s %10s"
                  % (loss, seed, c["accuracy"], c["ece"], c["brier"],
                     c.get("wall_clock_s", "-"), c.get("finite", "-")))

    acc_deltas = paired(results, "accuracy")
    ece_deltas = paired(results, "ece")
    brier_deltas = paired(results, "brier")
    acc_values = {loss: [cell(results, loss, s)["accuracy"] for s in SEEDS if cell(results, loss, s)]
                  for loss in LOSSES}
    complete = all(acc_deltas[s] is not None for s in SEEDS)
    present = [acc_deltas[s] for s in SEEDS if acc_deltas[s] is not None]

    print("\n== paired soft-ce - rlcd ==")
    print("accuracy delta by seed:", acc_deltas)
    print("accuracy delta summary:", summarise(present))
    print("sign consistency (all 3 > 0):", complete and all(d > 0 for d in present))
    print("sign consistency (all 3 < 0):", complete and all(d < 0 for d in present))
    print("rlcd accuracy:", summarise(acc_values["rlcd"]))
    print("soft-ce accuracy:", summarise(acc_values["soft-ce"]))
    ece_present = [ece_deltas[s] for s in SEEDS if ece_deltas[s] is not None]
    brier_present = [brier_deltas[s] for s in SEEDS if brier_deltas[s] is not None]
    print("ECE delta by seed (calibration scope only):", ece_deltas, summarise(ece_present))
    print("Brier delta by seed (calibration scope only):", brier_deltas, summarise(brier_present))

    per_type = {}
    for loss in LOSSES:
        for seed in SEEDS:
            c = cell(results, loss, seed)
            if c and c.get("per_type_accuracy"):
                for k, v in c["per_type_accuracy"].items():
                    per_type.setdefault(k, {}).setdefault(loss, []).append(float(v))
    print("\n== per-type accuracy (mean over seeds) ==")
    for k, by_loss in sorted(per_type.items()):
        print("  %-8s rlcd %s  soft-ce %s"
              % (k, summarise(by_loss.get("rlcd", [])), summarise(by_loss.get("soft-ce", []))))

    verdict = "INSUFFICIENT"
    reason = "panel incomplete (missing a loss/seed cell)"
    if complete:
        consistent_up = all(d > 0 for d in present)
        mean_delta = statistics.fmean(present)
        worst_type_regress = 0.0
        for k, by_loss in per_type.items():
            if by_loss.get("rlcd") and by_loss.get("soft-ce"):
                worst_type_regress = min(worst_type_regress,
                                         statistics.fmean(by_loss["soft-ce"]) - statistics.fmean(by_loss["rlcd"]))
        if consistent_up and mean_delta >= MATERIAL_ACCURACY_DELTA and worst_type_regress > -MATERIAL_REGRESSION:
            verdict = "CHANGE_DEFAULT_SUPPORTED"
            reason = ("soft-ce ahead on all 3 seeds, mean delta %.4f >= %.3f, no per-type regression beyond %.3f"
                      % (mean_delta, MATERIAL_ACCURACY_DELTA, MATERIAL_REGRESSION))
        elif consistent_up and mean_delta >= MATERIAL_ACCURACY_DELTA:
            verdict = "INSUFFICIENT"
            reason = ("accuracy favours soft-ce but a per-type regression of %.4f exceeds %.3f"
                      % (worst_type_regress, MATERIAL_REGRESSION))
        elif all(d < 0 for d in present):
            verdict = "KEEP_RLCD_SUPPORTED"
            reason = "soft-ce behind on all 3 seeds"
        else:
            verdict = "INSUFFICIENT"
            reason = "sign not consistent or mean delta %.4f below the %.3f material bar" % (mean_delta, MATERIAL_ACCURACY_DELTA)

    summary = {
        "accuracy_delta_by_seed": acc_deltas,
        "accuracy_delta_summary": summarise(present),
        "sign_all_positive": complete and all(d > 0 for d in present),
        "rlcd_accuracy": summarise(acc_values["rlcd"]),
        "soft_ce_accuracy": summarise(acc_values["soft-ce"]),
        "ece_delta_by_seed": ece_deltas,
        "brier_delta_by_seed": brier_deltas,
        "per_type_mean": {k: {l: summarise(v) for l, v in by.items()} for k, by in per_type.items()},
        "verdict": verdict,
        "reason": reason,
        "thresholds": {"material_accuracy_delta": MATERIAL_ACCURACY_DELTA,
                       "material_regression": MATERIAL_REGRESSION},
        "caveats": [
            "n=3 training seeds: direction/magnitude, not significance.",
            "ECE/Brier reflect the current evaluator's calibration path (notebook-aligned scope), "
            "not a universal all-question metric.",
        ],
    }
    print("\n== verdict ==")
    print(verdict, "--", reason)
    print("thresholds:", summary["thresholds"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
