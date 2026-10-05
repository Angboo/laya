"""Read-only evidence inspection over existing Laya artifacts.

This module reports what evidence exists for a checkpoint -- and what is missing,
insufficient, or incomparable -- without loading model weights, without needing
torch, and without writing a new manifest. It only reads artifacts the upstream
train/eval path already persists:

- a checkpoint directory's ``rl_agent_config.json`` (``training.laya_train`` and
  ``training.laya_train_calibration`` as written by merged #931/#933);
- an existing ``laya-evals`` report JSON (identity semantics from `laya.evals`).

Every state it assigns is either derived directly from an upstream artifact or
marked UNKNOWN. It never manufactures a healthy-looking state from absent input.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

PRESENT = "PRESENT"
MISSING = "MISSING"
INSUFFICIENT = "INSUFFICIENT"
UNKNOWN = "UNKNOWN"
INCOMPARABLE = "INCOMPARABLE"

# Documented upstream floors/thresholds. `laya.calibrate.MIN_TYPE_N` and
# `laya.train.CALIB_WARN_N` live in modules that import torch at module scope, so a
# read-only command cannot import them without pulling torch in. We prefer the live
# constants when torch is importable and fall back to the documented values with an
# explicit provenance note when it is not; we never silently duplicate.
_FALLBACK_MIN_TYPE_N = 10
_FALLBACK_CALIB_WARN_N = 50


def _thresholds() -> Tuple[int, int, str]:
    try:
        from .calibrate import MIN_TYPE_N  # noqa: F401  (imports torch on success)
        from .train import CALIB_WARN_N
        return int(MIN_TYPE_N), int(CALIB_WARN_N), "upstream modules"
    except Exception:
        return _FALLBACK_MIN_TYPE_N, _FALLBACK_CALIB_WARN_N, "documented fallback (torch unavailable)"


def _entry(state: str, reason: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"state": state, "reason": reason}
    if detail:
        out["detail"] = detail
    return out


def inspect_calibration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence state of the persisted calibration fit in a checkpoint config."""
    training = config.get("training")
    if not isinstance(training, dict):
        return _entry(UNKNOWN, "legacy checkpoint: no training metadata persisted; cannot tell whether calibration ran")
    calibration = training.get("laya_train_calibration")
    if not isinstance(calibration, dict):
        return _entry(UNKNOWN, "training metadata present but no calibration block; checkpoint predates #933")
    min_type_n, calib_warn_n, threshold_source = _thresholds()
    types: Dict[str, Any] = {}
    for name, entry in calibration.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("items"), int):
            types[name] = _entry(UNKNOWN, "calibration entry is malformed")
            continue
        items = entry["items"]
        issues = entry.get("issues") if isinstance(entry.get("issues"), list) else []
        if items == 0:
            types[name] = _entry(MISSING, "question type present but zero calibration items", {"items": 0})
        elif items < min_type_n:
            types[name] = _entry(INSUFFICIENT, "not fitted: %d items, below MIN_TYPE_N=%d, temperature stays 1.0"
                                 % (items, min_type_n), {"items": items, "issues": issues})
        elif issues:
            insufficient = [i for i in issues if "fitted on only" in i or "clamp" in i or "unchanged at 1.0" in i]
            if insufficient:
                types[name] = _entry(INSUFFICIENT, "fitted with evidence limitations: " + "; ".join(issues),
                                     {"items": items, "issues": issues})
            else:
                types[name] = _entry(PRESENT, "fitted; issues recorded: " + "; ".join(issues),
                                     {"items": items, "issues": issues, "temperature": entry.get("temperature")})
        elif items < calib_warn_n:
            types[name] = _entry(INSUFFICIENT, "fitted on only %d items, below CALIB_WARN_N=%d"
                                 % (items, calib_warn_n), {"items": items, "issues": issues})
        else:
            types[name] = _entry(PRESENT, "fitted on %d items with no recorded issues" % items,
                                 {"items": items, "temperature": entry.get("temperature")})
    present = sum(1 for t in types.values() if t["state"] == PRESENT)
    overall = PRESENT if types and present == len(types) else (
        MISSING if types and all(t["state"] == MISSING for t in types.values()) else INSUFFICIENT)
    return {
        "state": overall,
        "threshold_source": threshold_source,
        "types": types,
    }


def inspect_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence state of an eval report's identity fields (reuses EvalReport semantics)."""
    config = report.get("config")
    if not isinstance(config, dict):
        return _entry(UNKNOWN, "report has no config block; identity unknown")
    fields = {}
    for key in ("schema", "dataset_sha256", "questions_sha256", "laya_version"):
        value = config.get(key)
        if value is None and report.get(key) is not None:
            value = report.get(key)
        fields[key] = _entry(PRESENT, "recorded", {"value": value}) if value is not None else _entry(UNKNOWN, "not recorded in this report")
    identity_keys = [fields["schema"], fields["dataset_sha256"], fields["questions_sha256"]]
    overall = PRESENT if all(f["state"] == PRESENT for f in identity_keys) else UNKNOWN
    return {"state": overall, "fields": fields}


def relate(config: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic checkpoint <-> report relationship. Never infers from names/paths."""
    report_config = report.get("config") if isinstance(report.get("config"), dict) else {}
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    laya_train = training.get("laya_train") if isinstance(training.get("laya_train"), dict) else {}
    conflicts = []
    matches = []
    # Only compare keys that exist on both sides with a real value.
    for key in ("laya_version",):
        a = laya_train.get(key) or config.get(key)
        b = report_config.get(key) or (report.get("config") or {}).get("checkpoint", {}).get("laya")
        if a is not None and b is not None:
            (matches if str(a) == str(b) else conflicts).append(key)
    if conflicts:
        return _entry(INCOMPARABLE, "checkpoint and report disagree on: " + ", ".join(conflicts))
    if matches:
        return _entry(PRESENT, "checkpoint and report agree on: " + ", ".join(matches))
    return _entry(UNKNOWN, "no shared identity field between this checkpoint config and this report")


def inspect_checkpoint(checkpoint_dir: str, report_path: Optional[str] = None) -> Dict[str, Any]:
    """Full read-only evidence inspection for one checkpoint (optionally one report)."""
    config_path = os.path.join(checkpoint_dir, "rl_agent_config.json")
    if not os.path.isdir(checkpoint_dir) or not os.path.isfile(config_path):
        raise FileNotFoundError("no rl_agent_config.json under %r" % checkpoint_dir)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("%s is not a JSON object" % config_path)
    out: Dict[str, Any] = {"checkpoint": checkpoint_dir, "calibration": inspect_calibration(config)}
    if report_path is not None:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict):
            raise ValueError("%s is not a JSON object" % report_path)
        out["report"] = inspect_report(report)
        out["relationship"] = relate(config, report)
    return out


def format_summary(result: Dict[str, Any]) -> str:
    lines: List[str] = ["evidence summary for %s" % result["checkpoint"]]
    cal = result["calibration"]
    lines.append("calibration: %s (thresholds: %s)" % (cal["state"], cal.get("threshold_source", "n/a")))
    for name, t in (cal.get("types") or {}).items():
        lines.append("  %-24s %s -- %s" % (name, t["state"], t["reason"]))
    if cal.get("state") == UNKNOWN and "reason" in cal:
        lines.append("  " + cal["reason"])
    if "report" in result:
        rep = result["report"]
        lines.append("eval report identity: %s" % rep["state"])
        for key, f in rep.get("fields", {}).items():
            lines.append("  %-18s %s" % (key, f["state"]))
        lines.append("checkpoint<->report: %s -- %s" % (result["relationship"]["state"], result["relationship"]["reason"]))
    return "\n".join(lines)
