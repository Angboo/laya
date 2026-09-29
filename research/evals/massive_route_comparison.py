"""Compare language routing by locale on the pinned MASSIVE test split.

This model-free diagnostic reports both checkpoint routing and named-language detection before
and after a proposed change to ``laya.lang``. It evaluates the full test split for every locale.

Run from the repository root:

    python research/evals/massive_route_comparison.py \
        --before-module /path/to/baseline/laya/lang.py \
        --after-module laya/lang.py --out research/results/massive_route_comparison.json

No model weights are loaded. A non-English MASSIVE locale should route to ``multilingual``;
English should route to ``english``. Named-language accuracy is reported separately because the
router only needs to choose a readable checkpoint, not identify every language by name.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import hf_hub_download, list_repo_files


DATASET = "mteb/amazon_massive_intent"
REVISION = "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_detector(path: Path, name: str):
    """Load a standalone Laya language module from an arbitrary checkout."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load language detector: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def route(detector: Any, router_module: Any, text: str) -> tuple[str, str]:
    """Use the repository's real Router._route decision path without loading a checkpoint."""
    router = router_module.Router(default="english")
    decision = router._route(text)
    return decision.model, (decision.get("detection") or {}).get("language") or "undecided"


def source_revision(path: Path) -> str:
    """Return the Git revision owning a detector source file."""
    source = path.resolve()
    root = next((parent for parent in source.parents if (parent / ".git").exists()), source.parent)
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def score_locale(detector: Any, router_module: Any, locale: str, path: Path) -> Dict[str, Any]:
    routes: Counter[str] = Counter()
    named_languages: Counter[str] = Counter()
    named_correct = 0
    examples = 0
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            model, named_language = route(detector, router_module, row["text"])
            routes[model] += 1
            named_languages[named_language] += 1
            named_correct += named_language == locale
            examples += 1
    return {
        "examples": examples,
        "route_correct": 0,
        "route_accuracy": None,
        "routed_to": dict(sorted(routes.items())),
        "named_language_correct": named_correct,
        "named_language_accuracy": round(named_correct / examples, 6) if examples else None,
        "named_language_guesses": dict(sorted(named_languages.items())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-module", required=True, type=Path,
                        help="path to baseline laya/lang.py")
    parser.add_argument("--after-module", required=True, type=Path,
                        help="path to candidate laya/lang.py")
    parser.add_argument("--out", type=Path, help="write the JSON report to this path")
    args = parser.parse_args(argv)

    files = list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    locales = sorted(name[len("test/"):-len(".json.gz")] for name in files
                     if name.startswith("test/") and name.endswith(".json.gz"))
    if not locales:
        raise RuntimeError("no per-locale MASSIVE test files at pinned revision")
    before_path = args.before_module.resolve()
    after_path = args.after_module.resolve()
    before = load_detector(before_path, "laya_lang_before")
    after = load_detector(after_path, "laya_lang_after")
    # Use this checkout's Router implementation and replace only its pure detector function for
    # each pass; all checkpoint-selection rules remain the repository code.
    from laya import router as router_module

    report: Dict[str, Any] = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "test",
        "model_free": True,
        "routing_policy": "english locale -> english checkpoint; every other locale -> multilingual",
        "routing_metric_note": "Router._route decisions are recorded; route accuracy uses expected checkpoint by MASSIVE locale, including Router's configured default for undecided text.",
        "before_module": str(before_path),
        "before_revision": source_revision(before_path),
        "after_module": str(after_path),
        "after_revision": source_revision(after_path),
        "locales": {},
        "overall": {},
    }
    before_total = {"examples": 0, "route_correct": 0, "named_language_correct": 0}
    after_total = dict(before_total)
    for locale in locales:
        local_path = Path(hf_hub_download(
            repo_id=DATASET,
            filename="test/%s.json.gz" % locale,
            repo_type="dataset",
            revision=REVISION,
        ))
        router_module.analyse = before.analyse
        old = score_locale(before, router_module, locale, local_path)
        router_module.analyse = after.analyse
        new = score_locale(after, router_module, locale, local_path)
        expected_model = "english" if locale == "en" else "multilingual"
        for item in (old, new):
            item["expected_checkpoint"] = expected_model
            item["route_correct"] = item["routed_to"].get(expected_model, 0)
            item["route_accuracy"] = round(item["route_correct"] / item["examples"], 6)
        report["locales"][locale] = {"before": old, "after": new,
                                      "route_accuracy_delta": round(new["route_accuracy"] - old["route_accuracy"], 6),
                                      "named_language_accuracy_delta": round(new["named_language_accuracy"] - old["named_language_accuracy"], 6)}
        for total, item in ((before_total, old), (after_total, new)):
            total["examples"] += item["examples"]
            total["route_correct"] += item["route_correct"]
            total["named_language_correct"] += item["named_language_correct"]

    for name, total in (("before", before_total), ("after", after_total)):
        count = total["examples"]
        report["overall"][name] = {
            **total,
            "route_accuracy": round(total["route_correct"] / count, 6) if count else None,
            "named_language_accuracy": round(total["named_language_correct"] / count, 6) if count else None,
        }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
