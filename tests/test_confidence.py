"""`confidence` carries two definitions, and only one of them is the calibrated quantity.

    choice / score  ->  confidence_from_probs(p, k) == 1 - H(p) / log(k)
    noul            ->  max(p_true, 1 - p_true)     == max(p)

They are not on the same scale. The same two-option distribution comes back as 0.90 from a
`noul` and 0.53 from an equivalent two-option `choice`, and every shipped preset mixes the two
types in one call -- `moderation_questions()` is four `noul` and one `score`. The README's
"Automated Confidence Gating" section gates all of them on a single threshold.

Both benchmark harnesses in research/scripts take `conf = max(probs)` before calling
`ece_score`, so every ECE figure in the README describes max(p) and not normalized entropy.
`answer_confidence` reports that quantity on every question type, additively: `confidence` is
untouched, so nothing a caller gates on today moves.

No weights are loaded: the confidence helpers are pure.
"""
import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import answer_confidence, confidence_from_probs, ece_score  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --------------------------------------------------------------- answer_confidence is max(p)
for probs in ([0.5, 0.5], [0.1, 0.9], [0.7, 0.2, 0.1], [0.25] * 4, [1.0, 0.0, 0.0]):
    p = np.array(probs)
    check_true("answer/max of %s" % (probs,), close(answer_confidence(p, len(probs)), max(probs)))

check("answer/only the first k entries count", answer_confidence(np.array([0.4, 0.6, 0.99]), 2), 0.6)
check("answer/k=1 is certain", answer_confidence(np.array([1.0]), 1), 1.0)
check("answer/k=0 does not crash", answer_confidence(np.array([]), 0), 1.0)

# --------------------------------------------------------------- it matches noul's definition
# `noul` reports max(p_true, 1 - p_true), which over two options is exactly max(p). So for a
# noul answer the new field equals the existing one, and the two agree by construction.
for p_true in (0.0, 0.05, 0.3, 0.5, 0.62, 0.9, 1.0):
    p = np.array([1.0 - p_true, p_true])
    check_true("noul/answer_confidence equals the shipped noul confidence at p=%.2f" % p_true,
               close(answer_confidence(p, 2), max(p_true, 1.0 - p_true)))

# --------------------------------------------------------------- the two scales really differ
# Same distribution, two question types. 0.85 is the threshold the README's gating section uses.
disagree = []
for p_true in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
    p = np.array([1.0 - p_true, p_true])
    if (answer_confidence(p, 2) >= 0.85) != (confidence_from_probs(p, 2) >= 0.85):
        disagree.append(p_true)
check_true("scales/the two disagree across the documented threshold", disagree == [0.85, 0.90, 0.95],
           "disagreed at %s" % (disagree,))
check_true("scales/entropy sits far below max(p) on the same distribution",
           confidence_from_probs(np.array([0.1, 0.9]), 2) < 0.55 < answer_confidence(np.array([0.1, 0.9]), 2))
# max(p) over two options has a floor of 0.5; entropy reads 0.0 for the same coin flip.
check("scales/a coin flip is 0.5 on max(p)", answer_confidence(np.array([0.5, 0.5]), 2), 0.5)
check("scales/and 0.0 on entropy", confidence_from_probs(np.array([0.5, 0.5]), 2), 0.0)

# --------------------------------------------------------------- only one of them is calibrated
# Perfectly calibrated predictions: an answer reported at top probability c is right exactly c
# of the time. ECE on max(p) must be near zero. ECE on normalized entropy must not be, which is
# why it cannot be compared against a probability threshold.
rng = np.random.default_rng(0)
tops, ents, correct = [], [], []
for c in np.linspace(0.30, 0.99, 24):
    rest = (1.0 - c) / 2.0
    p = np.array([c, rest, rest])
    for _ in range(400):
        tops.append(answer_confidence(p, 3))
        ents.append(confidence_from_probs(p, 3))
        correct.append(1.0 if rng.random() < c else 0.0)

tops, ents, correct = np.array(tops), np.array(ents), np.array(correct)
ece_top, ece_ent = ece_score(tops, correct), ece_score(ents, correct)
check_true("calibration/max(p) is calibrated on calibrated data (ECE %.4f)" % ece_top,
           ece_top < 0.03, "ECE %.4f" % ece_top)
check_true("calibration/entropy is not (ECE %.4f)" % ece_ent, ece_ent > 0.20, "ECE %.4f" % ece_ent)
check_true("calibration/entropy is worse by a wide margin", ece_ent > 5 * ece_top,
           "top %.4f vs entropy %.4f" % (ece_top, ece_ent))

# --------------------------------------------------------------- the entropy helper is untouched
for probs, k in (([0.1, 0.9], 2), ([0.25] * 4, 4), ([0.7, 0.2, 0.1], 3)):
    p = np.array(probs)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    check_true("entropy/formula for k=%d" % k,
               close(confidence_from_probs(p, k), float(np.clip(1 - ent / math.log(k), 0.0, 1.0))))

# --------------------------------------------------------------- min_confidence validation & abstention (#361)
from laya.confidence import check_min_confidence, flag_low_confidence  # noqa: E402

# check_min_confidence validates [0.0, 1.0] and rejects non-floats and bools
for valid in (0.0, 0.5, 1.0, 0, 1, 0.85):
    check("min_confidence/valid %.2f" % valid, check_min_confidence(valid), float(valid))

for invalid in (True, False, -0.01, 1.01, -1.0, 2.0, float("nan"), float("inf"), float("-inf"), "0.5", None, [0.5]):
    try:
        check_min_confidence(invalid)
        FAIL.append("min_confidence/should reject %r" % (invalid,))
    except ValueError:
        PASS.append("min_confidence/rejected %r" % (invalid,))

# flag_low_confidence marks answers where answer_confidence < threshold
sample_res = [{
    "answers": {
        "q_high": {"type": "choice", "choice": "a", "answer_confidence": 0.92, "confidence": 0.8},
        "q_low": {"type": "score", "score": 1, "answer_confidence": 0.45, "confidence": 0.4},
        "q_fallback": {"type": "choice", "choice": "b", "confidence": 0.3},
        "q_exact": {"type": "choice", "choice": "c", "answer_confidence": 0.70},
    }
}]

# min_confidence=0.0 is a no-op (default behavior untouched)
flag_low_confidence(sample_res, 0.0)
check_true("flag/0.0 is no-op", not any("low_confidence" in a for a in sample_res[0]["answers"].values()))

# min_confidence=0.70: q_low (0.45) and q_fallback (0.3) flagged; q_high (0.92) and q_exact (0.70) NOT flagged
flag_low_confidence(sample_res, 0.70)
ans = sample_res[0]["answers"]
check_true("flag/q_low flagged", ans["q_low"].get("low_confidence") is True)
check_true("flag/q_fallback flagged", ans["q_fallback"].get("low_confidence") is True)
check_true("flag/q_high unflagged", "low_confidence" not in ans["q_high"])
check_true("flag/q_exact unflagged", "low_confidence" not in ans["q_exact"])
check("flag/answer_confidence intact", ans["q_low"]["answer_confidence"], 0.45)
check("flag/raw choice intact", ans["q_high"]["choice"], "a")

# --------------------------------------------------------------- the gate's state is reported
# `low_confidence` is written only when the gate fires, so its absence cannot tell a caller
# "a gate ran and this answer cleared it" from "no gate ran at all". An operator with 10,000
# logged decisions cannot compute an abstention rate, cannot tell whether a run was gated, and
# cannot re-split a batch that used different thresholds per request class -- the threshold is
# consumed and dropped. `docs/staged-adoption.md` asks a shadow record to carry "errors and any
# fallback or review decision"; there was no field to put that in.
#
# The contract: where a gate ran, its state is an explicit value, so it is never inferred from a
# missing field. Where no gate was configured, nothing is written at all -- an ungated call returns
# the payload it always returned, rather than growing a field every existing caller would then have
# to read. The two cases are told apart by the presence of `abstention`, which is why the vocabulary
# has no "not configured" member to read out of the field.
from laya.confidence import (  # noqa: E402
    GATE_ABSTAINED,
    GATE_PASSED,
    GATE_STATES,
    GATE_UNEVALUATED,
    apply_confidence_gate,
)

check("gate/the vocabulary is closed",
      list(GATE_STATES), ["passed", "abstained", "unevaluated"])

fresh = lambda: [{  # noqa: E731
    "answers": {
        "q_high": {"type": "choice", "choice": "a", "answer_confidence": 0.92},
        "q_low": {"type": "choice", "choice": "b", "answer_confidence": 0.45},
    }
}]

# No gate: the payload is untouched. Not a sentinel, not a flag -- byte-for-byte what the caller
# got before this helper existed, so no existing caller has to learn a field to keep working.
off = fresh()
before = copy.deepcopy(off)
apply_confidence_gate(off, None)
check("gate/an ungated call writes nothing at all", off, before)
for qid, a in off[0]["answers"].items():
    check_true("gate/ungated %s reports no state" % qid, "abstention" not in a)
    check_true("gate/ungated %s carries no threshold" % qid, "abstention_threshold" not in a)
    check_true("gate/ungated %s is not a flag" % qid, "low_confidence" not in a)

# A gate that ran: the answer that cleared it says `passed`, which is the state the boolean
# could not express. This is the whole point -- before, this answer was byte-identical to the
# ungated one above.
on = fresh()
apply_confidence_gate(on, 0.80)
check("gate/cleared answer says passed", on[0]["answers"]["q_high"]["abstention"], GATE_PASSED)
check("gate/cleared answer is still unflagged", "low_confidence" in on[0]["answers"]["q_high"], False)
check("gate/cleared answer echoes the threshold", on[0]["answers"]["q_high"]["abstention_threshold"], 0.80)
check("gate/below-threshold answer says abstained", on[0]["answers"]["q_low"]["abstention"], GATE_ABSTAINED)
check("gate/below-threshold answer is flagged", on[0]["answers"]["q_low"]["low_confidence"], True)
check("gate/below-threshold echoes the threshold", on[0]["answers"]["q_low"]["abstention_threshold"], 0.80)
check_true("gate/the raw answer is untouched",
           on[0]["answers"]["q_high"]["choice"] == "a" and on[0]["answers"]["q_low"]["choice"] == "b")

# The two states a caller used to be unable to tell apart are now told apart by a value, and the
# distinction survives a round trip through JSON, which is how a shadow record reaches a log.
import json as _json  # noqa: E402

round_tripped = _json.loads(_json.dumps(on[0]["answers"]))
check("gate/survives a JSON round trip", round_tripped["q_high"]["abstention"], GATE_PASSED)

# One rule, one implementation: the flag stays `flag_low_confidence`'s, not a second copy of it.
# `apply_confidence_gate(results, 0.0)` is a gate that ran and nothing can fail, which is why
# `flag_low_confidence`'s 0.0 no-op above is still correct and still asserted.
zero = fresh()
apply_confidence_gate(zero, 0.0)
check("gate/0.0 reports passed", zero[0]["answers"]["q_low"]["abstention"], GATE_PASSED)
check_true("gate/0.0 still flags nothing", "low_confidence" not in zero[0]["answers"]["q_low"])

# A non-finite or missing confidence is skipped by `flag_low_confidence`, so it is not an
# abstention. It is also not a pass: the gate ran and could not decide, which is the state Argo
# Rollouts ships as `Inconclusive` and the one collapsing it into success would be the same lie
# `low_confidence`'s absence already tells.
odd = [{"answers": {
    "q_nan": {"type": "choice", "answer_confidence": float("nan")},
    "q_none": {"type": "choice"},
    "q_inf": {"type": "choice", "answer_confidence": float("inf")},
    "q_bool": {"type": "choice", "answer_confidence": True},
    # The fallback path, which is where the number used to leak. With no usable
    # `answer_confidence` the gate reads the entropy `confidence`, and an entropy NaN has to come
    # back as "nothing to gate on" -- not as the NaN itself, which is not None and would
    # therefore read as a pass. `flag_low_confidence` never noticed (NaN < x is False, and
    # None is not None is also False), so nothing else pinned this down.
    "q_nan_entropy": {"type": "choice", "confidence": float("nan")},
    "q_bool_entropy": {"type": "choice", "confidence": True},
    "q_fine": {"type": "choice", "answer_confidence": 0.99},
}}]
apply_confidence_gate(odd, 0.80)
for qid, a in odd[0]["answers"].items():
    if qid == "q_fine":
        check("gate/a usable confidence is evaluated %s" % qid, a["abstention"], GATE_PASSED)
    else:
        check("gate/unusable confidence is unevaluated %s" % qid, a["abstention"], GATE_UNEVALUATED)
    check_true("gate/unusable confidence is not flagged %s" % qid, "low_confidence" not in a)

# `answer_confidence` is the quantity the gate reads, so a `noul`-style entropy `confidence`
# below threshold does not abstain while `answer_confidence` above it does. The state follows
# the same field the flag follows.
pref = [{"answers": {
    "q_a": {"type": "noul", "answer_confidence": 0.91, "confidence": 0.10},
    "q_b": {"type": "choice", "answer_confidence": 0.10, "confidence": 0.99},
}}]
apply_confidence_gate(pref, 0.80)
check("gate/gate reads answer_confidence (a)", pref[0]["answers"]["q_a"]["abstention"], GATE_PASSED)
check("gate/gate reads answer_confidence (b)", pref[0]["answers"]["q_b"]["abstention"], GATE_ABSTAINED)

# The guards the six call sites rely on: a result that is not a dict, answers that are not a
# dict, and answers that are not dicts. Every call site passes a list it has already partly
apply_confidence_gate([None, {"answers": None}, {"answers": [1, 2]}], 0.5)
apply_confidence_gate([], None)
PASS.append("gate/non-dict results are skipped without raising")

# --------------------------------------------------------------- BUG-005 regression: stale low_confidence across reuse (#910)
# `flag_low_confidence` and `apply_confidence_gate` used to only set `low_confidence: True` and
# never clear it. When a result dict was re-evaluated with a more permissive threshold (or 0.0),
# the stale flag remained, causing downstream callers to treat the answer as permanently abstained.

# 1. flag_low_confidence clears stale flag on re-evaluation
reuse_res = [{
    "answers": {
        "intent": {"choice": "refund", "answer_confidence": 0.8},
    }
}]
reuse_ans = reuse_res[0]["answers"]["intent"]
flag_low_confidence(reuse_res, 0.9)
check_true("stale/flag strict threshold sets flag", reuse_ans.get("low_confidence") is True)
flag_low_confidence(reuse_res, 0.5)
check_true("stale/flag permissive threshold clears flag", "low_confidence" not in reuse_ans)

clean_res = [{"answers": {"q": {"answer_confidence": 0.95}}}]
flag_low_confidence(clean_res, 0.5)
check_true("stale/unflagged answer stays clean", "low_confidence" not in clean_res[0]["answers"]["q"])

# 2. apply_confidence_gate clears stale flag and updates abstention state on re-evaluation
gate_reuse = [{
    "answers": {
        "intent": {"choice": "refund", "answer_confidence": 0.8},
    }
}]
g_ans = gate_reuse[0]["answers"]["intent"]
apply_confidence_gate(gate_reuse, 0.9)
check("stale/gate strict threshold abstains", g_ans["abstention"], GATE_ABSTAINED)
check_true("stale/gate strict threshold sets low_conf", g_ans.get("low_confidence") is True)

apply_confidence_gate(gate_reuse, 0.5)
check("stale/gate permissive threshold passes", g_ans["abstention"], GATE_PASSED)
check_true("stale/gate permissive threshold clears low_conf", "low_confidence" not in g_ans)
check("stale/gate echoes updated threshold", g_ans["abstention_threshold"], 0.5)

apply_confidence_gate(gate_reuse, 0.0)
check("stale/gate 0.0 passes", g_ans["abstention"], GATE_PASSED)
check_true("stale/gate 0.0 clears low_conf", "low_confidence" not in g_ans)

# 3. unusable confidence clears stale low_confidence and marks unevaluated
odd_reuse = [{"answers": {"q": {"choice": "x", "answer_confidence": 0.2, "low_confidence": True}}}]
odd_reuse[0]["answers"]["q"]["answer_confidence"] = float("nan")
apply_confidence_gate(odd_reuse, 0.5)
check("stale/unusable conf marks unevaluated", odd_reuse[0]["answers"]["q"]["abstention"], GATE_UNEVALUATED)
check_true("stale/unusable conf clears stale low_conf", "low_confidence" not in odd_reuse[0]["answers"]["q"])

# --------------------------------------------------------------- the operator's three questions
# End to end through a real call site, weight-free. `decide` forwards to any runner with a
# `predict`, so this exercises `laya/structured.py`'s own gate call site rather than the helper
# in isolation -- the six call sites are where the state used to be skipped entirely.
from laya.structured import decide  # noqa: E402

_GATE_Q = {"type": "choice", "instructions": "Pick one.",
           "criteria": {"a": "alpha", "b": "beta"}}
CONFIDENCES = {"high": 0.95, "low": 0.40}
QUESTIONS = {qid: dict(_GATE_Q) for qid in CONFIDENCES}


class _StubRunner:
    def __init__(self, conf):
        self.conf = conf

    def predict(self, state, questions, **kwargs):
        return {"model": "stub", "answers": {
            qid: {"type": "choice", "choice": "a", "answer_confidence": self.conf[qid]}
            for qid in questions}}


stub = _StubRunner(CONFIDENCES)

# 1. Was the gate in effect at all? Before, both answers were byte-identical to the gated pair
#    below, so the answer was "no way to tell". Now the ungated run simply has no `abstention`
#    key, and the gated one does -- which is how a caller tells them apart.
ungated = decide(stub, "state", questions=QUESTIONS)
for qid in CONFIDENCES:
    check_true("workflow/ungated %s reports no state" % qid, "abstention" not in ungated[qid])

gated = decide(stub, "state", questions=QUESTIONS, min_confidence=0.80)
check("workflow/the cleared answer reports passed", gated["high"]["abstention"], GATE_PASSED)
check("workflow/the low answer reports abstained", gated["low"]["abstention"], GATE_ABSTAINED)

# 2. What fraction abstained, over whatever decisions the operator kept? Computable from the
#    artifact, with no re-run and no out-of-band record of which threshold was used.
log = [gated, decide(stub, "other", questions=QUESTIONS, min_confidence=0.80)]
answers = [a for d in log for a in d.values()]
abstained = sum(a["abstention"] == GATE_ABSTAINED for a in answers)
check("workflow/abstention rate over the log", abstained, 2)
check("workflow/…out of", len(answers), 4)
check_true("workflow/…and no answer is left unlabelled", all("abstention" in a for a in answers))

# 3. What threshold produced these? `flag_low_confidence` consumes it and drops it, so before
#    this a batch that gated different request classes differently could not be re-split.
check("workflow/the threshold is recoverable", gated["low"]["abstention_threshold"], 0.80)
check_true("workflow/…and a per-class mix is re-splittable",
           {a["abstention_threshold"] for a in gated.values()} == {0.80})

# The schema projection is unchanged -- an abstained field is still `null` -- so the flat
# `values` mapping still cannot say why. `return_details=True` is where the state is readable,
# and it was unreadable before.
SCHEMA = {"type": "object",
          "properties": {"high": {"type": "string", "enum": ["a", "b"]},
                         "low": {"type": "string", "enum": ["a", "b"]}}}
projected = decide(stub, "state", schema=SCHEMA, min_confidence=0.80)
check("workflow/the schema projection is unchanged", projected["low"], None)
check("workflow/…and still reports the cleared field", projected["high"], "a")
detailed = decide(stub, "state", schema=SCHEMA, min_confidence=0.80, return_details=True)
check("workflow/return_details carries the state",
      detailed.answers["low"]["abstention"], GATE_ABSTAINED)

# --------------------------------------------------------------- exported
import laya  # noqa: E402

check_true("export/answer_confidence is importable from laya", hasattr(laya, "answer_confidence"))
check_true("export/answer_confidence is in __all__", "answer_confidence" in laya.__all__)
check_true("export/answer_confidence is in dir()", "answer_confidence" in dir(laya))
check_true("export/check_min_confidence is importable from laya", hasattr(laya, "check_min_confidence"))
check_true("export/check_min_confidence is in __all__", "check_min_confidence" in laya.__all__)
check_true("export/check_min_confidence is in dir()", "check_min_confidence" in dir(laya))
check_true("export/flag_low_confidence is importable from laya", hasattr(laya, "flag_low_confidence"))
check_true("export/flag_low_confidence is in __all__", "flag_low_confidence" in laya.__all__)
check_true("export/flag_low_confidence is in dir()", "flag_low_confidence" in dir(laya))
check_true("export/apply_confidence_gate is importable from laya", hasattr(laya, "apply_confidence_gate"))
check_true("export/apply_confidence_gate is in __all__", "apply_confidence_gate" in laya.__all__)
check_true("export/apply_confidence_gate is in dir()", "apply_confidence_gate" in dir(laya))
check_true("export/GATE_STATES is importable from laya", hasattr(laya, "GATE_STATES"))
check_true("export/GATE_STATES is in __all__", "GATE_STATES" in laya.__all__)
check_true("export/GATE_STATES is in dir()", "GATE_STATES" in dir(laya))
# The three values are reached through GATE_STATES rather than exported one by one: iterating the
# vocabulary is the whole use, and every `laya.__all__` name has to earn a documented entry
# (`tests/test_packaging.py` enforces that).
import laya.confidence as _confidence  # noqa: E402

check("export/GATE_STATES is the vocabulary a caller iterates", list(GATE_STATES),
      [_confidence.GATE_PASSED, _confidence.GATE_ABSTAINED, _confidence.GATE_UNEVALUATED])
check("export/laya re-exports the same tuple", laya.GATE_STATES, _confidence.GATE_STATES)

# ------------------------------------------------- a preset page's conclusions follow their numbers
# `examples/28_presets_moderation.py` prints a summary over `laya.moderation_questions()` answers,
# and the version this section replaces hardcoded three of its conclusions:
#   * `benign -> every flag 0.000, severity 0.24 / 3` printed `toxic` alone, while the same page's
#     table showed the benign post reading 0.066 on `spam`;
#   * ``severity orders the set correctly (1.34 > 1.05 > 0.99 > 0.24)`` put the `>` signs in the
#     format string, so the sentence stayed "correct" whatever the scores came back as;
#   * `the questions that separate them are harassment and threat` named two of four flags, while
#     the gaps measured on that same run were harassment 0.518, spam 0.193, threat 0.054 -- threat
#     was third, and spam was never mentioned.
# The page now builds those sentences from pure functions over its answers. No weights are loaded:
# the functions are extracted from the example's own AST and exec'd here on fabricated answers, so
# the gate drives the code the page actually runs rather than re-reading its prose.
import ast  # noqa: E402
import builtins  # noqa: E402
import re  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_28 = os.path.join(ROOT, "examples", "28_presets_moderation.py")
HELPERS_28 = ("noul_flags", "score_ceiling", "flag_line", "past_half", "gaps", "rank_by_severity")

# The page as it ships on main: the banner's universality claim and the three conclusions, as
# literal source lines. Every ban below is witnessed against this text, and every positive rule
# below is witnessed by showing this text does not satisfy it.
OLD_PAGE = '''
    lights up harassment, a generic one only toxic, spam only spam, and a normal post
    nothing at all.
        label, a["toxic"]["noul"], a["harassment"]["noul"], a["threat"]["noul"],
print("   benign          -> every flag %.3f, severity %.2f / 3"
      % (ben["toxic"]["noul"], ben["severity"]["score"]))
print("   `severity` orders the set correctly (%.2f > %.2f > %.2f > %.2f) but is a coarse 0-3"
print("   posts differ by only %.3f on `toxic`: the questions that separate them are")
print("   `harassment` and `threat`, not `toxic` on its own.")
'''

UNIVERSAL_ONE_FLOAT = re.compile(r"every (?:flag|score|level)[^.]{0,12}%\.\d+f", re.I)
ASSERTED_CHAIN = re.compile(r"%\.2f\s*>\s*%\.2f")
NAMED_SEPARATORS = re.compile(r"separat\w+[^.]{0,60}`(?:toxic|harassment|threat|spam)`"
                              r"[^.]{0,30}`(?:toxic|harassment|threat|spam)`", re.I)
HARDCODED_FLAG_READ = re.compile(r"\[[\"'](?:toxic|harassment|threat|spam)[\"']\]\[[\"']noul[\"']\]")
NOTHING_QUALIFIED = re.compile(r"nothing past 0\.5[^,]{0,10}, not zero", re.I)
DERIVED_FLAGS = re.compile(r"q\[.type.\]\s*==\s*[\"']noul[\"']")
DERIVED_CEILING = re.compile(r"len\(q\[.criteria.\]\)\s*-\s*1")


def _src28():
    with open(EXAMPLE_28, encoding="utf-8") as fh:
        return fh.read()


def _helpers28():
    """Exec only the example's top-level helper functions -- its `load()` call needs weights."""
    tree = ast.parse(_src28())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS_28]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), EXAMPLE_28, "exec"), ns)  # noqa: S102
    return ns


def _called28():
    """Names the example's top-level *statements* use, so a helper cannot pass by being dead."""
    tree = ast.parse(_src28())
    top = [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return {node.id for stmt in top for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _a(toxic, harassment, threat, spam, severity):
    return {"toxic": {"noul": toxic}, "harassment": {"noul": harassment},
            "threat": {"noul": threat}, "spam": {"noul": spam},
            "severity": {"score": severity}}


def test_preset_is_four_flags_and_one_rubric():
    """The docstring and the example's banner both claim this shape; nothing asserted it."""
    mq = laya.moderation_questions()
    assert list(mq) == ["toxic", "harassment", "threat", "spam", "severity"], list(mq)
    assert {k: q["type"] for k, q in mq.items()} == {
        "toxic": "noul", "harassment": "noul", "threat": "noul", "spam": "noul",
        "severity": "score"}
    assert len(mq["severity"]["criteria"]) == 4, len(mq["severity"]["criteria"])


def _free_names(node):
    """Names a function reads that it neither binds locally nor takes as an argument."""
    bound = {a.arg for fn in [node] + [k for k in ast.walk(node) if isinstance(k, ast.Lambda)]
             for a in fn.args.args}
    bound |= {n.id for n in ast.walk(node)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound}


def test_helpers_are_live_and_pure():
    """Each helper is called by the page, takes its whole world as arguments, and returns."""
    ns = _helpers28()
    got = sorted(k for k in ns if not k.startswith("__"))
    assert got == sorted(HELPERS_28), "example 28 lost a helper: %s" % got
    unused = set(HELPERS_28) - _called28()
    assert not unused, "defined but never called at module level: %s" % sorted(unused)
    for name in HELPERS_28:
        node = _fn_node(name)
        outside = sorted(n for n in _free_names(node) if not hasattr(builtins, n))
        assert not outside, "%s reads %s from the page, so this gate cannot drive it" % (name, outside)
        prints = [c for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "print"]
        assert not prints, "%s prints instead of returning, so the summary is not checkable" % name


def _fn_node(name):
    for n in ast.parse(_src28()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def test_flag_line_prints_every_flag():
    ns = _helpers28()
    flags = ns["noul_flags"](laya.moderation_questions())
    assert flags == ["toxic", "harassment", "threat", "spam"], flags
    line = ns["flag_line"](_a(0.0, 0.0, 0.0, 0.066, 0.24), flags)
    assert "0.066" in line, "the non-zero flag vanished: %r" % line
    assert re.findall(r"`([a-z]+)`", line) == flags, line
    assert len(re.findall(r"\d\.\d{3}", line)) == 4, "one number per flag: %r" % line


def test_past_half_uses_describes_cut():
    """`_common.describe` labels a noul true at `> 0.5`; the page must agree with it exactly."""
    ns = _helpers28()
    flags = ns["noul_flags"](laya.moderation_questions())
    assert ns["past_half"](_a(0.0, 0.0, 0.0, 0.066, 0.24), flags) == []
    assert ns["past_half"](_a(0.587, 0.662, 0.142, 0.049, 1.34), flags) == ["toxic", "harassment"]
    assert ns["past_half"](_a(0.5, 0.5, 0.5, 0.5, 1.0), flags) == [], "0.5 reads false, not true"


def test_gaps_names_the_widest_separators():
    """The old page said harassment and threat; on these numbers it is harassment and spam."""
    ns = _helpers28()
    flags = ns["noul_flags"](laya.moderation_questions())
    tgt, gen = _a(0.587, 0.662, 0.142, 0.049, 1.34), _a(0.584, 0.144, 0.088, 0.242, 1.05)
    g = ns["gaps"](tgt, gen, flags)
    assert [k for k, _ in g] == ["harassment", "spam", "threat", "toxic"], g
    assert abs(dict(g)["harassment"] - 0.518) < 1e-9 and dict(g)["spam"] < 0, g
    # and the name follows the data: move `threat` to the top and it is the separator.
    assert ns["gaps"](_a(0.1, 0.1, 0.9, 0.1, 1), _a(0.1, 0.1, 0.1, 0.1, 1), flags)[0][0] == "threat"


def test_rank_cannot_assert_its_own_order():
    ns = _helpers28()
    labels = ["a", "b", "c", "d"]
    scores = {"a": 1.34, "b": 1.05, "c": 0.99, "d": 0.24}
    results = {k: _a(0, 0, 0, 0, v) for k, v in scores.items()}
    chain, in_order = ns["rank_by_severity"](results, labels)
    assert in_order is True, chain
    assert [float(x) for x in re.findall(r"\d+\.\d{2}", chain)] == [1.34, 1.05, 0.99, 0.24], chain

    shuffled = {k: _a(0, 0, 0, 0, scores[j]) for k, j in
                zip(labels, ["d", "a", "c", "b"])}
    chain2, in_order2 = ns["rank_by_severity"](shuffled, labels)
    assert in_order2 is False, "the verdict must flip when the ranking does: %s" % chain2
    assert [float(x) for x in re.findall(r"\d+\.\d{2}", chain2)] == [1.34, 1.05, 0.99, 0.24], (
        "the printed chain must descend, whatever the order it lists: %s" % chain2)
    assert chain2.index("b") < chain2.index("a"), chain2


def test_page_drops_the_hardcoded_conclusions():
    """Each ban fires on main's page and not on this one -- a ban with no witness is a guess."""
    src = _src28()
    for name, rule in (("every flag %.3f", UNIVERSAL_ONE_FLOAT),
                       ("%.2f > %.2f in a format string", ASSERTED_CHAIN),
                       ("separate them are `harassment` and `threat`", NAMED_SEPARATORS),
                       ('a["toxic"]["noul"] in the summary', HARDCODED_FLAG_READ)):
        assert rule.search(OLD_PAGE), "%s does not fire on the wording it bans" % name
        assert not rule.search(src), "%s is still in example 28" % name


def test_page_qualifies_the_banner_and_derives_its_numbers():
    """The positive half: the page must read its flag list and rubric size from the preset."""
    src = _src28()
    assert NOTHING_QUALIFIED.search(src), "the banner must say that `nothing` is not zero"
    assert not re.search(r"nothing at all", src, re.I), "the unqualified claim is back"
    assert DERIVED_FLAGS.search(src), "flags must be read off the preset, not typed"
    assert DERIVED_CEILING.search(src), "the rubric ceiling must be read off the criteria"
    assert not re.search(r"/ 3\b", src), "a hardcoded severity divisor is back"
    # and main's page satisfies none of that, so these rules could not have passed before.
    assert not NOTHING_QUALIFIED.search(OLD_PAGE)
    assert re.search(r"nothing at all", OLD_PAGE, re.I)
    assert not DERIVED_FLAGS.search(OLD_PAGE) and not DERIVED_CEILING.search(OLD_PAGE)


# ------------------------------- example 03 must give `confidence` separately per question type
# `examples/03_reading_the_result.py` is the page a reader comes back to "when writing your own
# glue code", and its field tour closed with "`confidence` is 1 minus normalised entropy -- a scale
# that moves with the option count on the same answer". `Agent._decode_answers` runs two formulas
# and picks by question type (`laya/agent.py:1366-1402`):
#   * `choice` and `score`  ->  `round(confidence_from_probs(p, k), 4)` = `1 - H(p) / log(k)`
#   * `noul`                ->  `round(max(float(p[1]), 1.0 - float(p[1])), 4)`
# The page calls all three types in one `predict()`, so no single sentence covers `confidence` --
# and the proof is in the run the page itself prints: a `noul` answer reports the same number in
# `confidence` and `answer_confidence`, which a normalized entropy of a two-option distribution
# never equals (0.9/0.1 reads 0.531 as entropy, 0.900 as max(p)). Same defect class as examples
# 40/18/30 and as `DecisionResult`'s docstring, on the page a beginner reads first. The two pages
# that still carry the blanket sentence (`docs/structured.md`, `docs/questions-and-answers.md`) are
# left for a follow-up.
EXAMPLE_03 = os.path.join(ROOT, "examples", "03_reading_the_result.py")
AGENT_PY = os.path.join(ROOT, "laya", "agent.py")
ONNX_PY = os.path.join(ROOT, "laya", "onnx_agent.py")

# The wording this section bans, as it ships on main. Every ban below is witnessed against it, and
# every positive rule below is witnessed by showing that it does not satisfy the rule.
OLD_PAGE_03 = """
   `confidence` is 1 minus normalised entropy -- a scale that moves with the option
   count on the same answer. `answer_confidence` is max(p), the probability mass on
   the answer being reported, and it is the field to gate on (example 18 routes at a
   threshold on it).
"""

BAN_UNIFORM_ENTROPY = re.compile(r"`confidence` is 1 minus normalised entropy -- a scale", re.I)
NAMES_ENTROPY_TYPES = re.compile(r"normalised entropy on a `choice` or `score` answer", re.I)
NAMES_NOUL_FORMULA = re.compile(r"max\(p\[1\],\s*1\s*-\s*p\[1\]\)")
ATTRIBUTES_NOUL = re.compile(r"-- on a `noul`\.")
CALLS_SCALES_INCOMPARABLE = re.compile(r"scales are not comparable", re.I)
NAMES_ANSWER_MAXP = re.compile(r"`answer_confidence` is max\(p\),")
GATES_ON_ANSWER = re.compile(r"field to gate on \(example 18", re.I)
NAMES_BINNING_SCOPE = re.compile(
    r"`binning_map` remaps `answer_confidence` and leaves `confidence` alone", re.I)


def _src03(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _printed_text(path):
    """The page's output as one flattened string.

    Every string literal handed to `print()` anywhere in the file, joined and whitespace-collapsed,
    so a sentence the page wraps across four `print()` calls is still one sentence to the rules
    below -- and a claim that survives only because it is split across lines cannot hide from them.
    """
    parts = []
    for node in ast.walk(ast.parse(_src03(path))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            parts.extend(a.value for a in ast.walk(node)
                         if isinstance(a, ast.Constant) and isinstance(a.value, str))
    return " ".join(" ".join(parts).split())


def _confidence_exprs(path):
    """{answer type: source of the expression that becomes its `confidence`}.

    Read off `_decode_answers` through AST, one entry per answer dict the builder constructs, so
    the page is held to the code that produces the field rather than to a comment about it. A value
    that is a local name (the ONNX path pre-computes `conf_score`) is resolved to what it is bound
    to inside the same function.
    """
    src = _src03(path)
    func = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "_decode_answers")
    bound = {}
    for node in ast.walk(func):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            bound[node.targets[0].id] = ast.get_source_segment(src, node.value)
    found = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Dict):
            continue
        literal = {k.value: v for k, v in zip(node.keys, node.values)
                   if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if "type" not in literal or "confidence" not in literal:
            continue
        if not isinstance(literal["type"], ast.Constant):
            continue
        expr = ast.get_source_segment(src, literal["confidence"])
        found[literal["type"].value] = bound.get(expr, expr)
    return found


def test_page_drops_the_single_entropy_formula():
    text = _printed_text(EXAMPLE_03)
    assert BAN_UNIFORM_ENTROPY.search(OLD_PAGE_03), "the ban does not fire on the wording it bans"
    assert not BAN_UNIFORM_ENTROPY.search(
        text), "example 03 again gives `confidence` one formula for all three question types"
    for name, rule in (("the entropy formula is attributed to `choice` and `score`",
                        NAMES_ENTROPY_TYPES),
                       ("the `noul` formula is written out", NAMES_NOUL_FORMULA),
                       ("that formula is attributed to `noul`", ATTRIBUTES_NOUL),
                       ("the two scales are called incomparable", CALLS_SCALES_INCOMPARABLE),
                       ("`answer_confidence` is max(p)", NAMES_ANSWER_MAXP),
                       ("the gate field is named with its page", GATES_ON_ANSWER),
                       ("the binning map remaps only `answer_confidence`", NAMES_BINNING_SCOPE)):
        assert rule.search(text), "example 03 no longer states that %s" % name
    # and main's page satisfies none of the new rules, so they could not have passed before.
    old = " ".join(OLD_PAGE_03.split())
    assert not NAMES_ENTROPY_TYPES.search(old) and not NAMES_NOUL_FORMULA.search(old)
    assert not CALLS_SCALES_INCOMPARABLE.search(old) and not NAMES_BINNING_SCOPE.search(old)


def test_page_names_the_formulas_the_agents_build():
    exprs = _confidence_exprs(AGENT_PY)
    assert set(exprs) == {"choice", "score", "noul"}, (
        "the scan reached %s, not all three answer dicts" % sorted(exprs))
    assert all(exprs.values()), "a `confidence` value could not be read: %s" % exprs
    assert "confidence_from_probs" in exprs["choice"], exprs["choice"]
    assert "confidence_from_probs" in exprs["score"], exprs["score"]
    assert "1.0 - float(p[1])" in exprs["noul"], exprs["noul"]
    assert "confidence_from_probs" not in exprs["noul"], (
        "`noul` has switched to entropy -- the page's per-type split is now the stale claim: %s"
        % exprs["noul"])
    text = _printed_text(EXAMPLE_03)
    assert NAMES_ENTROPY_TYPES.search(text) and NAMES_NOUL_FORMULA.search(text), (
        "the page must name both expressions the builder uses")


def test_page_cites_the_numbers_the_repo_functions_produce():
    """The earned-numbers arm: the pairs the page prints must be what `laya.common` returns."""
    text = _printed_text(EXAMPLE_03)
    for p_true in (0.60, 0.90):
        p = np.array([1.0 - p_true, p_true])
        noul = "%.3f" % max(float(p[1]), 1.0 - float(p[1]))
        entropy = "%.3f" % round(float(confidence_from_probs(p, 2)), 3)
        rule = re.compile(r"%s reads %s[^.]*%s" % (re.escape(noul), re.escape(noul),
                                                   re.escape(entropy)))
        assert rule.search(text), (
            "the page must cite p(true)=%s as %s on a `noul` and %s as entropy; both come from "
            "confidence_from_probs above, so a number that drifts fails here" % (noul, noul,
                                                                                entropy))
    # the reason one sentence could not work: on a `noul` the shipped `confidence` *is* max(p), the
    # same quantity `answer_confidence` reports, which an entropy cannot be.
    for p_true in (0.60, 0.90):
        p = np.array([1.0 - p_true, p_true])
        assert close(max(float(p[1]), 1.0 - float(p[1])), answer_confidence(p, 2)), (
            "the page's `noul` answer would no longer carry the same number in both fields")


def _binning_targets(path):
    """What the installed binning map is applied to inside `_decode_answers`.

    The page closes its tour with "`binning_map` remaps `answer_confidence` and leaves `confidence`
    alone", which is a claim about the builder, not about prose: the map's input is the raw
    `answer_confidence`, and the two `confidence` expressions sit outside that branch.
    """
    src = _src03(path)
    func = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "_decode_answers")
    return [ast.get_source_segment(src, node.args[0]) for node in ast.walk(func)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "apply_binning_map" and node.args)]


def test_the_remap_reaches_only_answer_confidence():
    for path in (AGENT_PY, ONNX_PY):
        targets = _binning_targets(path)
        assert targets, "%s no longer applies a binning map inside the answer builder" % path
        assert all(t == "ans_raw" for t in targets), (
            "the page says a calibration payload remaps `answer_confidence` and leaves "
            "`confidence` alone, but %s remaps %s" % (path, sorted(set(targets))))


def test_the_two_agents_answer_the_same_way():
    """Parity arm: the ONNX path builds the same two expressions, so one page can be true of both."""
    agent_exprs = _confidence_exprs(AGENT_PY)
    onnx_exprs = _confidence_exprs(ONNX_PY)
    assert set(onnx_exprs) == {"choice", "score", "noul"}, (
        "the ONNX answer builder no longer yields all three types: %s" % sorted(onnx_exprs))
    assert "confidence_from_probs" in onnx_exprs["choice"], onnx_exprs["choice"]
    assert "confidence_from_probs" in onnx_exprs["score"], onnx_exprs["score"]
    assert onnx_exprs["noul"] == agent_exprs["noul"], (
        "the two agents now disagree on what a `noul` confidence is (%r vs %r), and the page "
        "describes only one of them" % (onnx_exprs["noul"], agent_exprs["noul"]))


for _fn in (test_page_drops_the_single_entropy_formula,
            test_page_names_the_formulas_the_agents_build,
            test_page_cites_the_numbers_the_repo_functions_produce,
            test_the_remap_reaches_only_answer_confidence,
            test_the_two_agents_answer_the_same_way):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-03/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-03/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-03/%s" % _fn.__name__)


for _fn in (test_preset_is_four_flags_and_one_rubric, test_helpers_are_live_and_pure,
            test_flag_line_prints_every_flag, test_past_half_uses_describes_cut,
            test_gaps_names_the_widest_separators, test_rank_cannot_assert_its_own_order,
            test_page_drops_the_hardcoded_conclusions,
            test_page_qualifies_the_banner_and_derives_its_numbers):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-28/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-28/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-28/%s" % _fn.__name__)


# --------------------------------------------- example 40 must not call a raw softmax calibrated
# `examples/40_caching_and_monitoring.py` gates 15 tickets on `answer_confidence`. On main it calls
# that field "the calibrated probability of the answer Laya reports" in both its banner and its
# closing paragraph, and its `bucket()` docstring reads "application policy over a calibrated
# number". `answer_confidence` is max(p) -- the quantity temperature scaling fits and ECE measures
# -- but the README's own Calibration section says that reading holds *only* after temperatures are
# fitted and validated, the shipped checkpoints are over-confident, and the loader emits at load
# time `RuntimeWarning: ... choice:11+=0.10058... -> 0.5. Treat confidence from the affected
# entries as uncalibrated.` The page now reads the temperature each shape was actually scaled by off
# the loaded agent and states the calibration as conditional.
#
# No weights are loaded here: the four helpers are pulled from the example's AST and exec'd against
# a fabricated agent, and `scale_for`'s lookup is checked against core's own `temp_bucket`, so the
# gate drives the code the page runs instead of re-reading its prose.
from laya.common import QTYPES as _QTYPES40, temp_bucket as _temp_bucket  # noqa: E402

EXAMPLE_40 = os.path.join(ROOT, "examples", "40_caching_and_monitoring.py")
HELPERS_40 = ("option_count", "scale_for", "clamped_buckets", "entropy_confidence")
# Names a helper may read from the page's world: the two core symbols it must defer to, and `math`
# for the entropy. Any other non-builtin free name means this gate cannot drive that helper.
CORE_GLOBALS_40 = {"temp_bucket", "QTYPES", "math"}

# main's page, as literal source lines. Every ban is witnessed against this text; every positive
# rule below is witnessed by showing this text does NOT satisfy it.
OLD_PAGE_40 = '''
    the triage preset and gates on `answer_confidence`, the calibrated probability of the
    """Our thresholds, not the model's: application policy over a calibrated number."""
   The monitor gates on `answer_confidence`: the calibrated probability of the answer Laya
   distribution has H/log(k) = %.2f, so `confidence` is %.2f while
   """ % (entropy, 1 - entropy, vague["intent"]["answer_confidence"]))
'''

UNCONDITIONAL_CALIBRATION = re.compile(r"the calibrated probability", re.I)
POLICY_OVER_CALIBRATED = re.compile(r"over a calibrated number", re.I)
SUBSTITUTED_CONFIDENCE = re.compile(r"1\s*-\s*entropy")
DERIVES_SCALING = re.compile(r"temp_bucket\(")
READS_REPORTED_CONFIDENCE = re.compile(r"\[[\"']confidence[\"']\]")
HARDCODED_BUCKET = re.compile(r"[\"'](choice|score|noul):[0-9]")


def _src40():
    with open(EXAMPLE_40, encoding="utf-8") as fh:
        return fh.read()


def _helpers40():
    """Exec only the example's top-level helpers -- its `load()` call needs weights."""
    tree = ast.parse(_src40())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS_40]
    ns = {"temp_bucket": _temp_bucket, "QTYPES": _QTYPES40, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), EXAMPLE_40, "exec"), ns)  # noqa: S102
    return ns


def _called40():
    tree = ast.parse(_src40())
    top = [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return {node.id for stmt in top for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _fn40(name):
    for n in ast.parse(_src40()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


class _FakeAgent(object):
    """Only the three temperature attributes `scale_for` and `clamped_buckets` read."""

    def __init__(self, by_options, per_type, raw=None):
        self.temperature_by_options = by_options
        self.temperature = per_type
        self.temperature_by_options_raw = dict(by_options) if raw is None else raw


def test_page40_helpers_are_live_defer_to_core():
    defined = sorted(n.name for n in ast.parse(_src40()).body
                     if isinstance(n, ast.FunctionDef) and n.name in HELPERS_40)
    assert defined == sorted(HELPERS_40), "example 40 lost a helper: %s" % defined
    ns = _helpers40()
    unused = set(HELPERS_40) - _called40()
    assert not unused, "defined but never called at module level: %s" % sorted(unused)
    for name in HELPERS_40:
        node = _fn40(name)
        outside = sorted(n for n in _free_names(node)
                         if not hasattr(builtins, n) and n not in CORE_GLOBALS_40)
        assert not outside, "%s reads %s from the page, so this gate cannot drive it" % (name, outside)
        prints = [c for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "print"]
        assert not prints, "%s prints instead of returning" % name
    # `scale_for` must resolve the bucket through core, not carry a typed-in table.
    assert not HARDCODED_BUCKET.search(ast.dump(_fn40("scale_for"))), (
        "scale_for hardcodes a bucket name instead of calling temp_bucket")


def test_scale_for_replays_core_lookup():
    """For every shape, scale_for returns exactly `temperature_by_options.get(bucket, per_type)`."""
    ns = _helpers40()
    per_type = [7.0, 8.0, 9.0]                       # choice, score, noul fallbacks
    by_options = {"choice:6-10": 1.0, "noul:2": 1.9834, "score:3-5": 1.2514}
    fake = _FakeAgent(by_options, per_type)
    for qtype, k in (("choice", 6), ("noul", 2), ("score", 4), ("noul", 2)):
        name, applied, in_map = ns["scale_for"](fake, qtype, k)
        bucket = _temp_bucket(_QTYPES40[qtype], k)
        assert name == bucket, (qtype, k, name, bucket)
        assert in_map == (bucket in by_options), (bucket, in_map)
        assert abs(applied - by_options.get(bucket, per_type[_QTYPES40[qtype]])) < 1e-9, (
            "%s: %r not core's get(...)" % (bucket, applied))
    # the fallback arm: an unmapped shape must fall to the per-type temperature, flagged not-in-map.
    name, applied, in_map = ns["scale_for"](_FakeAgent({}, [0.5, 0.6, 0.7]), "choice", 12)
    assert (name, applied, in_map) == ("choice:11+", 0.5, False), (name, applied, in_map)
    # a mapped temperature of exactly 1.0 must be reported as 1.0, not nudged.
    assert ns["scale_for"](_FakeAgent({"choice:6-10": 1.0}, [1.0, 1.0, 1.0]), "choice", 6) == (
        "choice:6-10", 1.0, True)


def test_option_count_matches_the_shipped_preset():
    ns = _helpers40()
    counts = {qid: ns["option_count"](q) for qid, q in laya.triage_questions().items()}
    assert counts == {"intent": 6, "is_urgent": 2, "frustration": 4,
                      "refund_requested": 2, "churn_risk": 2}, counts


def test_clamped_buckets_names_only_a_shipped_value_the_runtime_refused():
    ns = _helpers40()
    # `choice:11+` ships below TEMP_MIN and is clamped to 0.5; everything else ships unchanged.
    applied = {"choice:11+": 0.5, "noul:2": 1.9834}
    raw = {"choice:11+": 0.10058280825614929, "noul:2": 1.9834}
    out = ns["clamped_buckets"](_FakeAgent(applied, [0.5, 0.5, 0.5], raw))
    assert out == ["choice:11+=0.1006 -> 0.5000"], out
    # and an honest checkpoint reports nothing.
    same = {"noul:2": 1.9834}
    assert ns["clamped_buckets"](_FakeAgent(same, [0.5, 0.5, 0.5], dict(same))) == []


def test_entropy_confidence_recomputes_core_exactly():
    ns = _helpers40()
    for dist in ([0.5, 0.3, 0.2], [0.9, 0.1], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0]):
        p = np.array(dist, dtype=float)
        got = ns["entropy_confidence"](dist)
        want = confidence_from_probs(p, len(dist))
        assert close(got, want, 1e-9), (dist, got, want)
    assert ns["entropy_confidence"]([1.0]) == 1.0, "k<2 is 1.0 by definition"


def test_page40_drops_the_unconditional_calibration_claim():
    """Each ban fires on main's page and not on this one -- a ban with no witness is a guess."""
    src = _src40()
    for name, rule in (("'the calibrated probability'", UNCONDITIONAL_CALIBRATION),
                       ("'over a calibrated number'", POLICY_OVER_CALIBRATED),
                       ("prints `1 - entropy` as the confidence", SUBSTITUTED_CONFIDENCE)):
        assert rule.search(OLD_PAGE_40), "%s does not fire on the wording it bans" % name
        assert not rule.search(src), "%s is still in example 40" % name


def test_page40_reads_the_scaling_and_the_reported_field():
    """The positive half: the page must derive the temperature and print the reported field."""
    src = _src40()
    assert DERIVES_SCALING.search(src), "the bucket must be resolved through core's temp_bucket"
    assert READS_REPORTED_CONFIDENCE.search(src), "the page must read the reported `confidence`"
    # and main's page satisfies none of that, so these rules could not have passed before.
    assert not DERIVES_SCALING.search(OLD_PAGE_40)
    assert not READS_REPORTED_CONFIDENCE.search(OLD_PAGE_40)


for _fn in (test_page40_helpers_are_live_defer_to_core, test_scale_for_replays_core_lookup,
            test_option_count_matches_the_shipped_preset,
            test_clamped_buckets_names_only_a_shipped_value_the_runtime_refused,
            test_entropy_confidence_recomputes_core_exactly,
            test_page40_drops_the_unconditional_calibration_claim,
            test_page40_reads_the_scaling_and_the_reported_field):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-40/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-40/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-40/%s" % _fn.__name__)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all confidence tests passed")
sys.exit(1 if FAIL else 0)
