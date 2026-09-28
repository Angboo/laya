"""Confidence validation, calibrated thresholds, and abstention gating helpers (#361).

Pure Python: safe to import without PyTorch so that Router and structured
decisions stay lightweight and free of torch import overhead.
"""
import math
from typing import Any, Dict, List, Optional


def _gate_confidence(answer: Dict[str, Any]) -> Optional[float]:
    """The confidence the abstention gate reads for `answer`, or None if there is not a usable one.

    `answer_confidence` is the calibrated max(p) confidence and is what the gate must read;
    `confidence` is the fallback, and for `choice` and `score` it is normalized entropy on a
    different scale, which must not be compared against the same threshold. A bool, a NaN, an
    infinity or a missing field is not a number the gate can decide on.

    Shared by :func:`flag_low_confidence` and :func:`apply_confidence_gate` so the flag and the
    reported state cannot disagree about which quantity was read.
    """
    conf = answer.get("answer_confidence")
    if conf is None:
        conf = answer.get("confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf):
        return conf
    return None


def check_min_confidence(v: Any) -> float:
    """Validate opt-in abstention threshold `min_confidence` (#361).

    Must be a real number in [0.0, 1.0]. Booleans are rejected (even though `isinstance(True, int)`).
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError("min_confidence must be a float in [0.0, 1.0], got %r" % (v,))
    return float(v)


def flag_low_confidence(results: List[Dict[str, Any]], min_confidence: float) -> None:
    """Opt-in abstention marker (#361): flag answers whose confidence falls below `min_confidence`.

    Reads `answer_confidence` (the calibrated max(p) confidence, invariant to label count k),
    falling back to `confidence` if `answer_confidence` is absent.
    The raw answer and confidence stay intact; `low_confidence: True` is added.
    """
    if min_confidence == 0.0:
        return
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            conf = _gate_confidence(a)
            if conf is not None and conf < min_confidence:
                a["low_confidence"] = True


#: The confidence gate's state, as reported on every answer by :func:`apply_confidence_gate`.
#:
#: `low_confidence` is written only when the gate fires, so its absence cannot tell a caller "no
#: ``min_confidence`` was passed" apart from "a gate ran and this answer cleared it". These four
#: values make the state a thing you read rather than a thing you infer from a missing field.
GATE_NOT_CONFIGURED = "not_configured"  # no threshold was passed, so there is no gate to apply
GATE_PASSED = "passed"                  # a gate ran and this answer's confidence cleared it
GATE_ABSTAINED = "abstained"            # a gate ran and this answer's confidence fell below it
GATE_UNEVALUATED = "unevaluated"        # a gate ran and this answer carried no usable confidence

GATE_STATES = (GATE_NOT_CONFIGURED, GATE_PASSED, GATE_ABSTAINED, GATE_UNEVALUATED)


def apply_confidence_gate(results: List[Dict[str, Any]], min_confidence: Optional[float] = None) -> None:
    """Report the confidence gate's state on every answer, whether or not a gate was configured.

    A gate is a policy, and a policy whose application cannot be observed is not one. This writes
    ``abstention`` -- one of :data:`GATE_STATES` -- onto every answer, plus ``abstention_threshold``
    whenever a gate actually ran. It is what lets a caller answer three questions it otherwise
    cannot:

    * what fraction of decisions abstained, without guessing which of "no gate" and "gate
      cleared it" produced a missing ``low_confidence``;
    * whether the gate was in effect at all on this run;
    * what threshold produced these results, which ``flag_low_confidence`` consumes and drops --
      without it a batch that used different thresholds per request class cannot be re-split.

    ``GATE_UNEVALUATED`` is the case a boolean cannot express: the gate ran and the answer carried
    no usable confidence, so the gate could not decide. Reporting that as a pass is the same lie as
    reporting it as a flag.

    The flag itself stays :func:`flag_low_confidence`'s -- this delegates to it rather than
    re-implementing the rule, so the boolean and the state cannot drift apart. Call this
    unconditionally, once per call, in place of a `if min_confidence is not None:` guard: that
    guard is what leaves a path reporting nothing at all.
    """
    if min_confidence is not None:
        flag_low_confidence(results, min_confidence)
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            if min_confidence is None:
                a["abstention"] = GATE_NOT_CONFIGURED
                continue
            if a.get("low_confidence"):
                a["abstention"] = GATE_ABSTAINED
            elif _gate_confidence(a) is None:
                a["abstention"] = GATE_UNEVALUATED
            else:
                a["abstention"] = GATE_PASSED
            a["abstention_threshold"] = float(min_confidence)
