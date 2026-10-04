#!/usr/bin/env python3
"""Generate the parity fixtures `laya-java` asserts against, by running the real `laya`.

Nothing in `laya-java` may hand-write an expectation that this script can produce. A port's
tokenizer, its sequence budgets and its decode arithmetic all drift silently, and a hand-typed
expectation drifts with them; a generated one cannot. `sdk/typescript/scripts/sync_presets.py`
sets the same precedent for the TypeScript client's presets, including the `--check` gate CI runs.

    gen_fixtures.py             # write laya-java/fixtures/*.json
    gen_fixtures.py --check     # exit 1 if any committed fixture would change

`--check` is what CI runs: it regenerates into memory and diffs, so a change to `laya/` that moves
the contract fails the Java build instead of being discovered by a user.

Phase 0 families (no model weights needed -- these are the tables and the text the model is shown):

  lang_tables.json    every table `laya/lang.py` routes on, and every threshold constant
  presets.json        the preset question dicts, verbatim, as the model receives them
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "fixtures"))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _sorted_set(values):
    """A set as a sorted list: a fixture has to be byte-stable across runs and interpreters."""
    return sorted(values)


def lang_tables():
    """Every table and threshold `laya.lang` decides a route with.

    Read from the imported module, never parsed out of the source: `_SHARED_WORDS`,
    `_EN_ONLY_WORDS` and `_EN_COLLISION_WORDS` are *derived* at import time from `_STOP`, so
    parsing the file would miss the derivation and a port built from it would be subtly wrong.
    """
    from laya import lang

    return {
        "script_ranges": [
            # (name, [(lo, hi), ...]) with the bounds as ints -- a port needs the numbers, not
            # whatever repr the host language gives a range object.
            {"name": name, "ranges": [[int(lo), int(hi)] for lo, hi in ranges]}
            for name, ranges in lang._SCRIPT_RANGES
        ],
        "stop_words": {code: _sorted_set(words) for code, words in sorted(lang._STOP.items())},
        "short_swedish_words": _sorted_set(lang._SHORT_SWEDISH_WORDS),
        "non_en_diacritics": _sorted_set(lang._NON_EN_DIACRITICS),
        "shared_words": _sorted_set(lang._SHARED_WORDS),
        "nordic_overlap_words": _sorted_set(lang._NORDIC_OVERLAP_WORDS),
        "en_only_words": _sorted_set(lang._EN_ONLY_WORDS),
        "en_collision_words": _sorted_set(lang._EN_COLLISION_WORDS),
        "thresholds": {
            name: getattr(lang, name)
            for name in sorted(n for n in dir(lang)
                               if n.isupper() and isinstance(getattr(lang, n), (int, float))
                               and not isinstance(getattr(lang, n), bool))
        },
        "regexes": {
            # The PATTERN TEXT, not a compiled object: a port must translate these deliberately
            # (Python `\w` under `re.UNICODE` is not JavaScript's `\w`, and neither is Java's).
            name: getattr(lang, name).pattern
            for name in ("_WORD", "_IDENTIFIER", "_CODE_LINE", "_JOINED", "_LETTER_RUN")
            if hasattr(lang, name) and hasattr(getattr(lang, name), "pattern")
        },
    }


def presets():
    """The preset question dicts exactly as a caller receives them, and as the model is shown them.

    `laya.presets` builds these; one changed word is a different question and therefore a
    different answer, so a port must generate them rather than retype them.
    """
    from laya import presets as mod

    out = {}
    for name in sorted(n for n in dir(mod) if n.endswith("_questions") or n.endswith("Questions")):
        value = getattr(mod, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, dict):
            out[name] = value
    return out


# A corpus chosen to break a tokenizer port, not to flatter it. Each entry is (id, text); the id
# is the join key between this fixture and the Java test, so it must stay stable.
TOKENIZER_CORPUS = [
    ("empty", ""),
    ("space", " "),
    ("ascii-prose", "We were billed twice for March and want a refund today."),
    ("ascii-punct", "!!! ??? ... --- *** ((())) [[]] {{}} <<>> ~~~ ///"),
    ("digits", "0 1 42 007 3.14159 1,000,000 -17 1e9 0x1F 2026-10-04"),
    ("leading-space", "   leading spaces"),
    ("trailing-space", "trailing spaces   "),
    ("inner-runs", "a  b\t\tc\n\nd\r\ne   f"),
    ("newlines-only", "\n\n\n"),
    ("cjk-zh", "我们三月份被重复收费了两次，请今天退还重复的金额。"),
    ("cjk-ja", "三月に二重請求されました。至急返金してください。"),
    ("cjk-ko", "3월에 두 번 청구되었습니다. 환불해 주세요."),
    ("arabic", "لقد تم محاسبتنا مرتين في شهر مارس، يرجى رد المبلغ المكرر اليوم."),
    ("hebrew", "חויבנו פעמיים במרץ, אנא החזירו את הסכום הכפול."),
    ("devanagari", "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया राशि वापस करें।"),
    ("thai", "เราถูกเรียกเก็บเงินสองครั้งในเดือนมีนาคม"),
    ("cyrillic", "С нас дважды списали оплату в марте, верните деньги."),
    ("greek", "Μας χρέωσαν δύο φορές τον Μάρτιο, θέλουμε επιστροφή."),
    ("mixed-scripts", "Refund 退款 استرداد वापसी now!"),
    # NFC matters: english normalises NFC, multilingual does not. The same grapheme, composed and
    # decomposed, must tokenize the way the real tokenizer says -- not the way a port assumes.
    ("nfc-composed", "caf\u00e9 na\u00efve \u00fcber"),
    ("nfd-decomposed", "cafe\u0301 nai\u0308ve u\u0308ber"),
    ("combining-stack", "a\u0301\u0302\u0303\u0304\u0305"),
    ("emoji", "\U0001f600 \U0001f389 \u2764\ufe0f"),
    ("emoji-zwj", "\U0001f468\u200d\U0001f469\u200d\U0001f466 \U0001f3f3\ufe0f\u200d\U0001f308"),
    ("emoji-skin-tone", "\U0001f44d\U0001f3fd \U0001f64f\U0001f3ff"),
    # U+2581 is the Metaspace replacement itself: a port that substitutes spaces naively will
    # collide with a caller who sent this character on purpose.
    ("metaspace-char", "price\u2581list\u2581here"),
    # Characters absent from the multilingual vocabulary, so `byte_fallback` must fire and emit
    # `<0xNN>` tokens. The curated corpus above contains NOT ONE such character, and a port whose
    # fallback was disabled outright still passed every entry of it while diverging on 25% of a
    # 30,000-string random sweep. The fixture now carries the case the sweep found.
    ("byte-fallback-rare-scripts", "\U00010A00 \u0CF1 \U00016FE0 \U0001E900"),
    ("byte-fallback-in-prose", "refund \U00010A00 now please"),
    ("byte-fallback-adjacent", "\U00010A00\U00010A00\u0CF1"),
    ("zero-width", "a\u200bb\u200cc\u200dd\ufeffe"),
    ("rtl-marks", "a\u202eb\u202cc \u200f\u200e"),
    ("mask-literal-angle", "please <mask> this and <pad> that"),
    ("mask-literal-square", "please [MASK] this and [PAD] that"),
    ("unspaced-long", "a" * 2000),
    ("unspaced-cjk-long", "中" * 1000),
    ("repeated-word", "refund " * 400),
    ("one-char-lines", "\n".join("a" * 500)),
    ("url-ish", "see https://example.com/a/b?c=d&e=f#g and mail to a.b+c@example.co.uk"),
    ("code-ish", "if (x == 1) { return y[0].z(); } // comment"),
    ("tabs-json", '{"a": [1, 2], "b": {"c": null}}'),
]

# Option text long enough that `build_head`'s `max_length=48` truncation at the tokenizer bites.
LONG_OPTION = ("a refund of the duplicate charge together with written confirmation that the "
               "payment method on file has been removed and will not be charged again under any "
               "circumstances whatsoever, including renewals")


def tokenizer_ids():
    """Exact token ids per checkpoint for a corpus designed to break a port.

    Ids, not answers: a tokenizer divergence that changes an answer is almost impossible to
    localise from the answer, and trivial from the ids. Each checkpoint records what it IS as well
    -- model type, pre-tokenizer, normaliser, vocab and merge counts, special ids -- so a fixture
    generated against a different checkpoint cannot be mistaken for a matching one.
    """
    import os

    from laya.common import encode_text

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")
    out = {}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        if not os.path.isdir(root):
            out[name] = {"skipped": "checkpoint not present at %s" % root}
            continue
        # Resolve the tokenizer directory EXACTLY as laya resolves it at run time:
        # `os.path.join(model_dir, "tokenizer")`, falling back to the configured encoder
        # (`onnx_agent.py:156`, `agent.py:235`). Searching for the shortest path containing a
        # `tokenizer.json` is not the same thing and silently picked the wrong directory: the
        # multilingual checkpoint ships `tokenizer.json` at BOTH its root and its `tokenizer/`
        # subdirectory -- byte-identical, same sha256 -- but only the subdirectory carries
        # `tokenizer_config.json`. Loading the root therefore reported cls/sep/mask/pad as all
        # None, and a port built against that fixture would have handled a checkpoint with no
        # special tokens (which does not exist) while missing the real ids: multilingual reuses
        # `<bos>` as CLS and `<eos>` as SEP, which nothing about the tokenizer itself reveals.
        directory = os.path.join(root, "tokenizer")
        if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            found = [os.path.join(d, "tokenizer.json")
                     for d, _subdirs, files in os.walk(root) if "tokenizer.json" in files]
            if not found:
                out[name] = {"skipped": "no tokenizer.json under %s" % root}
                continue
            directory = os.path.dirname(sorted(found, key=len)[0])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(directory)
        raw = json.load(open(os.path.join(directory, "tokenizer.json"), encoding="utf-8"))
        out[name] = {
            "identity": {
                "tokenizer_dir": os.path.relpath(directory, rig),
                "model_type": raw["model"]["type"],
                "vocab_size": len(raw["model"].get("vocab", {})),
                "merges": len(raw["model"].get("merges", [])),
                "pre_tokenizer": (raw.get("pre_tokenizer") or {}).get("type"),
                "normalizer": (raw.get("normalizer") or {}).get("type"),
                "unk_token": raw["model"].get("unk_token"),
            },
            "special_ids": {
                "cls": tok.cls_token_id, "sep": tok.sep_token_id,
                "mask": tok.mask_token_id, "pad": tok.pad_token_id,
                "mask_token": tok.mask_token,
            },
            # add_special_tokens=False everywhere: `build_sequence` adds [CLS]/[SEP] itself, so the
            # contract a port must match is the bare encoding.
            "corpus": {cid: encode_text(tok, text, add_special_tokens=False)["input_ids"]
                       for cid, text in TOKENIZER_CORPUS},
            # the 48-token cap `build_head` applies, and the same text uncapped, so a port cannot
            # pass by slicing after the fact
            "option_truncation": {
                "text": LONG_OPTION,
                "uncapped": encode_text(tok, " " + LONG_OPTION,
                                        add_special_tokens=False)["input_ids"],
                "capped_48": encode_text(tok, " " + LONG_OPTION, add_special_tokens=False,
                                         truncation=True, max_length=48)["input_ids"],
            },
        }
    # The corpus TEXT, once, beside the per-checkpoint ids. A port needs the inputs as well as the
    # expected outputs, and transcribing 37 hostile strings -- 2,000-character runs, lone
    # surrogates' worth of zero-width marks, RTL overrides -- into a second language by hand is
    # exactly how a parity suite ends up asserting against a corpus that is not the one Python
    # measured. Shared across checkpoints because both encode the same inputs.
    out["corpus_text"] = {cid: text for cid, text in TOKENIZER_CORPUS}
    return out


# Sequence cases chosen to exercise every branch of `build_head`'s budget arithmetic and every
# rendering rule, not to read nicely. `MASKS` is substituted per checkpoint so one case can test
# that each model scrubs ITS OWN mask string: the english checkpoint masks with `[MASK]` and the
# multilingual one with `<mask>`, and a port that hard-codes either lets a caller inject option
# markers into the other.
SEQUENCE_CASES = [
    ("choice-basic", "The customer was billed twice in March.",
     {"t": "choice", "ins": "What should we do?",
      "crit": {"refund": "send the money back", "replace": "ship a new unit"}}, {}),
    ("choice-no-description", "Billed twice.",
     {"t": "choice", "ins": "Approve?", "crit": {"yes": None, "no": ""}}, {}),
    ("choice-falsy-criteria", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"zero": 0, "false": False, "none": None}}, {}),
    ("choice-structured-criteria", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {"a": {"desc": "nested", "n": 1}, "b": [1, 2, "x"], "c": 2.5}}, {}),
    ("choice-many-options", "Billed twice.",
     {"t": "choice", "ins": "Pick one of many.",
      "crit": {("opt%02d" % i): ("criterion number %d with some words" % i) for i in range(40)}}, {}),
    ("choice-long-option", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {"short": "ok", "long": " ".join(["verylongcriterion%d" % i for i in range(120)])}}, {}),
    # Two options whose rendered text is identical for the first 48 tokens, so the tokenizer cap
    # collapses them to the SAME span. The marker count still matches the option count, so the
    # guard in `Agent._encode_state` passes and nothing downstream can tell the question lost the
    # ability to name them apart (#538) -- only `options_distinct` records it. The difference has
    # to sit past token 48, which means it must be in the LABEL's tail: differing labels put it at
    # token 1 and the collision never happens, which an earlier version of this case got wrong.
    ("choice-colliding-options", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {("prefix " * 60 + "one"): "d", ("prefix " * 60 + "two"): "d"}}, {}),
    ("score-levels", "The reply was polite but slow.",
     {"t": "score", "ins": "Rate the reply.", "crit": ["terrible", "poor", "fine", "good"]}, {}),
    ("score-structured", "The reply was polite but slow.",
     {"t": "score", "ins": "Rate.", "crit": [{"d": 1}, "ok", 3]}, {}),
    ("noul-default", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "The invoice is from March."}, {}),
    ("noul-criteria", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "Holds?", "crit": {"false": "no it does not", "true": "yes it does"}}, {}),
    ("noul-custom-labels", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "Holds?", "labels": {"false": "nope", "true": "yep"}}, {}),
    ("mask-in-instruction", "Billed twice.",
     {"t": "choice", "ins": "Pick MASKS one MASKS now", "crit": {"a": "x", "b": "y"}}, {}),
    ("mask-in-option", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x MASKS y", "b": "MASKS"}}, {}),
    ("mask-in-state", "state with MASKS inside it",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("long-instruction", "Billed twice.",
     {"t": "choice", "ins": "instruction words " * 300, "crit": {"a": "x", "b": "y"}}, {}),
    ("long-state", "evidence sentence number one. " * 400,
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("long-state-truncate-left", "evidence sentence number one. " * 400,
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {"truncate_left": True}),
    ("state-dict", {"order": 1, "items": ["a", "b"], "n": 2.5, "ok": True, "missing": None},
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("state-list", [1, "a", {"b": 2}, None, True, 1.5],
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    # `serialize_state` and `render_criterion` both go through `json.dumps`, so a port needs
    # Python's NUMBER formatting, not its own language's. Python writes floats with `repr`
    # (shortest round-trip, scientific outside [1e-4, 1e16) with a two-digit exponent) and
    # integers at arbitrary precision; Java's `Double.toString` disagrees on both the threshold
    # and the spelling, and JDK 17's is not even always shortest. These values make the
    # disagreement a token-id difference instead of a latent one.
    ("state-hostile-numbers",
     {"big": 1e16, "small": 1e-05, "edge": 0.0001, "third": 1.0 / 3.0, "huge": 1e23,
      "whole": 2.0, "negzero": -0.0, "max": 1e308, "denormal": 5e-324,
      "bigint": 123456789012345678901234567890, "negint": -7},
     {"t": "choice", "ins": "Pick.", "crit": {"a": 1e16, "b": 1.0 / 3.0}}, {}),
    ("state-empty", "",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("state-unicode", "\u6211\u4eec\u88ab\u91cd\u590d\u6263\u8d39 \U0001f600 \u0644\u0642\u062f \u062a\u0645",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "\u9000\u6b3e", "b": "\u0627\u0633\u062a\u0631\u062f\u0627\u062f"}}, {}),
    ("option-order-permuted", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y", "c": "z"}},
     {"option_order": [2, 0, 1]}),
    ("tiny-budget", "some state here that will not fit at all",
     {"t": "choice", "ins": "a fairly long instruction that cannot fit either",
      "crit": {"a": "alpha", "b": "beta", "c": "gamma"}},
     {"max_len": 24, "head_max_len": 16}),
    ("head-larger-than-max", "state",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}},
     {"max_len": 12, "head_max_len": 192}),
]


def sequences():
    """`build_sequence` output for every rendering and budget branch, per checkpoint.

    Ids AND markers AND stats: a marker that is off by one still produces a plausible answer, and
    the option-collision counter (`options_distinct`) is invisible from the ids alone.
    """
    import copy
    import os

    from laya.common import build_sequence

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")
    out = {}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        directory = os.path.join(root, "tokenizer")
        if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            out[name] = {"skipped": "no tokenizer/ under %s" % root}
            continue
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(directory)
        cfg = json.load(open(os.path.join(root, "rl_agent_config.json"), encoding="utf-8"))
        default_max = cfg.get("max_len", 512)
        default_head = cfg.get("head_max_len", 192)
        cases = {}
        for cid, state, question, kwargs in SEQUENCE_CASES:
            q = copy.deepcopy(question)
            st = state
            # one case per checkpoint substitutes that checkpoint's own mask string
            if isinstance(st, str):
                st = st.replace("MASKS", tok.mask_token)
            q["ins"] = q["ins"].replace("MASKS", tok.mask_token)
            if isinstance(q.get("crit"), dict):
                q["crit"] = {k: (v.replace("MASKS", tok.mask_token) if isinstance(v, str) else v)
                             for k, v in q["crit"].items()}
            max_len = kwargs.get("max_len", default_max)
            head_max_len = kwargs.get("head_max_len", default_head)
            ids, markers, stats, trunc = build_sequence(
                tok, st, q, max_len=max_len, head_max_len=head_max_len,
                option_order=kwargs.get("option_order"),
                truncate_left=kwargs.get("truncate_left", False),
                return_stats=True, return_truncation_stats=True)
            cases[cid] = {
                "state": st, "question": q, "max_len": max_len, "head_max_len": head_max_len,
                "option_order": kwargs.get("option_order"),
                "truncate_left": kwargs.get("truncate_left", False),
                "ids": ids, "markers": markers, "stats": stats, "truncation": trunc,
            }
        out[name] = {
            "config": {"max_len": default_max, "head_max_len": default_head,
                       "temperature": cfg.get("temperature"),
                       "temperature_by_options": cfg.get("temperature_by_options"),
                       "encoder": cfg.get("encoder")},
            "special_ids": {"cls": tok.cls_token_id, "sep": tok.sep_token_id,
                            "mask": tok.mask_token_id, "pad": tok.pad_token_id,
                            "mask_token": tok.mask_token},
            "cases": cases,
        }
    return out


def decode_answers():
    """`Agent._decode_answers` output for hostile logit rows, per checkpoint.

    Driven through the REAL method rather than a transcription of it: the expectations have to come
    from the shipped decoder, including its rounding, its argmax tie-breaking and its clamps.

    Logits are synthesised rather than taken from a forward pass so the case list can aim at the
    branches -- every temperature bucket, a tie at the top, a row whose softmax is exactly uniform,
    magnitudes that would overflow a naive `exp`, and the language override -- and so the fixture
    does not need a 1.2 GB graph to regenerate.
    """
    import math
    import os

    import numpy as np

    from laya.agent import Agent, QTYPES
    from laya.common import clamp_temperature, resolve_lang_temperatures

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         "/Users/nombauser/Documents/nomba-dev/laya-e2e/checkpoints")

    def row(values):
        return np.asarray(values, dtype=np.float32)

    # (id, question, option_order, lang, logits)
    cases = [
        ("choice-2", {"t": "choice", "ins": "i", "crit": {"a": "x", "b": "y"}},
         None, None, [2.0, 1.0]),
        ("choice-2-tied", {"t": "choice", "ins": "i", "crit": {"a": "x", "b": "y"}},
         None, None, [1.0, 1.0]),
        ("choice-4", {"t": "choice", "ins": "i",
                      "crit": {"a": "1", "b": "2", "c": "3", "d": "4"}},
         None, None, [0.5, 2.5, -1.0, 2.5]),
        ("choice-4-permuted", {"t": "choice", "ins": "i",
                               "crit": {"a": "1", "b": "2", "c": "3", "d": "4"}},
         [2, 0, 3, 1], None, [0.5, 2.5, -1.0, 2.5]),
        ("choice-7", {"t": "choice", "ins": "i",
                      "crit": {("o%d" % i): str(i) for i in range(7)}},
         None, None, [0.1 * i for i in range(7)]),
        ("choice-12", {"t": "choice", "ins": "i",
                       "crit": {("o%d" % i): str(i) for i in range(12)}},
         None, None, [(-1.0) ** i * i * 0.3 for i in range(12)]),
        ("choice-20-uniform", {"t": "choice", "ins": "i",
                               "crit": {("o%d" % i): str(i) for i in range(20)}},
         None, None, [0.0] * 20),
        ("choice-huge-magnitude", {"t": "choice", "ins": "i",
                                   "crit": {"a": "1", "b": "2", "c": "3"}},
         None, None, [700.0, 699.0, -700.0]),
        ("choice-tiny-spread", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, None, [1.0, 1.0 + 1e-7]),
        ("choice-lang-override", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "zh-Hans", [2.0, 1.0]),
        ("choice-lang-override-prefix", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "ZH", [2.0, 1.0]),
        ("choice-lang-unknown", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "xx", [2.0, 1.0]),
        ("score-3", {"t": "score", "ins": "i", "crit": ["bad", "ok", "good"]},
         None, None, [0.2, 1.4, 0.9]),
        ("score-6", {"t": "score", "ins": "i", "crit": [str(i) for i in range(6)]},
         None, None, [0.0, 1.0, 2.0, 1.0, 0.0, -1.0]),
        ("score-structured-legend", {"t": "score", "ins": "i", "crit": [{"d": 1}, "ok", 3]},
         None, None, [0.4, 0.4, 0.4]),
        ("score-permuted", {"t": "score", "ins": "i", "crit": ["bad", "ok", "good"]},
         [1, 2, 0], None, [0.2, 1.4, 0.9]),
        ("noul-true", {"t": "noul", "ins": "i"}, None, None, [0.3, 1.9]),
        ("noul-false", {"t": "noul", "ins": "i"}, None, None, [2.4, 0.1]),
        ("noul-balanced", {"t": "noul", "ins": "i"}, None, None, [1.0, 1.0]),
        ("noul-labels", {"t": "noul", "ins": "i",
                         "labels": {"false": "nope", "true": "yep"}}, None, None, [0.3, 1.9]),
    ]

    out = {}
    for name in ("english", "multilingual"):
        path = os.path.join(rig, name, "rl_agent_config.json")
        if not os.path.isfile(path):
            out[name] = {"skipped": "no rl_agent_config.json for %s" % name}
            continue
        cfg = json.load(open(path, encoding="utf-8"))
        # Clamped exactly as the runtime clamps: `choice:11+` ships at 0.1006 on one checkpoint,
        # a ~10x sharpener, and a port that applied the raw value would publish a coin flip as a
        # certainty. The fixture therefore records the CLAMPED table the decoder actually uses.
        temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        by_options = {k: clamp_temperature(v)
                      for k, v in (cfg.get("temperature_by_options") or {}).items()}
        # Neither shipped checkpoint sets `lang_temperatures`, so the override branch would never
        # be exercised by their configs. One is supplied here, through the same resolver the
        # runtime uses, so a port cannot skip the branch and still pass.
        lang_raw = {"zh": {"temperature": [2.0, 1.5, 3.0],
                           "temperature_by_options": {"choice:2": 2.5}}}
        lang_temperatures = resolve_lang_temperatures(lang_raw, temperature)

        shim = Agent.__new__(Agent)
        shim.temperature = temperature
        shim.temperature_by_options = by_options
        shim.lang_temperatures = lang_temperatures
        shim.binning_map = None

        recorded = {}
        for cid, question, option_order, lang, logits in cases:
            k = len(logits)
            q = dict(question)
            if option_order is not None:
                q["option_order"] = option_order
            items = [{"markers": list(range(k))}]
            # a wider logits block than k, so a port that forgets to slice to k columns is caught
            block_width = k + 3
            padded = row(list(logits) + [99.0] * 3).reshape(1, block_width)
            act = row([0.25, -0.5]).reshape(1, 2)
            answers = Agent._decode_answers(shim, padded, act, items, [cid], {cid: q}, 0, lang)
            recorded[cid] = {
                "question": question, "option_order": option_order, "lang": lang,
                "logits": [float(x) for x in logits], "logits_block_width": block_width,
                "act": [0.25, -0.5], "answer": answers[cid],
            }
        out[name] = {
            "config": {"temperature": temperature, "temperature_by_options": by_options,
                       "lang_temperatures_raw": lang_raw},
            "qtypes": QTYPES,
            "cases": recorded,
        }
    return out


FAMILIES = {
    "lang_tables.json": lang_tables,
    "presets.json": presets,
    "tokenizer_ids.json": tokenizer_ids,
    "sequences.json": sequences,
    "decode.json": decode_answers,
}


def render(payload):
    """One canonical serialisation, so `--check` compares content and never formatting.

    `sort_keys` is deliberately OFF. Key order is not formatting here, it is data: a `choice`
    question's options are rendered in the criteria map's INSERTION order, so sorting the keys on
    the way into the fixture asks a different question than the one that was measured, and a state
    dict re-serialised in sorted order tokenizes differently. Sorting them made the committed
    fixture disagree with the library it was generated from, and a port that matched the fixture
    was wrong in exactly the way the fixture existed to prevent. Python dicts preserve insertion
    order and this generator builds them deterministically, so the output is still canonical.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if a committed fixture would change")
    parser.add_argument("--only", action="append", metavar="FILE",
                        help="restrict to one fixture file (repeatable)")
    args = parser.parse_args(argv)

    os.makedirs(FIXTURES, exist_ok=True)
    stale, written = [], []
    for filename, build in sorted(FAMILIES.items()):
        if args.only and filename not in args.only:
            continue
        path = os.path.join(FIXTURES, filename)
        fresh = render(build())
        if args.check:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    current = handle.read()
            except OSError:
                stale.append("%s is missing" % filename)
                continue
            if current != fresh:
                stale.append("%s is stale" % filename)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(fresh)
            written.append("%s (%d bytes)" % (filename, len(fresh)))

    if args.check:
        for line in stale:
            print("gen_fixtures: " + line, file=sys.stderr)
        if stale:
            print("gen_fixtures: run laya-java/scripts/gen_fixtures.py and commit the result",
                  file=sys.stderr)
            return 1
        print("gen_fixtures: fixtures match laya")
        return 0
    for line in written:
        print("wrote " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
