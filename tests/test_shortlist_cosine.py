import inspect
import re
import unittest
import numpy as np
from laya.shortlist import _cosine, predict_shortlist, shortlist_choice

class ShortlistCosineTests(unittest.TestCase):
    def test_cosine_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        docs = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        sims = _cosine(v, docs)
        self.assertAlmostEqual(sims[0], 1.0, places=5)
        self.assertAlmostEqual(sims[1], -1.0, places=5)

    def test_cosine_zero_vectors(self):
        v = np.zeros(4, dtype=np.float64)
        docs = np.ones((3, 4), dtype=np.float64)
        sims = _cosine(v, docs)
        self.assertTrue(np.all(sims == 0.0))

    def test_cosine_empty_docs(self):
        v = np.ones(4, dtype=np.float64)
        docs = np.zeros((0, 4), dtype=np.float64)
        sims = _cosine(v, docs)
        self.assertEqual(len(sims), 0)


# A query of [1, 0] against six option vectors, so the whole signed range is in play:
# pos is cosine 1, orth scores 0 by being perpendicular, zero by having no signal, nan
# by being non-finite, n2 is -0.707 and n1 is -1.
VECTORS = {
    "q": [1.0, 0.0],
    "pos": [1.0, 0.0],
    "orth": [0.0, 1.0],
    "zero": [0.0, 0.0],
    "nan": [float("nan"), float("nan")],
    "n2": [-1.0, 1.0],
    "n1": [-1.0, 0.0],
}


class Embed:
    """A stand-in embedder: records every call and answers from a vector table."""

    def __init__(self, vectors=VECTORS):
        self.vectors = vectors
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class FakeAgent:
    def __init__(self):
        self.questions = None

    def predict(self, state, questions, **kwargs):
        self.questions = questions
        return {"model": "fake", "answers": {}}


def choice(labels):
    return {"i": {"type": "choice", "criteria": dict.fromkeys(labels)}}


def ranked(labels, k, vectors=VECTORS):
    # criteria values of None make render_options send the bare label as the option text
    return shortlist_choice("q", dict.fromkeys(labels), Embed(vectors), k)


class ShortlistOrderingTests(unittest.TestCase):
    """What a signed cosine does to top-``k``. Fabricated vectors only: no checkpoint, no Hub.

    ``_cosine`` returns negative similarities, so the ordering the shortlist docstrings
    promise has to be read across the whole ``[-1, 1]`` range, not off 0 alone.
    """

    def test_a_zero_score_outranks_an_earlier_negative(self):
        # n1 is declared first and scores -1; zero and orth both score 0 and pass it.
        self.assertEqual(ranked(["n1", "zero", "orth"], 2), ["zero", "orth"])
        self.assertEqual(ranked(["n1", "pos", "zero", "orth"], 3), ["pos", "zero", "orth"])

    def test_equal_scores_keep_criteria_order(self):
        self.assertEqual(ranked(["zero", "orth", "n1"], 2), ["zero", "orth"])
        self.assertEqual(ranked(["orth", "zero", "n1"], 2), ["orth", "zero"])

    def test_a_non_finite_vector_scores_zero_not_last(self):
        self.assertEqual(ranked(["nan", "orth", "n1"], 2), ["nan", "orth"])

    def test_a_zero_query_keeps_criteria_order(self):
        vectors = dict(VECTORS)
        vectors["q"] = [0.0, 0.0]
        self.assertEqual(ranked(["n1", "pos", "zero", "orth"], 2, vectors), ["n1", "pos"])

    def test_scores_are_signed_not_clamped(self):
        agent = FakeAgent()
        result = predict_shortlist(agent, "q", choice(["pos", "n1", "n2"]), Embed(), k=2)
        meta = result["shortlist"]["i"]
        self.assertEqual(meta["labels"], ["pos", "n2"])
        self.assertEqual(meta["scores"], [1.0, -0.7071067811865475])
        self.assertEqual(list(agent.questions["i"]["criteria"]), ["pos", "n2"])

    def test_return_scores_reports_the_same_pair(self):
        self.assertEqual(shortlist_choice("q", dict.fromkeys(["n1", "pos", "zero"]), Embed(), 2,
                                          return_scores=True),
                         (["pos", "zero"], [1.0, 0.0]))
        self.assertEqual(shortlist_choice("q", dict.fromkeys(["n1", "pos", "zero"]), Embed(), 3,
                                          return_scores=True),
                         (["n1", "pos", "zero"], None))

    def test_passthrough_labels_are_criteria_order(self):
        agent, embed = FakeAgent(), Embed()
        meta = predict_shortlist(agent, "q", choice(["n1", "pos", "zero"]), embed, k=3)["shortlist"]["i"]
        # pos would win any ranking and n1 would lose one, so this is only the declared
        # order: nothing was ranked, and embed_fn never ran.
        self.assertEqual(meta["labels"], ["n1", "pos", "zero"])
        self.assertIsNone(meta["scores"])
        self.assertTrue(meta["passthrough"])
        self.assertEqual(embed.calls, [])
        self.assertEqual(list(agent.questions["i"]["criteria"]), ["n1", "pos", "zero"])


class ShortlistOrderingDocTests(unittest.TestCase):
    """Both docstrings render onto docs/reference/helpers.md, so they are the published
    ordering contract. Each rule is witnessed against main's own wording, kept verbatim
    below, so a rule the old text already satisfied could not be called a fix.
    """

    OLD_TIE = ("Ties keep the earlier label. A zero vector scores 0 and does not outrank a\n"
               "label that came before it.")
    OLD_META = ("``shortlist[qid]`` holds\n"
                "``labels`` (rank order), ``scores`` (cosine, or ``None`` when nothing was\n"
                "dropped), ``k``, ``n``, and ``passthrough``.")

    DOES_NOT_OUTRANK = re.compile(r"does not outrank", re.I)

    def paragraphs(self, fn):
        doc = inspect.getdoc(fn) or ""
        return [" ".join(part.split()) for part in re.split(r"\n\s*\n", doc) if part.strip()]

    def test_the_tie_paragraph_names_the_negative_side(self):
        doc = inspect.getdoc(shortlist_choice) or ""
        self.assertNotRegex(doc, self.DOES_NOT_OUTRANK)
        about = [p for p in self.paragraphs(shortlist_choice) if "outrank" in p]
        self.assertTrue(about, "nothing says what a score of 0 does against a negative one")
        self.assertIn("ties keep the earlier label", about[0].lower())
        self.assertIn("negative", about[0].lower())
        # main's sentence asserts the opposite of the first rule and none of the second.
        self.assertRegex(self.OLD_TIE, self.DOES_NOT_OUTRANK)
        self.assertNotIn("negative", self.OLD_TIE.lower())

    def test_the_metadata_paragraph_names_both_label_orders(self):
        about = [p for p in self.paragraphs(predict_shortlist) if "rank order" in p]
        self.assertTrue(about, "the labels/scores contract has left the docstring")
        self.assertIn("criteria order", about[0].lower())
        self.assertIn("passthrough", about[0])
        self.assertIn("signed cosine", about[0].lower())
        self.assertNotIn("criteria order", self.OLD_META.lower())
        self.assertNotIn("signed", self.OLD_META.lower())

    def test_a_negative_score_really_survives_the_ranking(self):
        # the metadata paragraph promises scores can be negative; prove the pair it
        # describes is what a signed cosine actually returns for a label pointing away.
        self.assertIn("never clamped to 0", " ".join((inspect.getdoc(predict_shortlist) or "").split()))
        scores = predict_shortlist(FakeAgent(), "q", choice(["pos", "n1", "n2"]), Embed(),
                                   k=2)["shortlist"]["i"]["scores"]
        self.assertTrue(any(score < 0.0 for score in scores), scores)


if __name__ == "__main__":
    unittest.main()
