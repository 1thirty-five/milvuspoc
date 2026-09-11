"""
Tests for customcluster.py -- the criterion-driven clustering path.

Runs entirely offline: no Milvus, no Docker, no model download. The embedding
model and the Milvus client are stubbed, and the vectors are synthetic ones with
*known* structure planted in them, so every assertion is about whether the
clustering finds structure that is actually there. That's the only way to test
this without a live corpus: on real embeddings you can eyeball whether a cluster
looks sensible, but you can't assert it.

Run:
    python test_customcluster.py            # all tests
    python test_customcluster.py -v         # one line per test
    python test_customcluster.py TestAnchorMode
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import cluster
import customcluster as cc

DIM = 32
SEED = 7


def unit(a):
    """L2-normalize, matching how loadmilvus.embed stores vectors."""
    a = np.asarray(a, dtype="float32")
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def planted(n_topics=3, per_topic=8, spread=0.2, seed=SEED):
    """Synthetic corpus with known structure: `n_topics` tight blobs.

    Returns (topic centroids, document vectors, true labels). Every clustering
    assertion below is "does the code recover *these* labels", which is a real
    test rather than a snapshot of whatever the code happened to output.
    """
    rng = np.random.default_rng(seed)
    topics = unit(rng.normal(size=(n_topics, DIM)))
    docs = unit(np.repeat(topics, per_topic, axis=0)
                + spread * rng.normal(size=(n_topics * per_topic, DIM)))
    truth = np.repeat(np.arange(n_topics), per_topic)
    return topics, docs, truth


def write_criteria(text):
    """Write a criterion file to a temp path and return it (cleaned up by the OS)."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return handle.name


def quietly(fn, *args, **kwargs):
    """Call `fn`, swallowing its progress output. Returns its result."""
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------
# criteria.md parsing
# --------------------------------------------------------------------------

class TestParsing(unittest.TestCase):

    def test_full_file(self):
        path = write_criteria("""
<!-- A comment block.
# Mode
this heading is inside a comment and must not be seen
-->

# Mode
anchor

# Labels
- pricing: fees, discounts, billing
* legal risk: liability and indemnity
1. onboarding: getting a new customer started
- bare name with no description


# Settings
assign = hard
floor = 0.35
model = minilm
scheme = commercial
preview = 3
""")
        config = cc.parse_criteria(path)
        self.assertEqual(config["mode"], "anchor")
        # Every bullet style is accepted, and `name: description` splits on the
        # first colon only.
        self.assertEqual([n for n, _ in config["labels"]],
                         ["pricing", "legal risk", "onboarding",
                          "bare name with no description"])
        self.assertEqual(config["labels"][0][1], "fees, discounts, billing")
        self.assertEqual(config["labels"][3][1], "")
        self.assertEqual(config["assign"], "hard")
        self.assertAlmostEqual(config["floor"], 0.35)
        self.assertEqual(config["model"], "minilm")
        self.assertEqual(config["scheme"], "commercial")
        self.assertEqual(config["preview"], 3)

    def test_comment_block_is_ignored(self):
        """criteria.md documents itself in HTML comments; none of it may parse."""
        path = write_criteria("""
<!--
# Mode
kmeans
# Labels
- decoy: this must never be read
-->
# Mode
anchor
# Labels
- a: first
- b: second
""")
        config = cc.parse_criteria(path)
        self.assertEqual(config["mode"], "anchor")
        self.assertEqual([n for n, _ in config["labels"]], ["a", "b"])

    def test_anchor_k_is_the_label_count(self):
        """In anchor mode the bucket count is the user's vocabulary, not `k`."""
        path = write_criteria("# Mode\nanchor\n# Labels\n- a: x\n- b: y\n- c: z\n"
                              "# Settings\nk = 99\n")
        self.assertEqual(cc.parse_criteria(path)["k"], 3)

    def test_defaults_when_settings_absent(self):
        path = write_criteria("# Mode\nanchor\n# Labels\n- a: x\n- b: y\n")
        config = cc.parse_criteria(path)
        self.assertEqual(config["assign"], cc.DEFAULT_ASSIGN)
        self.assertEqual(config["floor"], cc.DEFAULT_FLOOR)
        self.assertEqual(config["model"], cc.DEFAULT_MODEL)
        self.assertEqual(config["scheme"], cc.DEFAULT_SCHEME)
        self.assertEqual(config["preview"], cc.DEFAULT_PREVIEW)

    def test_unknown_heading_body_is_skipped(self):
        path = write_criteria("# Mode\nanchor\n# Notes\n- not a label\n"
                              "# Labels\n- a: x\n- b: y\n")
        self.assertEqual([n for n, _ in cc.parse_criteria(path)["labels"]], ["a", "b"])

    def test_shipped_criteria_file_parses(self):
        """The criteria.md in the repo must always be a valid example."""
        shipped = Path(__file__).with_name("criteria.md")
        if not shipped.exists():
            self.skipTest("criteria.md not present")
        config = cc.parse_criteria(shipped)
        self.assertIn(config["mode"], cc.MODES)
        self.assertGreaterEqual(len(config["labels"]), 2)

    def test_bullet_markers(self):
        for line in ("- pricing", "* pricing", "+ pricing", "1. pricing"):
            with self.subTest(line=line):
                self.assertEqual(cc.strip_bullet(line), "pricing")
        # A bare line is left alone, and a decimal isn't mistaken for a marker.
        self.assertEqual(cc.strip_bullet("pricing"), "pricing")
        self.assertEqual(cc.strip_bullet("3.5. of a thing"), "3.5. of a thing")


class TestValidation(unittest.TestCase):
    """Every bad criterion file must fail loudly, naming the fix."""

    def assert_rejects(self, text, *expected_fragments):
        path = write_criteria(text)
        with self.assertRaises(SystemExit) as caught:
            cc.parse_criteria(path)
        message = str(caught.exception)
        for fragment in expected_fragments:
            self.assertIn(fragment, message)

    def test_missing_file(self):
        with self.assertRaises(SystemExit) as caught:
            cc.parse_criteria("no-such-criteria-file.md")
        self.assertIn("not found", str(caught.exception))

    def test_no_mode(self):
        self.assert_rejects("# Labels\n- a: x\n- b: y\n", "No mode", "anchor")

    def test_unknown_mode(self):
        self.assert_rejects("# Mode\nkmeans\n# Labels\n- a: x\n- b: y\n",
                            "Unknown mode", "kmeans")

    def test_anchor_needs_two_labels(self):
        self.assert_rejects("# Mode\nanchor\n# Labels\n- a: x\n", "at least 2")

    def test_setting_without_equals(self):
        self.assert_rejects("# Mode\nanchor\n# Labels\n- a: x\n- b: y\n"
                            "# Settings\nfloor 0.3\n", "key = value")

    def test_floor_out_of_range(self):
        self.assert_rejects("# Mode\nanchor\n# Labels\n- a: x\n- b: y\n"
                            "# Settings\nfloor = 7\n", "between -1 and 1")

    def test_non_numeric_setting(self):
        self.assert_rejects("# Mode\nanchor\n# Labels\n- a: x\n- b: y\n"
                            "# Settings\npreview = lots\n", "must be a int")

    def test_bad_assign(self):
        self.assert_rejects("# Mode\nanchor\n# Labels\n- a: x\n- b: y\n"
                            "# Settings\nassign = fuzzy\n", "seeded", "hard")


# --------------------------------------------------------------------------
# anchor mode
# --------------------------------------------------------------------------

class TestAnchorMode(unittest.TestCase):

    def setUp(self):
        self.topics, self.docs, self.truth = planted()

    def test_anchor_text_includes_description(self):
        """The description carries the signal, so it must reach the embedder."""
        self.assertEqual(cc.anchor_text("Tier 1", "capital adequacy"),
                         "Tier 1: capital adequacy")
        self.assertEqual(cc.anchor_text("Tier 1", ""), "Tier 1")

    def test_embed_anchors_embeds_name_and_description(self):
        seen = {}

        def fake_embed(model, texts):
            seen["texts"] = list(texts)
            return np.zeros((len(texts), DIM), dtype="float32")

        with patch.object(cc, "embed", fake_embed):
            quietly(cc.embed_anchors, "MODEL", [("a", "first"), ("b", "")])
        self.assertEqual(seen["texts"], ["a: first", "b"])

    def test_hard_assign_recovers_planted_topics(self):
        labels, similarity = quietly(cc.assign_anchors, self.docs, self.topics,
                                     "hard", 0.0)
        np.testing.assert_array_equal(labels, self.truth)
        # Returned similarity is the row's cosine to what it was assigned to.
        expected = (self.docs @ self.topics.T).max(axis=1)
        np.testing.assert_allclose(similarity, expected, atol=1e-5)

    def test_seeded_assign_recovers_planted_topics(self):
        labels, _ = quietly(cc.assign_anchors, self.docs, self.topics,
                            "seeded", 0.0)
        np.testing.assert_array_equal(labels, self.truth)

    def test_seeded_preserves_label_order(self):
        """Centroid i must stay the one that started on anchor i.

        This is the whole reason seeded mode can keep the user's names: if
        KMeans reordered the centroids, every cluster would be mislabelled while
        still looking perfectly reasonable. Shuffling the anchors must permute
        the output labels in exactly the same way.
        """
        labels, _ = quietly(cc.assign_anchors, self.docs, self.topics,
                            "seeded", 0.0)
        order = [2, 0, 1]
        shuffled, _ = quietly(
            cc.assign_anchors, self.docs, self.topics[order], "seeded", 0.0)
        inverse = np.argsort(order)
        np.testing.assert_array_equal(shuffled, inverse[labels])

    def test_floor_sends_bad_fits_to_unassigned(self):
        """Junk far from every anchor must not be force-fit into a bucket.

        Tight blobs (spread=0.1) so there is a real gap to assert across: the
        planted rows sit at cosine ~0.8+ from their anchor and the junk below
        ~0.5, so a floor of 0.6 separates them cleanly. At the looser spread the
        other tests use, the two distributions genuinely overlap and no floor
        can split them -- which is the honest behaviour, not a bug.
        """
        topics, docs, truth = planted(spread=0.1)
        rng = np.random.default_rng(99)
        junk = unit(rng.normal(size=(6, DIM)))
        corpus = np.vstack([docs, junk])

        for assign in ("hard", "seeded"):
            with self.subTest(assign=assign):
                loose, _ = quietly(cc.assign_anchors, corpus, topics, assign, 0.0)
                strict, _ = quietly(cc.assign_anchors, corpus, topics, assign, 0.6)
                # Without a floor everything lands somewhere...
                self.assertEqual((loose == cc.UNASSIGNED).sum(), 0)
                # ...with one, exactly the junk drops out and the real rows keep
                # the labels they already had.
                np.testing.assert_array_equal(
                    strict[len(docs):], np.full(len(junk), cc.UNASSIGNED))
                np.testing.assert_array_equal(strict[:len(docs)], truth)

    def test_seeded_floor_measures_the_fitted_centroid(self):
        """Not the original anchor: after the centroids move, that's the wrong ruler.

        The fitted centroid sits in the middle of its members, so rows are on
        average closer to it than to the anchor they started from. Not *every*
        row -- one on the far edge of a cluster can still sit nearer the anchor
        -- so this asserts the aggregate, plus that the two rulers disagree at
        all, which is what would break if the anchor were used by mistake.
        """
        _, similarity = quietly(cc.assign_anchors, self.docs, self.topics, "seeded")
        to_anchor = (self.docs @ self.topics.T).max(axis=1)
        self.assertGreater(similarity.mean(), to_anchor.mean())
        self.assertFalse(np.allclose(similarity, to_anchor, atol=1e-3))

    def test_stats_report_similarity_per_cluster(self):
        labels, similarity = quietly(cc.assign_anchors, self.docs, self.topics,
                                     "hard", 0.0)
        stats = cc.anchor_stats(labels, similarity, 3)
        self.assertEqual(set(stats), {0, 1, 2})
        for text in stats.values():
            self.assertIn("cosine mean", text)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

class TestReporting(unittest.TestCase):

    def render(self, labels, names, preview=2):
        rows = [{"text": f"doc {i}"} for i in range(len(labels))]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cc.print_clusters(rows, labels, names, preview=preview)
        return buffer.getvalue()

    def test_preview_truncates_long_clusters(self):
        output = self.render([0] * 10, ["big"], preview=2)
        self.assertIn("doc 0", output)
        self.assertNotIn("doc 5", output)
        self.assertIn("... and 8 more", output)

    def test_unassigned_is_printed_last_and_named(self):
        labels = [cc.UNASSIGNED, 0, 1]
        output = self.render(labels, ["first", "second"])
        self.assertIn(cc.UNASSIGNED_NAME, output)
        self.assertGreater(output.index(cc.UNASSIGNED_NAME), output.index("second"))

    def test_cluster_names_are_shown(self):
        output = self.render([0, 1], ["pricing", "legal risk"])
        self.assertIn("pricing", output)
        self.assertIn("legal risk", output)


# --------------------------------------------------------------------------
# storage -- what actually lands in Milvus
# --------------------------------------------------------------------------

class TestStoreLabels(unittest.TestCase):
    """store_labels is shared with cluster.py, so both paths are pinned here."""

    ROWS = [
        {"text": "a", "embedding": [0.1, 0.2], "chunk_hash": "h0", "source": None},
        {"text": "b", "embedding": [0.3, 0.4], "chunk_hash": "h1", "source": "d.pdf"},
    ]

    def store(self, **kwargs):
        captured = {}

        def fake_insert(client, data, collection):
            captured["data"] = data
            return len(data)

        with patch.object(cluster, "ensure_collection", lambda *a, **k: None), \
                patch.object(cluster, "insert_batched", fake_insert):
            quietly(cluster.store_labels, None, self.ROWS, [0, 1], **kwargs)
        return captured["data"]

    def test_cluster_py_path_is_unchanged(self):
        """No names/scheme -> byte-identical to what cluster.py always wrote."""
        data = self.store()
        self.assertEqual(set(data[0]), {"text", "embedding", "chunk_hash", "cluster"})
        self.assertNotIn("cluster_name", data[0])

    def test_custom_path_adds_name_and_scheme(self):
        data = self.store(names=["pricing", "legal risk"], scheme="commercial")
        self.assertEqual(data[0]["cluster_name"], "pricing")
        self.assertEqual(data[1]["cluster_name"], "legal risk")
        self.assertEqual({e["cluster_scheme"] for e in data}, {"commercial"})

    def test_cluster_stays_an_int(self):
        """search.py filters and visualize.py's legend sort both depend on it."""
        data = self.store(names=["a", "b"], scheme="s")
        for entry in data:
            self.assertIsInstance(entry["cluster"], int)

    def test_source_is_carried_through_but_never_nulled(self):
        """The rebuild destroys any field left out; a null `source` is not a value."""
        data = self.store(names=["a", "b"], scheme="s")
        self.assertNotIn("source", data[0])
        self.assertEqual(data[1]["source"], "d.pdf")


# --------------------------------------------------------------------------
# end to end, with Milvus and the model stubbed
# --------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.topics, self.docs, self.truth = planted()
        self.rows = [{"text": f"doc {i} (topic {i // 8})",
                      "embedding": self.docs[i].tolist(),
                      "chunk_hash": f"h{i}", "source": None}
                     for i in range(len(self.docs))]
        self.stored = {}

    def fake_embed(self):
        """First three anchor/pole texts land on the planted topics."""
        rng = np.random.default_rng(2)
        table = {}

        def embed(model, texts):
            out = []
            for text in texts:
                if text not in table:
                    i = len(table)
                    base = self.topics[i] if i < 3 else rng.normal(size=DIM)
                    table[text] = unit(base + 0.05 * rng.normal(size=DIM))
                out.append(table[text])
            return np.asarray(out, dtype="float32")
        return embed

    def run_main(self, criteria_text, *argv, model_dim=DIM):
        def fake_store(client, rows, labels, collection="documents",
                       names=None, scheme=None):
            self.stored.update(labels=list(map(int, labels)), names=names,
                               scheme=scheme, rows=len(rows))
            return len(rows)

        path = write_criteria(criteria_text)
        with patch.object(cc, "connect", lambda *a, **k: "CLIENT"), \
                patch.object(cc, "fetch_all", lambda client, **k: self.rows), \
                patch.object(cc, "get_model", lambda name: f"MODEL({name})"), \
                patch.object(cc, "get_dim", lambda model: model_dim), \
                patch.object(cc, "embed", self.fake_embed()), \
                patch.object(cc, "store_labels", fake_store), \
                patch.object(cc, "write_result_labels", lambda *a, **k: None), \
                patch.object(sys, "argv",
                             ["customcluster.py", "--criteria", path, *argv]):
            with redirect_stdout(io.StringIO()) as output:
                cc.main()
        return output.getvalue()

    ANCHOR = ("# Mode\nanchor\n# Labels\n- alpha: first\n- beta: second\n"
              "- gamma: third\n# Settings\nassign = seeded\nfloor = 0\n"
              "scheme = synth\n")

    def test_anchor_run_stores_names_matching_the_planted_topics(self):
        self.run_main(self.ANCHOR)
        self.assertEqual(self.stored["scheme"], "synth")
        self.assertEqual(self.stored["rows"], len(self.rows))
        np.testing.assert_array_equal(self.stored["labels"], self.truth)
        self.assertEqual(self.stored["names"],
                         ["alpha"] * 8 + ["beta"] * 8 + ["gamma"] * 8)

    def test_every_row_gets_an_int_and_a_name(self):
        self.run_main(self.ANCHOR)
        self.assertEqual(len(self.stored["labels"]), len(self.rows))
        self.assertEqual(len(self.stored["names"]), len(self.rows))
        for label, name in zip(self.stored["labels"], self.stored["names"]):
            self.assertIsInstance(label, int)
            self.assertIsInstance(name, str)

    def test_dry_run_writes_nothing(self):
        output = self.run_main(self.ANCHOR, "--dry-run")
        self.assertEqual(self.stored, {})
        self.assertIn("nothing written", output)

    def test_k_override_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_main(self.ANCHOR, "--k", "5")
        self.assertIn("--k is not accepted", str(caught.exception))

    def test_floor_labels_are_named_unassigned(self):
        """A row below the floor must get the name, not an IndexError on names[-1]."""
        criteria = self.ANCHOR.replace("scheme = synth", "scheme = synth\nfloor = 0.99")
        self.run_main(criteria)
        self.assertEqual(set(self.stored["labels"]), {cc.UNASSIGNED})
        self.assertEqual(set(self.stored["names"]), {cc.UNASSIGNED_NAME})

    def test_model_dimension_mismatch_is_caught(self):
        """Wrong model -> the criterion lands in a different space entirely."""
        with self.assertRaises(SystemExit) as caught:
            self.run_main(self.ANCHOR, model_dim=999)
        self.assertIn("999", str(caught.exception))
        self.assertIn("stored vectors are 32", str(caught.exception))

    def test_empty_collection_is_reported(self):
        self.rows = []
        with self.assertRaises(SystemExit) as caught:
            self.run_main(self.ANCHOR)
        self.assertIn("loadmilvus.py", str(caught.exception))

    def test_k_larger_than_the_corpus_is_caught(self):
        self.rows = self.rows[:2]
        with self.assertRaises(SystemExit) as caught:
            self.run_main(self.ANCHOR)
        self.assertIn("Cannot make 3 clusters", str(caught.exception))

    def test_criterion_is_echoed_before_any_work(self):
        """A typo in criteria.md should be visible before the model even loads."""
        output = self.run_main(self.ANCHOR, "--dry-run")
        self.assertIn("alpha: first", output)
        self.assertIn("seeded", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
