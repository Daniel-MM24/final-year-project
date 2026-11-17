"""Stage 6 Part A acceptance test: retrieval is deterministic, correctly scored, labelled.

Four properties, none of which the evaluation itself can establish:

1. **Determinism** -- two independently built indexes rank identically, and the FNV hash
   used for the dense channel is stable across processes (Python's ``hash()`` is not, so
   this test would fail loudly if anyone swapped it back in).
2. **Tie-breaking** -- equal scores fall back to ascending doc_id, so recall@k on a
   scaffolded corpus is reproducible rather than luck-dependent.
3. **Metric correctness** -- ``recall_at_k`` and ``reciprocal_rank`` against hand-worked
   values, including the edge cases (no relevant doc in the list, k larger than the
   ranking, multi-relevant queries).
4. **Corpus labelling** -- every document declares ``synthetic``, every reference
   document carries publisher/URL/date, and every gold id in the query set resolves.

Run with::

    /home/dan/miniconda3/envs/learn_env/bin/python -m pytest tests/test_retrieval.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import retrieval as rt  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "corpus" / "documents.jsonl"
QUERIES_PATH = PROJECT_ROOT / "corpus" / "queries.jsonl"
LABEL_PATH = PROJECT_ROOT / "corpus" / "corpus_label.json"


def _docs(raw: list[dict]) -> list[rt.Document]:
    return [rt.Document(r["doc_id"], r["title"], r["text"], r.get("meta") or {})
            for r in raw]


@pytest.fixture(scope="module")
def raw_docs():
    return [json.loads(l) for l in CORPUS_PATH.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def raw_queries():
    return [json.loads(l) for l in QUERIES_PATH.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def index(raw_docs):
    return rt.HybridIndex(_docs(raw_docs))


# --- 1. determinism ----------------------------------------------------------------


def test_two_builds_rank_identically(raw_docs, index):
    other = rt.HybridIndex(_docs(raw_docs))
    queries = [json.loads(l) for l in QUERIES_PATH.read_text(encoding="utf-8").splitlines()]
    for q in [r["text"] for r in queries]:
        a = [h.doc_id for h in index.search(q, k=5)]
        b = [h.doc_id for h in other.search(q, k=5)]
        assert a == b, q


def test_hash_is_stable_across_processes():
    # Hashed buckets must not depend on PYTHONHASHSEED.
    code = ("import sys; sys.path.insert(0, %r); import retrieval as r; "
            "print([r._hash_bucket(t) for t in ['alpha', 'beta', 'gamma', '#sum#']])"
            % str(PROJECT_ROOT / "src"))
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       check=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin"}).stdout
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"hash buckets differ across processes: {runs}"


# --- 2. tie-breaking ---------------------------------------------------------------

_SCAFFOLD = [
    rt.Document("doc:c", "gamma", "shared phrase indeed"),
    rt.Document("doc:a", "alpha", "shared phrase indeed"),
    rt.Document("doc:b", "beta", "shared phrase indeed"),
]


def test_equal_scores_break_on_doc_id():
    idx = rt.HybridIndex(_SCAFFOLD)
    ids = [h.doc_id for h in idx.search("shared phrase indeed", k=3)]
    assert ids == sorted(ids) == ["doc:a", "doc:b", "doc:c"]


def test_search_respects_k():
    idx = rt.HybridIndex(_SCAFFOLD)
    assert len(idx.search("shared", k=2)) == 2
    assert len(idx.search("shared", k=99)) == 3


# --- 3. metric correctness ---------------------------------------------------------


def test_recall_at_k_hand_values():
    ranking = ["a", "b", "c", "d", "e"]
    assert rt.recall_at_k(ranking, {"a"}, 5) == 1.0
    assert rt.recall_at_k(ranking, {"e"}, 5) == 1.0
    assert rt.recall_at_k(ranking, {"f"}, 5) == 0.0
    # Two relevant, one in the top-2 -> half.
    assert rt.recall_at_k(ranking, {"a", "e"}, 2) == 0.5
    assert rt.recall_at_k(ranking, {"a", "e"}, 5) == 1.0


def test_recall_requires_relevant_set():
    with pytest.raises(ValueError, match="no relevant documents"):
        rt.recall_at_k(["a"], set(), 5)


def test_reciprocal_rank_hand_values():
    ranking = ["a", "b", "c"]
    assert rt.reciprocal_rank(ranking, {"a"}) == 1.0
    assert rt.reciprocal_rank(ranking, {"b"}) == 0.5
    assert rt.reciprocal_rank(ranking, {"c"}) == pytest.approx(1 / 3)
    assert rt.reciprocal_rank(ranking, {"z"}) == 0.0


def test_evaluate_averages_without_skipping():
    idx = rt.HybridIndex(_SCAFFOLD)
    queries = [
        rt.Query("hit", "shared phrase", frozenset({"doc:a"})),
        rt.Query("miss", "shared phrase", frozenset({"nonexistent"})),
    ]
    out = rt.evaluate(idx, queries, k=1)
    assert out["n_queries"] == 2
    # One perfect hit (doc:a first) and one certain miss; the miss is averaged in.
    assert out["recall_at_1"] == 0.5
    assert out["mrr"] == 0.5


def test_evaluate_rejects_empty_query_set(index):
    with pytest.raises(ValueError, match="no queries"):
        rt.evaluate(index, [], k=5)


# --- 4. corpus labelling and gold resolution ---------------------------------------


def test_every_document_declares_synthetic(raw_docs):
    for d in raw_docs:
        assert isinstance(d["meta"].get("synthetic"), bool), d["doc_id"]
        assert d["meta"]["synthetic"] is (d["meta"]["doc_type"] == "case"), d["doc_id"]


def test_reference_documents_carry_provenance(raw_docs):
    for d in raw_docs:
        if d["meta"]["doc_type"] != "reference":
            continue
        for key in ("publisher", "source_title", "source_url", "source_date", "retrieved"):
            assert d["meta"].get(key), f"{d['doc_id']} missing {key}"
        assert d["meta"]["source_url"].startswith("http"), d["doc_id"]


def test_case_documents_say_data_is_synthetic(raw_docs):
    for d in raw_docs:
        if d["meta"]["doc_type"] == "case":
            assert "synthetic" in d["meta"]["data_note"].lower(), d["doc_id"]


def test_gold_ids_resolve(raw_docs, raw_queries):
    known = {d["doc_id"] for d in raw_docs}
    for q in raw_queries:
        assert q["relevant"], f"{q['query_id']} has no gold"
        missing = set(q["relevant"]) - known
        assert not missing, f"{q['query_id']} -> {missing}"


def test_query_ids_unique_and_families_known(raw_queries):
    ids = [q["query_id"] for q in raw_queries]
    assert len(ids) == len(set(ids))
    assert {q["family"] for q in raw_queries} == {"case", "reference", "distractor"}
    assert len(raw_queries) >= 30  # the stage note's floor


def test_label_sha_matches_corpus(raw_docs):
    label = json.loads(LABEL_PATH.read_text())
    assert label["n_documents"] == len(raw_docs)
    assert label["n_synthetic"] + label["n_reference"] == len(raw_docs)
    import hashlib
    assert hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest() == label["sha256"]


def test_no_stub_documents(raw_docs):
    # Short chunks can only ever be false positives; see chunk_sections.
    short = [d["doc_id"] for d in raw_docs if len(d["text"].split()) < 40]
    assert not short, f"stub documents: {short}"
