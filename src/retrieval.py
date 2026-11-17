"""Stage 6 Part A: hybrid text retrieval over the case-document corpus.

Three channels, all deterministic and dependency-light (numpy + the standard library;
sklearn is optional and not required):

* **BM25** -- Okapi BM25 over word tokens. Good at rare names, banks and formats.
* **Dense** -- hashed character n-gram TF-IDF vectors with cosine similarity. Catches
  morphological variants and misspellings that BM25 misses, at zero download cost. The
  embedding is injectable (``embed_fn``) so a real sentence encoder can replace it
  without touching the ranking or evaluation code.
* **Rerank** -- score fusion over the fused candidate list: a weighted combination of
  the channel scores plus structured agreement (query account/bank/row appearing in the
  document's metadata), which is the signal a bag-of-words model cannot see.

Why no transformer embedder: this box has ~7 GB of RAM and the HF CDN (``cdn-lfs``) is
known to fail here, so a 100M-parameter encoder is both unloadable and unnecessary for a
synthetic corpus of a few hundred short documents. The hashed channel is a deliberate
trade: it is reproducible, needs no cache, and its recall is measured, not assumed.

Every ranking is deterministic: ties break on document id, so recall@k is stable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

WORD_RE = re.compile(r"[a-z0-9]+")
N_HASH_BUCKETS = 2 ** 18

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was "
    "were will with this these those".split())


# --- text handling -------------------------------------------------------------------


def words(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed. Deterministic."""
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def char_ngrams(text: str, lo: int = 3, hi: int = 5) -> list[str]:
    """Character n-grams over each word, word-boundary padded (``#word#``)."""
    out: list[str] = []
    for w in words(text):
        padded = f"#{w}#"
        for n in range(lo, hi + 1):
            if len(padded) >= n:
                out.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return out


def _hash_bucket(token: str) -> int:
    # Python's str hash is salted per process; use a stable FNV-1a instead so vectors
    # (and therefore rankings) are identical across runs.
    h = 2166136261
    for ch in token.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h % N_HASH_BUCKETS


@dataclass(frozen=True)
class Document:
    """One case-document. ``meta`` holds structured fields the reranker can match."""
    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def searchable(self) -> str:
        return f"{self.title} {self.text}"


# --- BM25 ----------------------------------------------------------------------------


class BM25:
    """Okapi BM25. Built once over a fixed corpus; ``scores`` is pure numpy."""

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids = [d.doc_id for d in docs]
        self.tokens = [words(d.searchable) for d in docs]
        self.doc_len = np.array([len(t) for t in self.tokens], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if len(docs) else 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        index: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(self.tokens):
            for term, tf in Counter(toks).items():
                index.setdefault(term, []).append((i, tf))
        n = len(docs)
        for term, plist in index.items():
            ids = np.array([p[0] for p in plist], dtype=np.int64)
            tfs = np.array([p[1] for p in plist], dtype=np.float64)
            idf = math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            self.postings[term] = (ids, tfs, idf)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_ids), dtype=np.float64)
        if not self.doc_ids:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len /
                          (self.avg_len if self.avg_len else 1.0))
        for term in set(words(query)):
            hit = self.postings.get(term)
            if hit is None:
                continue
            ids, tfs, idf = hit
            out[ids] += idf * (tfs * (self.k1 + 1.0)) / (tfs + norm[ids])
        return out


# --- hashed char-n-gram TF-IDF ("dense") ---------------------------------------------


class HashedTFIDF:
    """Character-n-gram TF-IDF vectors hashed into a fixed-width space, L2-normalised.

    ``embed_fn`` lets a real sentence encoder be substituted; it must map a string to a
    1-D float vector, and vectors are L2-normalised here either way.
    """

    def __init__(self, docs: Sequence[Document],
                 embed_fn: Callable[[str], np.ndarray] | None = None):
        self.embed_fn = embed_fn
        self.doc_ids = [d.doc_id for d in docs]
        raw = np.stack([self._embed(d.searchable) for d in docs]) if docs \
            else np.zeros((0, N_HASH_BUCKETS))
        self.matrix = raw

    def _embed(self, text: str) -> np.ndarray:
        if self.embed_fn is not None:
            v = np.asarray(self.embed_fn(text), dtype=np.float64)
            denom = np.linalg.norm(v)
            return v / denom if denom else v
        vec = np.zeros(N_HASH_BUCKETS, dtype=np.float64)
        for gram, count in Counter(char_ngrams(text)).items():
            vec[_hash_bucket(gram)] += 1.0 + math.log(count)
        return vec

    def scores(self, query: str) -> np.ndarray:
        if self.matrix.size == 0:
            return np.zeros(0)
        q = self._embed(query)
        denom = np.linalg.norm(q)
        if denom:
            q = q / denom
        return self.matrix @ q


# --- hybrid index + reranker ---------------------------------------------------------


@dataclass
class Hit:
    doc_id: str
    bm25: float
    dense: float
    rerank: float

    def as_dict(self) -> dict:
        return {"doc_id": self.doc_id, "bm25": round(self.bm25, 6),
                "dense": round(self.dense, 6), "rerank": round(self.rerank, 6)}


class HybridIndex:
    """BM25 + hashed-dense fusion, reranked with structured agreement.

    ``alpha`` weights BM25 against dense in the first-stage fusion; ``structured_bonus``
    is the per-match reward the reranker gives when a query term appears in a document's
    metadata (account number, bank, row id). Both are defaults, not tuned on the test
    queries -- tuning on the eval set would defeat the point of measuring it.
    """

    def __init__(self, docs: Sequence[Document], alpha: float = 0.6,
                 structured_bonus: float = 0.25,
                 embed_fn: Callable[[str], np.ndarray] | None = None):
        self.docs = list(docs)
        self.by_id = {d.doc_id: d for d in self.docs}
        self.alpha = alpha
        self.structured_bonus = structured_bonus
        self.bm25 = BM25(self.docs)
        self.dense = HashedTFIDF(self.docs, embed_fn=embed_fn)

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        hi = float(x.max())
        return x / hi if hi > 0 else x

    def rerank_score(self, bm25: np.ndarray, dense: np.ndarray,
                     meta_tokens: set[str]) -> np.ndarray:
        fused = self.alpha * self._norm(bm25) + (1.0 - self.alpha) * self._norm(dense)
        bonus = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            values = {str(v) for v in d.meta.values() if isinstance(v, (str, int))}
            hits = len(meta_tokens & values)
            if hits:
                bonus[i] = self.structured_bonus * min(hits, 3) / 3.0
        return fused + bonus

    def search(self, query: str, k: int = 5,
               candidate_pool: int = 32) -> list[Hit]:
        """Return the top ``k`` documents by reranked score, ties broken by doc_id."""
        if not self.docs:
            return []
        bm = self.bm25.scores(query)
        dn = self.dense.scores(query)
        meta_tokens = {str(v) for v in _structured_terms(query)}
        rr = self.rerank_score(bm, dn, meta_tokens)

        pool = max(k, min(candidate_pool, len(self.docs)))
        # Stable, deterministic pool: sort by rerank desc, then doc_id asc.
        order = sorted(range(len(self.docs)),
                       key=lambda i: (-rr[i], self.docs[i].doc_id))[:pool]
        order = sorted(order, key=lambda i: (-rr[i], self.docs[i].doc_id))[:k]
        return [Hit(self.docs[i].doc_id, float(bm[i]), float(dn[i]), float(rr[i]))
                for i in order]


def _structured_terms(query: str) -> set[str]:
    """Numbers and account-like tokens lifted out of the query text."""
    return {t for t in re.findall(r"[0-9]+", query) if len(t) >= 3}


# --- evaluation ----------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    """One evaluation query. ``relevant`` is a set of doc ids; grading is binary."""
    query_id: str
    text: str
    relevant: frozenset[str]


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        raise ValueError("a query with no relevant documents cannot be scored")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate(index: HybridIndex, queries: Sequence[Query], k: int = 5) -> dict:
    """Recall@k and MRR, scored per query and averaged. No skipped queries.

    ``per_query`` is kept so a single weak query cannot hide behind the mean -- the
    Stage 1 rule about never averaging over skipped folds applies here too.
    """
    if not queries:
        raise ValueError("no queries to evaluate")
    rows = []
    for q in queries:
        ranking = [h.doc_id for h in index.search(q.text, k=max(k, 5))]
        rows.append({
            "query_id": q.query_id,
            "text": q.text,
            "n_relevant": len(q.relevant),
            f"recall_at_{k}": recall_at_k(ranking, q.relevant, k),
            "mrr": reciprocal_rank(ranking, q.relevant),
            "top": ranking[:k],
        })
    return {
        "k": k,
        "n_queries": len(rows),
        f"recall_at_{k}": float(np.mean([r[f"recall_at_{k}"] for r in rows])),
        "mrr": float(np.mean([r["mrr"] for r in rows])),
        "per_query": rows,
    }
