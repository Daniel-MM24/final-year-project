"""Stage 6 Part A: evaluate the hybrid retriever and write the summary artefact.

Loads the built corpus and the hand-labelled query set, verifies every gold document id
actually exists (a gold id that resolves to nothing would silently deflate recall),
runs ``retrieval.evaluate`` at k=5, and writes:

* the aggregate recall@5 and MRR,
* a per-family breakdown (case / reference / distractor-resistance), because the three
  families answer different questions and a single mean hides which one is broken,
* every query that scored below 1.0 recall, in full, so a failure can be read rather
  than guessed at.

Run with::

    /home/dan/miniconda3/envs/learn_env/bin/python src/eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import retrieval as rt  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "corpus" / "documents.jsonl"
QUERIES_PATH = PROJECT_ROOT / "corpus" / "queries.jsonl"
OUT_DIR = PROJECT_ROOT / "results" / "retrieval"
OUT_PATH = OUT_DIR / "summary.json"
LABEL_PATH = PROJECT_ROOT / "corpus" / "corpus_label.json"

K = 5


def load_documents() -> list[rt.Document]:
    docs = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        meta = dict(raw.get("meta") or {})
        # Structured fields the reranker can match against: case documents carry the
        # account, bank and row ids a query is likely to quote.
        for key in ("account", "counterparty", "from_bank", "to_bank", "row"):
            if key in meta:
                meta[key] = str(meta[key])
        docs.append(rt.Document(raw["doc_id"], raw["title"], raw["text"], meta))
    return docs


def load_queries() -> list[rt.Query]:
    return [rt.Query(raw["query_id"], raw["text"], frozenset(raw["relevant"]))
            for raw in map(json.loads, QUERIES_PATH.read_text(
                encoding="utf-8").splitlines())]


def load_families() -> dict[str, str]:
    return {raw["query_id"]: raw["family"] for raw in map(
        json.loads, QUERIES_PATH.read_text(encoding="utf-8").splitlines())}


def check_gold(docs: list[rt.Document], queries: list[rt.Query]) -> None:
    known = {d.doc_id for d in docs}
    missing = [(q.query_id, g) for q in queries for g in q.relevant - known]
    if missing:
        raise SystemExit(f"gold ids not in corpus: {missing}")


def by_family(per_query: list[dict], families: dict[str, str]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_query:
        groups[families[row["query_id"]]].append(row)
    out = {}
    for name, rows in groups.items():
        out[name] = {
            "n_queries": len(rows),
            f"recall_at_{K}": round(sum(r[f"recall_at_{K}"] for r in rows) / len(rows), 4),
            "mrr": round(sum(r["mrr"] for r in rows) / len(rows), 4),
        }
    return dict(sorted(out.items()))


def main() -> int:
    docs = load_documents()
    queries = load_queries()
    families = load_families()
    check_gold(docs, queries)

    index = rt.HybridIndex(docs)
    report = rt.evaluate(index, queries, k=K)

    # Channel ablations: how much does each channel actually contribute?
    class _BM25Only:
        def search(self, text, k=5, candidate_pool=32):
            scores = index.bm25.scores(text)
            order = sorted(range(len(docs)), key=lambda i: (-scores[i], docs[i].doc_id))
            return [rt.Hit(docs[i].doc_id, float(scores[i]), 0.0, float(scores[i]))
                    for i in order[:k]]

    class _DenseOnly:
        def search(self, text, k=5, candidate_pool=32):
            scores = index.dense.scores(text)
            order = sorted(range(len(docs)), key=lambda i: (-scores[i], docs[i].doc_id))
            return [rt.Hit(docs[i].doc_id, 0.0, float(scores[i]), float(scores[i]))
                    for i in order[:k]]

    ablations = {
        "bm25_only": rt.evaluate(_BM25Only(), queries, k=K),
        "dense_only": rt.evaluate(_DenseOnly(), queries, k=K),
        "hybrid": report,
    }

    payload = {
        "corpus_sha256": json.loads(LABEL_PATH.read_text())["sha256"],
        "k": K,
        "n_documents": len(docs),
        "n_queries": report["n_queries"],
        f"recall_at_{K}": round(report[f"recall_at_{K}"], 4),
        "mrr": round(report["mrr"], 4),
        "by_family": by_family(report["per_query"], families),
        "ablations": {name: {f"recall_at_{K}": round(r[f"recall_at_{K}"], 4),
                             "mrr": round(r["mrr"], 4)}
                      for name, r in ablations.items()},
        "per_query": report["per_query"],
        "label": ("Synthetic case documents (IBM AML-data transactions rendered as "
                  "evidence packages) plus real public policy text as background; "
                  "results are not real-world evidence of retriever quality."),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"documents {len(docs)}  queries {len(queries)}  k {K}")
    print(f"hybrid     recall@{K} {payload[f'recall_at_{K}']:.4f}  mrr {payload['mrr']:.4f}")
    for name, r in payload["ablations"].items():
        print(f"{name:10s} recall@{K} {r[f'recall_at_{K}']:.4f}  mrr {r['mrr']:.4f}")
    print("\nby family:")
    for name, r in payload["by_family"].items():
        print(f"  {name:11s} n={r['n_queries']:>2}  "
              f"recall@{K} {r[f'recall_at_{K}']:.4f}  mrr {r['mrr']:.4f}")
    weak = [r for r in report["per_query"] if r[f"recall_at_{K}"] < 1.0]
    print(f"\nqueries below perfect recall@{K}: {len(weak)}")
    for r in sorted(weak, key=lambda x: x["mrr"]):
        print(f"  {r['query_id']}  r@{K}={r[f'recall_at_{K}']:.2f} mrr={r['mrr']:.3f}"
              f"  gold={r['n_relevant']}  top={r['top'][:3]}")
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
