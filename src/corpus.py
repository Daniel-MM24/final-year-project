"""Stage 6 Part A: build the retrieval corpus.

The corpus mixes two kinds of document, and every document says which it is:

* **Reference (``synthetic: false``)** -- real public policy text dropped into
  ``corpus/reference/``. Extracted with ``pdftotext``, chunked on paragraph boundaries,
  and tagged with the publisher, source URL and retrieval date recorded in
  ``REFERENCE_SOURCES``. This is background/distractor text: it makes ranking realistic,
  because a query about fan-out must not surface a generic typology page above the
  actual case document.
* **Case (``synthetic: true``)** -- one document per Stage 5 evidence package, rendering
  the package's transaction, model and context blocks as prose. These are the retrieval
  targets: queries point at them. The underlying data is synthetic, so these are too.

Both kinds live in one JSONL file with stable, human-readable ``doc_id``s. A separate
``corpus_label.json`` records the mix (counts, chunk sizes, and the reference sources
with URL/date) so nothing downstream can mistake policy prose for transaction evidence.

Build with::

    /home/dan/miniconda3/envs/learn_env/bin/python src/corpus.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "corpus" / "reference"
CORPUS_DIR = PROJECT_ROOT / "corpus"
DOCS_PATH = CORPUS_DIR / "documents.jsonl"
LABEL_PATH = CORPUS_DIR / "corpus_label.json"
EVIDENCE_DIR = PROJECT_ROOT / "results" / "evidence"

CHUNK_WORDS = 350
MIN_CHUNK_WORDS = 40
RETRIEVED = "2026-11-17"

# Publisher, URL and date for each reference file. Keyed by filename; a file with no
# entry still gets a chunked document, but marked ``source_unverified`` so the gap is
# visible rather than silently labelled as policy.
REFERENCE_SOURCES = {
    "Explanatory note for revised R.16.pdf.coredownload.pdf": {
        "publisher": "Financial Action Task Force (FATF)",
        "title": "Explanatory note for revised Recommendation 16 (wire transfers)",
        "url": "https://www.fatf-gafi.org/en/publications/Fatfrecommendations/"
               "R16-explanatory-note.html",
        "date": "2025-06",
        "retrieved": RETRIEVED,
    },
    "SAR-FAQs-October-2025.pdf": {
        "publisher": "FinCEN, Federal Reserve, FDIC, NCUA, OCC",
        "title": "Frequently Asked Questions Regarding Suspicious Activity Reporting "
                 "Requirements",
        "url": "https://www.fincen.gov/resources/statutes-regulations/guidance",
        "date": "2025-10-09",
        "retrieved": RETRIEVED,
    },
}


# --- reference documents -------------------------------------------------------------


def pdf_to_text(path: Path) -> str:
    """Extract text with poppler. Raises if ``pdftotext`` is missing or fails."""
    proc = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {path.name}: {proc.stderr.strip()}")
    return proc.stdout


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, join hard-wrapped lines, drop page furniture."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.splitlines()]
        lines = [ln for ln in lines if ln and not re.fullmatch(r"[-–—\s\d]+", ln)]
        if not lines:
            continue
        joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if joined:
            out.append(joined)
    return out


# Numbered headings start a new section: "12." / "1. Background" in FATF notes, "Q1."
# or "1. Q:" in FinCEN FAQs. Matching these keeps an answer in the same chunk as the
# question it answers, which greedy word packing cannot do.
SECTION_RE = re.compile(r"^(?:\d{1,3}\.|Q\d+[.:)]?)\s")


def chunk_sections(paras: list[str], chunk_words: int = CHUNK_WORDS) -> list[str]:
    """Group paragraphs into sections at numbered headings, then pack each section.

    A section longer than ``chunk_words`` is split by the same greedy packer, so no
    chunk exceeds the target and no heading is orphaned from its body. Text before the
    first heading (title blocks, standing headers) is prepended to the first section
    rather than becoming a chunk of its own.
    """
    if not any(SECTION_RE.match(p) for p in paras):
        return chunk_paragraphs(paras, chunk_words)
    sections: list[list[str]] = [[]]
    for para in paras:
        if SECTION_RE.match(para) and sections[-1]:
            sections.append([])
        sections[-1].append(para)
    chunks: list[str] = []
    carry = ""
    for sec in sections:
        if not sec:
            continue
        for piece in chunk_paragraphs(sec, chunk_words):
            if carry:
                piece, carry = carry + " " + piece, ""
            if len(piece.split()) < MIN_CHUNK_WORDS:
                # No previous chunk to merge into (leading title block), so hold the
                # text and prepend it to whatever comes next.
                if chunks:
                    chunks[-1] = chunks[-1] + " " + piece
                else:
                    carry = piece
                continue
            chunks.append(piece)
    if carry:
        chunks.append(carry)
    return chunks


def chunk_paragraphs(paras: list[str], chunk_words: int = CHUNK_WORDS) -> list[str]:
    """Greedy packing of paragraphs into ~``chunk_words`` word blocks.

    A paragraph is never split, so each chunk starts on a sentence the source began.
    Chunks shorter than ``MIN_CHUNK_WORDS`` fold into the previous chunk instead of
    becoming a stub document that can only ever be a false positive.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in paras:
        n = len(para.split())
        if current and size + n > chunk_words:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(para)
        size += n
    if current:
        chunks.append(" ".join(current))
    # Title pages and short lead-ins produce stubs that can only ever be false-positive
    # hits, so every chunk under the minimum folds into a neighbour: forward when one
    # follows, otherwise backward into the previous chunk.
    merged: list[str] = []
    carry = ""
    for chunk in chunks:
        if len(chunk.split()) < MIN_CHUNK_WORDS:
            if merged:
                merged[-1] = merged[-1] + " " + chunk
            else:
                carry = carry + " " + chunk if carry else chunk
            continue
        merged.append((carry + " " + chunk).strip() if carry else chunk)
        carry = ""
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)
    return merged


def reference_documents(path: Path) -> list[dict]:
    meta = REFERENCE_SOURCES.get(path.name)
    text = pdf_to_text(path)
    paras = _paragraphs(text)
    chunks = chunk_sections(paras)
    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()) \
        .replace("pdf-coredownload", "").strip("-")
    docs = []
    for i, chunk in enumerate(chunks):
        doc_meta = {
            "doc_type": "reference",
            "synthetic": False,
            "family": stem.split("-")[0],
            "chunk_index": i,
        }
        if meta:
            doc_meta.update({
                "publisher": meta["publisher"],
                "source_title": meta["title"],
                "source_url": meta["url"],
                "source_date": meta["date"],
                "retrieved": meta["retrieved"],
            })
        else:
            doc_meta["source_unverified"] = True
        docs.append({
            "doc_id": f"ref:{stem}:{i:03d}",
            "title": (meta["title"] if meta else path.stem) + f" (part {i + 1})",
            "text": chunk,
            "meta": doc_meta,
        })
    return docs


# --- case documents ------------------------------------------------------------------


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _render_package(pkg: dict) -> str:
    """Render one evidence package as prose. No numbers are invented; all are copied."""
    t = pkg["transaction"]
    m = pkg["model"]
    c = pkg["context"]
    cross = "yes" if t["from_bank"] != t["to_bank"] else "no"
    parts = [
        f"Transaction row {t['row']} sent {_usd(t['amount_usd'])} on {t['date']} from "
        f"account {t['from_account']} at bank {t['from_bank']} to account "
        f"{t['to_account']} at bank {t['to_bank']} via {t['payment_format']}.",
        f"Cross-bank transfer: {cross}.",
        f"Model score {m['raw_margin']:.4f}, day rank {m['rank_in_day']} of the "
        f"top-{m['budget']} alert budget, decision threshold {m['threshold']:.4f}; "
        f"alerted: {'yes' if m['alert'] else 'no'}.",
    ]
    for side, label in (("sender_history", "Sender"), ("receiver_history", "Receiver")):
        h = c.get(side)
        if not h:
            parts.append(f"{label} has no prior history at this timestamp.")
            continue
        out, inc = h["out"], h["in"]
        parts.append(
            f"{label} history: {out['txn_count']} prior outgoing transfers "
            f"(mean {_usd(out['amount_mean'])}, max {_usd(out['amount_max'])}, "
            f"{out['distinct_counterparties']} distinct counterparties, "
            f"{out['cross_bank_frac']:.0%} cross-bank) and {inc['txn_count']} incoming "
            f"(mean {_usd(inc['amount_mean'])}, max {_usd(inc['amount_max'])}, "
            f"{inc['distinct_counterparties']} counterparties).")
    sg = c["subgraph"]
    parts.append(
        f"Neighbourhood: {len(sg['inbound_to_sender'])} inbound edges to the sender and "
        f"{len(sg['outbound_from_receiver'])} outbound edges from the receiver within "
        f"{sg['window_ns']['hop1'] // 3_600_000_000_000:g} and "
        f"{sg['window_ns']['hop2'] // 3_600_000_000_000:g} hours."
        + (" Neighbourhood truncated at the top-N cut." if sg["truncated"] else ""))
    fan = c.get("fan_star") or {}
    parts.append(
        f"Fan degree: {fan.get('out_degree', 0)} outgoing and "
        f"{fan.get('in_degree', 0)} incoming distinct counterparties; "
        f"fan-out star {'present' if fan.get('fan_out_star') else 'absent'}, "
        f"fan-in star {'present' if fan.get('fan_in_star') else 'absent'}.")
    cycles = c.get("cycles") or []
    parts.append(
        f"Directed cycles through the sender: {len(cycles)}"
        + (f", lengths {sorted({cy['length'] for cy in cycles})}." if cycles else "."))
    chains = c.get("chains") or []
    parts.append(f"Follow-on payment chains: {len(chains)}.")
    return " ".join(parts)


def case_documents() -> list[dict]:
    docs = []
    for path in sorted(EVIDENCE_DIR.glob("package_*_row*.json")):
        pkg = json.loads(path.read_text())
        t, m = pkg["transaction"], pkg["model"]
        docs.append({
            "doc_id": f"case:row{t['row']}",
            "title": (f"Alert case row {t['row']}: {_usd(t['amount_usd'])} from account "
                      f"{t['from_account']} (bank {t['from_bank']}) to account "
                      f"{t['to_account']} (bank {t['to_bank']})"),
            "text": _render_package(pkg),
            "meta": {
                "doc_type": "case",
                "synthetic": True,
                "account": str(t["from_account"]),
                "counterparty": str(t["to_account"]),
                "from_bank": str(t["from_bank"]),
                "to_bank": str(t["to_bank"]),
                "row": str(t["row"]),
                "date": t["date"],
                "payment_format": t["payment_format"],
                "alert": bool(m["alert"]),
                "rank_in_day": int(m["rank_in_day"]),
                "package_file": path.name,
                "data_note": "synthetic IBM AML-data transactions; not real evidence",
            },
        })
    return docs


# --- build ---------------------------------------------------------------------------


def build() -> dict:
    case_docs = case_documents()
    ref_docs: list[dict] = []
    for path in sorted(REFERENCE_DIR.glob("*.pdf")):
        ref_docs.extend(reference_documents(path))
    docs = ref_docs + case_docs

    seen: set[str] = set()
    for d in docs:
        if d["doc_id"] in seen:
            raise ValueError(f"duplicate doc_id {d['doc_id']}")
        seen.add(d["doc_id"])

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with DOCS_PATH.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, sort_keys=True) + "\n")

    digest = hashlib.sha256(DOCS_PATH.read_bytes()).hexdigest()
    words = [len(d["text"].split()) for d in docs]
    label = {
        "corpus_path": str(DOCS_PATH.relative_to(PROJECT_ROOT)),
        "sha256": digest,
        "built": RETRIEVED,
        "n_documents": len(docs),
        "n_synthetic": sum(1 for d in docs if d["meta"]["synthetic"]),
        "n_reference": sum(1 for d in docs if not d["meta"]["synthetic"]),
        "word_counts": {
            "min": min(words), "max": max(words),
            "mean": round(sum(words) / len(words), 1), "total": sum(words),
        },
        "chunk_words_target": CHUNK_WORDS,
        "reference_sources": [
            {"file": k, **v} for k, v in REFERENCE_SOURCES.items()
        ],
        "label": (
            "Case documents are synthetic (IBM AML-data transactions rendered as "
            "evidence packages); they must never be presented as real-world evidence. "
            "Reference documents are verbatim public policy text, used only as "
            "background/distractor material for retrieval evaluation."),
    }
    LABEL_PATH.write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")
    return label


def main() -> int:
    label = build()
    print(f"wrote {label['corpus_path']} sha256={label['sha256'][:16]}...")
    print(f"documents: {label['n_documents']} "
          f"({label['n_synthetic']} synthetic, {label['n_reference']} reference)")
    print(f"words: {label['word_counts']}")
    unverified = sum(1 for d in map(json.loads, DOCS_PATH.read_text().splitlines())
                     if d["meta"].get("source_unverified"))
    if unverified:
        print(f"WARNING: {unverified} reference chunks have no source metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
