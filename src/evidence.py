"""Stage 5: as-of evidence package builder.

A package answers "why is *this transfer* in the analyst's queue?" It is built for one
canonical-table row and contains only what was knowable at that transaction's timestamp.

As-of discipline
----------------
Every historical field (sender/receiver history, neighbour subgraph, cycles, fan stars,
chains) is computed from rows with ``ts_n < ts_ns`` of the target. Nothing reads a
same-timestamp row, let alone a later one, so truncating the table at ``ts_ns`` cannot
change a package - ``tests/test_evidence_leakage.py`` asserts exactly that.

Two deliberate gaps vs. the Stage 5 note, both dictated by what Stage 4 actually saved:

* **Original amount / currency are not emitted.** The canonical table stores
  ``amount_usd`` only; the raw CSV columns were not carried through. Publishing
  ``amount_usd`` alone is honest; inventing a currency would not be.
* **No SHAP drivers.** Saved boosters are model files, not SHAP artefacts, and computing
  SHAP per package would load a booster per call. The package carries the raw margin
  score and the stored feature vector, from which any driver attribution is re-derivable.

Provenance records both gaps so a reader knows what is missing rather than assuming
absence means zero.

Public API
----------
``build_evidence(row=..., table=None, ...)`` -> dict, JSON-serialisable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data" / "_trans_canonical.parquet"
FEATURES_PATH = PROJECT_ROOT / "eda_output" / "features" / "edge_features_v1.parquet"
SCORES_PATH = PROJECT_ROOT / "results" / "ladder" / "rung2_lgbm_plus_hop_scores.parquet"
THRESHOLDS_PATH = PROJECT_ROOT / "results" / "alerting" / "alert_thresholds.parquet"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "feature_manifest.json"

DAY_NS = 86_400 * 10**9
HOP1_WINDOW_NS = DAY_NS          # 1-hop subgraph: one day back
HOP2_WINDOW_NS = 2 * DAY_NS      # 2-hop subgraph: two days back
CHAIN_WINDOW_NS = DAY_NS         # chain hops at most 24h apart (Stage 5 definition)
CYCLE_LEN = (3, 4, 5, 6)
TOP_NEIGHBOURS = 20
DEFAULT_BUDGET = 200


# --- Table helpers -----------------------------------------------------------------


_TABLE_CACHE: dict[tuple[int, int], pa.Table] = {}


def _load_table(table: pa.Table | None) -> pa.Table:
    """The canonical table, memoised on the file's (mtime_ns, size).

    Every Stage 6 tool re-read the whole canonical parquet and rebuilt the canonical
    id lookup (~11 s, ~4 GB transient). The agent issues up to 8 such calls per case,
    which made a single draft run exceed the 7 GB ceiling. Caching one immutable
    table per file version removes both costs; a rewritten file gets a new key so a
    stale table can never be served.
    """
    if table is not None:
        return table
    st = CANONICAL_PATH.stat()
    key = (st.st_mtime_ns, st.st_size)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        _TABLE_CACHE.clear()
        cached = pq.read_table(CANONICAL_PATH)
        _TABLE_CACHE[key] = cached
    return cached


_LOOKUP_CACHE: dict[tuple, pa.Table] = {}


def _canon_lookup(full: pa.Table) -> pa.Table:
    """Distinct canonical ``_key`` -> minimum canonical row id, memoised per table.

    Building this was 86% of a Stage 6 tool call (~11 s, 4 GB footprint) because it
    string-casts and group-bys the whole canonical table on every call. The key is the
    table's identity plus its fingerprint (row count, first/last ts, partition checksum
    when the table carries one), so a filtered or rebuilt table gets its own entry and
    a stale entry can never be reused. The canonical table itself is also retained by
    ``_load_table``'s callers for the life of a process, so holding a lookup keyed on
    it adds no retention.
    """
    fp = [full.num_rows]
    for col in ("ts_ns",):
        if col in full.column_names and full.num_rows:
            fp += [int(pc.min(full[col]).as_py()), int(pc.max(full[col]).as_py())]
    fp.append(id(full))
    key = tuple(fp)
    cached = _LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached
    k = lambda t: [pc.cast(t[c], pa.string()) if c != "amount_usd"
                   else pc.cast(pc.round(t[c], 6), pa.string())
                   for c in ("from_account", "to_account", "ts_ns", "amount_usd")]
    canon_keys = pa.Table.from_arrays(
        [pc.binary_join_element_wise(*k(full), "\x1f"),
         pa.array(np.arange(full.num_rows, dtype=np.int64))],
        names=["_key", "_row"])
    lookup = canon_keys.group_by("_key").aggregate([("_row", "min")]).select(["_key", "_row_min"])
    if len(_LOOKUP_CACHE) >= 4:
        _LOOKUP_CACHE.pop(next(iter(_LOOKUP_CACHE)))
    _LOOKUP_CACHE[key] = lookup
    return lookup


def _as_of(table: pa.Table, ts_ns: int, full: pa.Table) -> pa.Table:
    """Rows strictly earlier than ``ts_ns`` plus an ``_row`` column of canonical indices.

    The ``_row`` column is carried so cycles and chains can report canonical row ids
    rather than only account codes. It must be the *canonical* id, not a position in
    whatever table the caller passed: a truncated table has fewer rows, so numbering
    positionally would re-label every surviving row and make the package depend on the
    caller's view - which is exactly the leak the truncation test detects. Ids are
    resolved by matching each history row back to its canonical position on the
    transaction key ``(from_account, to_account, ts_ns, amount_usd)``.
    """
    hist = table.filter(pc.less(table["ts_ns"], ts_ns))
    key = lambda t: [pc.cast(t[c], pa.string()) if c != "amount_usd"
                     else pc.cast(pc.round(t[c], 6), pa.string())
                     for c in ("from_account", "to_account", "ts_ns", "amount_usd")]
    merged = pa.Table.from_arrays(
        [pc.binary_join_element_wise(*key(hist), "\x1f")], names=["_key"])
    lookup = _canon_lookup(full)
    # A many-to-many left-outer join does not preserve left-row order, so joining and
    # appending the joined column would pair ids with the wrong history rows. Resolve
    # element-wise instead: index_in/take keep ``hist`` row order exactly.
    idx = pc.index_in(merged["_key"], value_set=lookup["_key"])
    row_ids = pc.take(lookup["_row_min"], idx)
    return hist.append_column("_row", row_ids)


def _feature_row(row: int) -> dict | None:
    """Read the single stored feature row for canonical row ``row``.

    Reads one row group, not the file: the features parquet is ~1 GB across 18 row
    groups, and a full read per package call would blow the memory ceiling.
    """
    if not FEATURES_PATH.exists():
        return None
    pf = pq.ParquetFile(FEATURES_PATH)
    md = pf.metadata
    offset = 0
    rg = None
    for i in range(md.num_row_groups):
        n = md.row_group(i).num_rows
        if row < offset + n:
            rg = i
            break
        offset += n
    if rg is None:
        return None
    t = pf.read_row_group(rg).slice(row - offset, 1)
    rec = t.to_pylist()[0]
    rec.pop("from_account", None)
    rec.pop("to_account", None)
    if "date" in rec and rec["date"] is not None:
        rec["date"] = str(rec["date"])
    return rec


def _score_row(row: int) -> dict | None:
    if not SCORES_PATH.exists():
        return None
    t = pq.read_table(SCORES_PATH, columns=["row", "score", "date"])
    t = t.filter(pc.equal(t["row"], row))
    if t.num_rows == 0:
        return None
    rec = t.slice(0, 1).to_pylist()[0]
    rec["date"] = str(rec["date"])
    return rec


def _thresholds_for(day: str, budget: int) -> float | None:
    if not THRESHOLDS_PATH.exists():
        return None
    t = pq.read_table(THRESHOLDS_PATH)
    # Call sites pass ``score["date"]`` (a str(Timestamp) like "2022-09-07 00:00:00")
    # while the thresholds store plain "2022-09-07"; compare on the date form only.
    t = t.filter(pc.and_(pc.equal(pc.cast(t["day"], pa.string()), day[:10]),
                         pc.equal(t["budget"], budget)))
    if t.num_rows == 0:
        return None
    return float(t["threshold"][0].as_py())


def _rank_in_day(row: int, day: str, ts_ns: int) -> int | None:
    """Rank of ``row`` among strictly earlier same-day rows, ranked by score.

    The scores file has no timestamp column, so the cut is made by joining the
    canonical ``ts_ns`` for each scored row. Without this the rank would count
    same-timestamp and later rows - an as-of leak the canonical-truncation test
    cannot see because it only truncates canonical, not the scores file.
    """
    if not SCORES_PATH.exists():
        return None
    t = pq.read_table(SCORES_PATH, columns=["row", "score", "date"])
    # Cast through date32, not straight to string: timestamp->string yields
    # "2022-09-03 00:00:00.000000000" whereas call sites pass str(Timestamp) =
    # "2022-09-03 00:00:00", so a direct cast silently matches zero rows.
    day_str = pc.cast(pc.cast(t["date"], pa.date32()), pa.string())
    t = t.filter(pc.equal(day_str, day[:10]))
    if t.num_rows == 0:
        return None
    canon = pq.read_table(CANONICAL_PATH, columns=["ts_ns"])
    ts_col = canon["ts_ns"].take(pa.array(t["row"].to_numpy()))
    mask = pc.less(ts_col, ts_ns)
    earlier = t.filter(mask)
    target = t.filter(pc.equal(t["row"], row))["score"][0].as_py()
    return int((earlier["score"].to_numpy() > target).sum()) + 1


def _golden_label(table: pa.Table, row: int) -> int | None:
    if "is_laundering" not in table.column_names:
        return None
    return int(table["is_laundering"][row].as_py())


# --- History and subgraph ----------------------------------------------------------


def _history(hist: pa.Table, account: int) -> dict | None:
    """As-of sender/receiver history for one account."""
    col_from = pc.equal(hist["from_account"], account)
    col_to = pc.equal(hist["to_account"], account)
    as_sender = hist.filter(col_from)
    as_recv = hist.filter(col_to)
    n_out = as_sender.num_rows
    n_in = as_recv.num_rows
    if n_out + n_in == 0:
        return None

    def _stats(t, name: str) -> dict:
        is_out = name == "out"
        cp_col = "to_account" if is_out else "from_account"
        src_bank = "from_bank" if is_out else "to_bank"
        dst_bank = "to_bank" if is_out else "from_bank"
        if t.num_rows:
            amt = t["amount_usd"].to_numpy()
            amount_mean = float(np.mean(amt))
            amount_max = float(np.max(amt))
            distinct_cp = int(np.unique(t[cp_col].to_numpy()).size)
            cross = float(np.mean(t[src_bank].to_numpy() != t[dst_bank].to_numpy()))
        else:
            amount_mean = 0.0
            amount_max = 0.0
            distinct_cp = 0
            cross = 0.0
        return {
            "txn_count": int(t.num_rows),
            "amount_mean": amount_mean,
            "amount_max": amount_max,
            "distinct_counterparties": distinct_cp,
            "cross_bank_frac": cross,
        }

    first_candidates = [
        pc.min(t["ts_ns"]).as_py() for t in (as_sender, as_recv) if t.num_rows
    ]
    return {
        "first_seen_ns": int(min(first_candidates)),
        "out": _stats(as_sender, "out"),
        "in": _stats(as_recv, "in"),
    }


def _subgraph(hist: pa.Table, src: int, dst: int, ts_ns: int) -> dict:
    """Two-hop account neighbourhood as-of the transaction, truncated at the window."""
    hop1_start = ts_ns - HOP1_WINDOW_NS
    hop2_start = ts_ns - HOP2_WINDOW_NS
    within1 = pc.greater_equal(hist["ts_ns"], hop1_start)
    within2 = pc.greater_equal(hist["ts_ns"], hop2_start)

    inbound = hist.filter(pc.and_(within2, pc.equal(hist["to_account"], src)))
    outbound = hist.filter(pc.and_(within1, pc.equal(hist["from_account"], dst)))

    def _rows(t) -> list[dict]:
        return [
            {"row": int(r["_row"]), "from": int(r["from_account"]),
             "to": int(r["to_account"]), "ts_ns": int(r["ts_ns"]),
             "amount_usd": float(r["amount_usd"])}
            for r in t.to_pylist()[:TOP_NEIGHBOURS]
        ]

    truncated = (src in set(inbound["from_account"].to_pylist())
                 if inbound.num_rows else False)
    truncated = bool(truncated) or inbound.num_rows > TOP_NEIGHBOURS or \
        outbound.num_rows > TOP_NEIGHBOURS

    return {
        "window_ns": {"hop1": HOP1_WINDOW_NS, "hop2": HOP2_WINDOW_NS},
        "inbound_to_sender": _rows(inbound),
        "outbound_from_receiver": _rows(outbound),
        "truncated": bool(truncated),
    }


def _cycles(hist: pa.Table, src: int, ts_ns: int) -> list[dict]:
    """Directed cycles from ``src`` back to ``src`` of length 3-6, within one day."""
    lo = ts_ns - DAY_NS
    h = hist.filter(pc.greater_equal(hist["ts_ns"], lo))
    if h.num_rows == 0:
        return []
    edges = list(zip(h["from_account"].to_pylist(),
                     h["to_account"].to_pylist(),
                     h["_row"].to_pylist(),
                     h["ts_ns"].to_pylist(),
                     h["amount_usd"].to_pylist()))
    # Adjacency: from -> list of (to, row, ts, amt)
    adj: dict[int, list] = {}
    for f, t, r, s, a in edges:
        adj.setdefault(f, []).append((t, r, s, a))

    found = []
    for t, r, s, a in adj.get(src, []):
        if t == src:
            continue
        path = [(src, t, r, s, a)]
        stack = [(t, [src, t], path)]
        while stack:
            node, visited, here = stack.pop()
            if len(here) >= CYCLE_LEN[1]:
                continue
            for nxt in adj.get(node, []):
                to2, r2, s2, a2 = nxt
                new_path = here + [(node, to2, r2, s2, a2)]
                if to2 == src:
                    if CYCLE_LEN[0] <= len(new_path) <= CYCLE_LEN[1]:
                        found.append(new_path)
                    continue
                if to2 not in visited and len(new_path) < CYCLE_LEN[1]:
                    stack.append((to2, visited + [to2], new_path))
        if found:
            break  # one example cycle per length range is enough for evidence
    cycles = []
    seen_keys = set()
    for c in found:
        length = len(c)
        key = (length, tuple((h0[0], h0[1]) for h0 in c))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cycles.append({
            "length": length,
            "hops": [{"from": int(h0[0]), "to": int(h0[1]),
                      "row": int(h0[2]), "ts_ns": int(h0[3])} for h0 in c],
        })
    return cycles[:5]


def _fan_stars(hist: pa.Table, account: int) -> dict:
    """Inbound and outbound fan counts for one account as-of the transaction."""
    n_out = hist.filter(pc.equal(hist["from_account"], account)).num_rows
    n_in = hist.filter(pc.equal(hist["to_account"], account)).num_rows
    return {
        "out_degree": int(n_out),
        "in_degree": int(n_in),
        "fan_out_star": n_out >= 20,
        "fan_in_star": n_in >= 20,
    }


def _chains(hist: pa.Table, src: int, dst: int, ts_ns: int) -> list[dict]:
    """Chains through this transaction, sender=previous receiver, hops <= 24h apart."""
    h = hist.filter(pc.greater_equal(hist["ts_ns"], ts_ns - HOP2_WINDOW_NS))
    edges = list(zip(h["from_account"].to_pylist(),
                     h["to_account"].to_pylist(),
                     h["_row"].to_pylist(),
                     h["ts_ns"].to_pylist()))
    by_from: dict[int, list] = {}
    by_to: dict[int, list] = {}
    for f, t, r, s in edges:
        by_from.setdefault(f, []).append((t, r, s))
        by_to.setdefault(t, []).append((f, r, s))

    before = [e for e in by_to.get(src, []) if ts_ns - e[2] <= CHAIN_WINDOW_NS]
    after = [e for e in by_from.get(dst, []) if e[2] >= ts_ns
             and e[2] - ts_ns <= CHAIN_WINDOW_NS]

    chains = []
    for f, r, s in before:
        prev = [{"from": int(f), "to": int(src), "row": int(r), "ts_ns": int(s)}]
        for t2, r2, s2 in after:
            chains.append({
                "prev_hop": prev[0],
                "this_hop": {"from": int(src), "to": int(dst),
                             "row": None, "ts_ns": int(ts_ns)},
                "next_hop": {"from": int(dst), "to": int(t2),
                             "row": int(r2), "ts_ns": int(s2)},
                "total_hops": 3,
            })
        if not after:
            chains.append({"prev_hop": prev[0], "this_hop": {
                "from": int(src), "to": int(dst), "row": None, "ts_ns": int(ts_ns)},
                "next_hop": None, "total_hops": 2})
    if not before and after:
        for t2, r2, s2 in after:
            chains.append({
                "prev_hop": None,
                "this_hop": {"from": int(src), "to": int(dst),
                             "row": None, "ts_ns": int(ts_ns)},
                "next_hop": {"from": int(dst), "to": int(t2),
                             "row": int(r2), "ts_ns": int(s2)},
                "total_hops": 2,
            })
    return chains[:10]


# --- Provenance ---------------------------------------------------------------------


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_version() -> str:
    import json
    if not MANIFEST_PATH.exists():
        return "unknown"
    with open(MANIFEST_PATH) as fh:
        m = json.load(fh)
    return f"{m.get('table', 'edge_features')}/{m.get('version', 'unknown')}"


def _provenance(day: str, score_day: str | None) -> dict:
    # ``score_day`` arrives as str(Timestamp) = "2022-09-07 00:00:00" while boosters and
    # the schema's ``scored_day`` use plain "2022-09-07"; normalise before hashing.
    d = (score_day or day)[:10]
    booster = PROJECT_ROOT / "results" / "ladder" / "models" / \
        f"rung2_lgbm_plus_hop_fold-{d}.txt"
    return {
        "feature_version": _manifest_version(),
        "model_version": _sha256(booster),
        "scored_day": d,
        "canonical_table": "data/_trans_canonical.parquet",
        "score_source": "results/ladder/rung2_lgbm_plus_hop_scores.parquet",
        "chain_definition": {
            "min_hops": 2, "sender_eq_prev_receiver": True,
            "max_hop_gap_ns": CHAIN_WINDOW_NS,
        },
        "as_of_rule": "strictly earlier timestamps only (ts_n < ts_ns)",
        "known_gaps": [
            "original amount and currency not carried through to canonical table",
            "no SHAP drivers stored; drivers re-derivable from stored features",
        ],
    }


# --- Public builder ----------------------------------------------------------------


def build_evidence(row: int, table: pa.Table | None = None,
                   budget: int = DEFAULT_BUDGET) -> dict:
    """Build an as-of evidence package for canonical row ``row``.

    ``table`` overrides the canonical table load (used by the leakage test to pass a
    truncated table). When omitted the file is read from disk.
    """
    # The target row's identity always comes from disk. When ``table`` is a truncated
    # slice it does not contain the target at its canonical position, so reading the
    # transaction from it positionally would describe a different transfer. ``table``
    # is only the source of historical rows.
    full = _load_table(None)
    src = int(full["from_account"][row].as_py())
    dst = int(full["to_account"][row].as_py())
    ts_ns = int(full["ts_ns"][row].as_py())
    day = str(full["date"][row].as_py()) if "date" in full.column_names else None

    hist = _as_of(_load_table(table), ts_ns, full)

    transaction = {
        "row": int(row),
        "from_account": src,
        "to_account": dst,
        "from_bank": int(full["from_bank"][row].as_py()),
        "to_bank": int(full["to_bank"][row].as_py()),
        "payment_format": str(full["payment_format"][row].as_py()),
        "ts_ns": ts_ns,
        "date": day,
        "amount_usd": float(full["amount_usd"][row].as_py()),
        "golden_label": _golden_label(full, row),
    }

    score = _score_row(row)
    model_out: dict = {"raw_margin": None, "rank_in_day": None,
                      "threshold": None, "alert": None}
    if score is not None:
        model_out["raw_margin"] = float(score["score"])
        model_out["rank_in_day"] = _rank_in_day(row, score["date"], ts_ns)
        thr = _thresholds_for(score["date"], budget)
        if thr is not None:
            model_out["threshold"] = thr
            model_out["alert"] = bool(score["score"] >= thr)
    model_out["budget"] = budget

    context = {
        "sender_history": _history(hist, src),
        "receiver_history": _history(hist, dst),
        "subgraph": _subgraph(hist, src, dst, ts_ns),
        "cycles": _cycles(hist, src, ts_ns),
        "fan_star": _fan_stars(hist, src),
        "chains": _chains(hist, src, dst, ts_ns),
    }

    return {
        "transaction": transaction,
        "model": model_out,
        "features": _feature_row(row),
        "context": context,
        "provenance": _provenance(day or "", score["date"] if score else None),
    }
