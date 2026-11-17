"""Stage 6 Part B: typed as-of tools over the canonical transaction table.

Each tool answers one question an analyst or an agent asks about a point in time and
returns a plain dict. Every tool requires ``as_of`` and only ever reads rows with
``ts_ns < as_of`` -- the same truncation point ``evidence._as_of`` enforces for packages,
so a tool called mid-investigation sees exactly what the package builder saw.

Contracts shared by all five tools:

* ``as_of`` is **required**. It accepts a date string ``"YYYY-MM-DD"`` (interpreted as
  00:00 UTC on that day, so the day itself is *not* visible) or an integer epoch in
   nanoseconds. Omitting it raises ``ValueError`` rather than silently using "now".
* ``limit`` caps returned row lists. When a cap bites, ``truncated`` is ``True`` and
  the totals (``n_matching`` etc.) still report the uncapped count, so a reader can
  tell "20 of 340" from "exactly 20".
* Aggregates never expose raw ``account_id`` columns as model input; account codes are
  returned as identifiers only.

Nothing here re-derives as-of logic: the tools wrap ``evidence`` primitives
(``_subgraph``, ``_cycles``, ``_fan_stars``, ``build_evidence``) and add the row caps,
the ``as_of`` parsing and a small bank-level aggregate that has no package equivalent.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import evidence as ev

DEFAULT_LIMIT = 50


# --- Shared helpers ----------------------------------------------------------------


def _as_of_ns(as_of: str | int | None) -> int:
    """Parse ``as_of`` to epoch nanoseconds. Required; date strings are 00:00 UTC."""
    if as_of is None:
        raise ValueError("as_of is required (date 'YYYY-MM-DD' or epoch nanoseconds)")
    if isinstance(as_of, (int, np.integer)):
        return int(as_of)
    text = str(as_of).strip()
    try:
        day = _dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(
            f"as_of must be 'YYYY-MM-DD' or epoch ns, got {as_of!r}") from exc
    epoch = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc)
    return int(epoch.timestamp() * 10**9)


def _cap(rows: list, limit: int) -> tuple[list, bool]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    return rows[:limit], len(rows) > limit


def _ts_to_str(ts_ns: int) -> str:
    secs, nanos = divmod(int(ts_ns), 10**9)
    base = _dt.datetime.fromtimestamp(secs, tz=_dt.timezone.utc)
    return base.strftime("%Y-%m-%d %H:%M:%S") + f".{nanos:09d}"


def _account_rows(hist: pa.Table, account: int, direction: str) -> pa.Table:
    """As-of rows for one account, newest first. ``direction`` in out/in/both."""
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be 'out', 'in' or 'both'")
    masks = []
    if direction in ("out", "both"):
        masks.append(pc.equal(hist["from_account"], account))
    if direction in ("in", "both"):
        masks.append(pc.equal(hist["to_account"], account))
    mask = masks[0] if len(masks) == 1 else pc.or_(*masks)
    t = hist.filter(mask)
    order = pc.sort_indices(t, sort_keys=[("ts_ns", "descending")])
    return t.take(order)


def _edge_dict(r: dict, account: int) -> dict:
    """One history row, oriented from ``account``'s point of view."""
    is_out = int(r["from_account"]) == account
    return {
        "row": int(r["_row"]),
        "direction": "out" if is_out else "in",
        "counterparty": int(r["to_account"]) if is_out else int(r["from_account"]),
        "ts_ns": int(r["ts_ns"]),
        "ts": _ts_to_str(r["ts_ns"]),
        "amount_usd": float(r["amount_usd"]),
        "payment_format": str(r["payment_format"]),
        "counterparty_bank": int(r["to_bank"]) if is_out else int(r["from_bank"]),
    }


# --- The five tools ----------------------------------------------------------------


def get_account_history(account: int, as_of: str | int | None, *,
                        direction: str = "both", limit: int = DEFAULT_LIMIT,
                        table: pa.Table | None = None) -> dict:
    """Most recent transfers to/from ``account`` strictly before ``as_of``."""
    as_of_ns = _as_of_ns(as_of)
    full = ev._load_table(None)
    hist = ev._as_of(ev._load_table(table), as_of_ns, full)
    rows = _account_rows(hist, int(account), direction)
    payload = [_edge_dict(r, int(account)) for r in rows.to_pylist()]
    capped, truncated = _cap(payload, limit)
    return {
        "tool": "get_account_history",
        "account": int(account),
        "as_of": as_of,
        "direction": direction,
        "n_matching": len(payload),
        "rows": capped,
        "truncated": truncated,
        "known_gaps": ev._provenance("", None)["known_gaps"],
    }


def get_neighbours(account: int, as_of: str | int | None, *,
                   limit: int = ev.TOP_NEIGHBOURS,
                   table: pa.Table | None = None) -> dict:
    """Two-hop neighbourhood of ``account`` strictly before ``as_of``.

    Delegates to the package subgraph (one day back for outbound, two days for inbound)
    and re-caps both lists at ``limit``. ``truncated`` is true when either the package
    subgraph cut or the caller's ``limit`` cut.
    """
    as_of_ns = _as_of_ns(as_of)
    full = ev._load_table(None)
    hist = ev._as_of(ev._load_table(table), as_of_ns, full)
    account = int(account)

    # ``_subgraph`` is anchored on a (sender, receiver) *pair*: inbound edges are those
    # arriving at its sender, outbound those leaving its receiver. A single-account
    # query has no pair, so the two halves are enumerated directly over the as-of table.
    hop1_ns = ev.HOP1_WINDOW_NS
    hop2_ns = ev.HOP2_WINDOW_NS
    in_t = hist.filter(pc.and_(
        pc.equal(hist["to_account"], account),
        pc.greater(hist["ts_ns"], as_of_ns - hop2_ns)))
    out_t = hist.filter(pc.and_(
        pc.equal(hist["from_account"], account),
        pc.greater(hist["ts_ns"], as_of_ns - hop1_ns)))

    def _edge(r: dict, inbound: bool) -> dict:
        cp = int(r["from_account"]) if inbound else int(r["to_account"])
        return {
            "row": int(r["_row"]),
            "counterparty": cp,
            "ts_ns": int(r["ts_ns"]),
            "ts": _ts_to_str(r["ts_ns"]),
            "amount_usd": float(r["amount_usd"]),
            "counterparty_bank": int(r["from_bank"]) if inbound else int(r["to_bank"]),
        }

    in_order = in_t.take(pc.sort_indices(in_t, sort_keys=[("ts_ns", "descending")]))
    out_order = out_t.take(pc.sort_indices(out_t, sort_keys=[("ts_ns", "descending")]))
    inbound = [_edge(r, True) for r in in_order.to_pylist()]
    outbound = [_edge(r, False) for r in out_order.to_pylist()]
    cap_in, cut_in = _cap(inbound, limit)
    cap_out, cut_out = _cap(outbound, limit)
    fan = ev._fan_stars(hist, account)
    return {
        "tool": "get_neighbours",
        "account": account,
        "as_of": as_of,
        "window_ns": {"hop1": hop1_ns, "hop2": hop2_ns},
        "n_inbound": len(inbound),
        "n_outbound": len(outbound),
        "inbound": cap_in,
        "outbound": cap_out,
        # Windowed edge lists above count only the last hop1/hop2 of history; these
        # degrees count every edge strictly before ``as_of``. Different questions,
        # both legitimate, hence distinct names.
        "degree_all_time": {
            "out_degree_all_time": fan["out_degree"],
            "in_degree_all_time": fan["in_degree"],
            "fan_out_star_all_time": fan["fan_out_star"],
            "fan_in_star_all_time": fan["fan_in_star"],
        },
        "truncated": bool(cut_in or cut_out),
    }


def find_cycles(account: int, as_of: str | int | None, *,
                limit: int = 5, table: pa.Table | None = None) -> dict:
    """Directed cycles ``account -> ... -> account`` (3-6 hops, within one day)."""
    as_of_ns = _as_of_ns(as_of)
    full = ev._load_table(None)
    hist = ev._as_of(ev._load_table(table), as_of_ns, full)
    cycles = ev._cycles(hist, int(account), as_of_ns)
    capped, truncated = _cap(cycles, limit)
    for c in capped:
        for hop in c["hops"]:
            hop["ts"] = _ts_to_str(hop.pop("ts_ns"))
    return {
        "tool": "find_cycles",
        "account": int(account),
        "as_of": as_of,
        "n_matching": len(cycles),
        "cycles": capped,
        "truncated": bool(truncated),
        "note": "one example cycle per length range is retained by the builder",
    }


def get_alert_evidence(row: int, as_of: str | int | None, *,
                       budget: int = ev.DEFAULT_BUDGET,
                       table: pa.Table | None = None) -> dict:
    """Full evidence package for canonical ``row``, gated on ``as_of``.

    The package is built as-of the transaction's own timestamp (that is what a queue
    alert always is). ``as_of`` is still required and checked: asking for evidence at a
    time before the transaction happened would be lookahead, so it is rejected.
    """
    as_of_ns = _as_of_ns(as_of)
    full = ev._load_table(None)
    ts_ns = int(full["ts_ns"][int(row)].as_py())
    if as_of_ns <= ts_ns:
        raise ValueError(
            f"as_of precedes the transaction (ts {_ts_to_str(ts_ns)}); "
            "an alert package cannot be viewed before the alert fires")
    pkg = ev.build_evidence(row=int(row), table=table, budget=budget)
    pkg["tool"] = "get_alert_evidence"
    pkg["as_of"] = as_of
    return pkg


def get_bank_stats(bank: int, as_of: str | int | None, *,
                   limit: int = 20, table: pa.Table | None = None) -> dict:
    """Aggregates for one bank strictly before ``as_of``, plus its counterparties.

    No package primitive covers bank level, so the counts are computed here over the
    same as-of table the other tools use. ``top_counterparties`` is per counterparty
    bank (in + out) with cross-bank share, capped by ``limit``.
    """
    as_of_ns = _as_of_ns(as_of)
    full = ev._load_table(None)
    hist = ev._as_of(ev._load_table(table), as_of_ns, full)
    summary, truncated = _summarise_bank(hist, int(bank), limit)
    return {
        "tool": "get_bank_stats",
        "bank": int(bank),
        "as_of": as_of,
        **summary,
        "truncated": bool(truncated),
    }


def _summarise_bank(hist: pa.Table, bank: int, limit: int) -> tuple[dict, bool]:
    is_from = pc.equal(hist["from_bank"], bank)
    is_to = pc.equal(hist["to_bank"], bank)
    # ``out`` is outbound-only and ``in`` inbound-only; intra-bank rows (both sides) are
    # counted once as ``out`` here, so n_out + n_in stays disjoint from n_intra_bank.
    out_t = hist.filter(pc.and_(is_from, pc.invert(is_to)))
    in_t = hist.filter(pc.and_(is_to, pc.invert(is_from)))
    intra = int(hist.filter(pc.and_(is_from, is_to)).num_rows)
    both_t = hist.filter(pc.or_(is_from, is_to))
    n = both_t.num_rows
    if n:
        a = both_t["amount_usd"].to_numpy()
        amount_sum, amount_mean, amount_max = float(a.sum()), float(a.mean()), float(a.max())
    else:
        amount_sum = amount_mean = amount_max = 0.0

    accounts = np.unique(np.concatenate([
        out_t["from_account"].to_numpy(), in_t["to_account"].to_numpy()])) \
        if n else np.array([], dtype=np.int64)

    # Counterparty = the other side of each row touching this bank.
    out_rows = out_t.filter(pc.not_equal(out_t["to_bank"], bank))
    in_rows = in_t.filter(pc.not_equal(in_t["from_bank"], bank))
    other = np.concatenate([out_rows["to_bank"].to_numpy(),
                            in_rows["from_bank"].to_numpy()])
    amt = np.concatenate([out_rows["amount_usd"].to_numpy(),
                          in_rows["amount_usd"].to_numpy()])
    merged: dict[int, list] = {}
    for b, x in zip(other.tolist(), amt.tolist()):
        s = merged.setdefault(b, [0, 0.0])
        s[0] += 1
        s[1] += x
    ranked = sorted(
        ({"bank": b, "txn_count": c, "amount_usd": round(v, 2)} for b, (c, v) in merged.items()),
        key=lambda d: (-d["txn_count"], d["bank"]))
    capped, truncated = _cap(ranked, limit)
    summary = {
        "n_txn": int(n),
        "n_out": int(out_t.num_rows),
        "n_in": int(in_t.num_rows),
        "n_intra_bank": intra,
        "amount_usd_sum": round(amount_sum, 2),
        "amount_usd_mean": round(amount_mean, 6),
        "amount_usd_max": round(amount_max, 6),
        "n_distinct_accounts": int(accounts.size),
        "n_counterparty_banks": len(merged),
        "top_counterparties": capped,
    }
    return summary, truncated
