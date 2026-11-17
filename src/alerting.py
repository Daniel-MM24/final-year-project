"""Stage 5: budget-parameterised alert policy on the leaned-on Rung 2 scores.

Run from the project root with the learn_env interpreter::

    python src/alerting.py                 # default budgets 50 / 200 / 1000
    python src/alerting.py --budgets 50 200 1000

Why a budget, not a probability cutoff
--------------------------------------
An investigation team can review N alerts per day. The useful question is "what do we
catch with the ones we can actually look at", not "which rows exceed 0.5" - and on this
data every reasonable probability threshold sits far below 0.5 anyway (see
``results/calibration/calibration.md``). The policy therefore takes the **top-N rows by
score within each validation day** and nothing else: the Nth score is the threshold, and
it is re-derived every day.

Inputs
------
* ``results/ladder/rung2_lgbm_plus_hop_scores.parquet`` - raw-margin scores per fold.
* ``results/calibration/calibration.json`` - Platt coefficients fitted on an earlier
  validation day for the same fold. Rung 2 is re-fitted per day here from Rung 2's own
  earlier-day scores rather than reusing Rung 1's coefficients: the two rungs differ in
  scale, and borrowing Rung 1's calibrator would reintroduce the level mismatch the
  calibration stage exists to remove.

Calibration discipline
----------------------
For validation day D the calibrator is fitted on the most recent *earlier* validation
day, exactly as ``src/models/calibration.py`` does. The first validation day (09-03) has
no earlier scored day, so it is reported uncalibrated and **excluded from every budget
number** - a threshold that cannot be calibrated is not a threshold anyone could deploy.

Metrics per budget
------------------
* ``precision``    - true positives / alerts. The share of the analyst's queue that was
  actually laundering.
* ``recall``       - true positives / positives that day.
* ``true_positives``- the raw count the analyst would find.
* ``chained_edges_lost`` - of the laundering edges that *belong to a chain* (see below),
  how many sit in a chain where this budget alerts **none** of the chain's laundering
  edges. These are the paths the analyst would miss entirely at this budget: the cost of
  a tight queue in whole paths, not in partial views.

Chain definition
----------------
A **chain** is a path of two or more transfers ``a -> b -> c -> ...`` where each hop's
sender is the previous hop's receiver and consecutive hops are at most ``CHAIN_WINDOW``
apart. This is the definition agreed for Stage 5; it is written into every package's
provenance so a downstream reader can re-derive it.

Artefacts
---------
* ``results/alerting/alerting.json`` - every budget, every fold, plus the pooled numbers.
* ``results/alerting/alerting.md``   - the write-up.
* ``results/alerting/alert_thresholds.parquet`` - per-day score threshold for each budget,
  which Stage 5's evidence builder consumes.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SCORES_PATH = PROJECT_ROOT / "results" / "ladder" / "rung2_lgbm_plus_hop_scores.parquet"
CANONICAL_PATH = PROJECT_ROOT / "data" / "_trans_canonical.parquet"
OUT_DIR = PROJECT_ROOT / "results" / "alerting"

DEFAULT_BUDGETS = (50, 200, 1000)
# Matches src/models/calibration.py: the raw margin is a log-odds and is clipped well
# outside the informative region before Platt is fitted.
MARGIN_CLIP = 20.0
# Consecutive hops in a chain must be within this gap. 24h matches the 1-day embargo and
# is the agreed Stage 5 definition, not a tuned constant.
CHAIN_WINDOW = pd.Timedelta(days=1)


# --- Calibration (same protocol as src/models/calibration.py) ----------------------


def _platt(scores: np.ndarray, labels: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    m.fit(np.clip(scores, -MARGIN_CLIP, MARGIN_CLIP).reshape(-1, 1), labels)
    return m


def calibrated_scores_by_day(scores: pd.DataFrame) -> pd.DataFrame:
    """Attach ``calibrated`` to each row, fitted on the previous validation day.

    Day 09-03 has no earlier day and gets ``calibrated = NaN``; it is excluded from every
    budget metric rather than scored with an unfittable threshold.
    """
    days = sorted(scores["date"].unique())
    parts = []
    for i, day in enumerate(days):
        cur = scores[scores["date"] == day].copy()
        if i == 0:
            cur["calibrated"] = np.nan
            cur["calibrated"] = cur["calibrated"].astype(float)
            parts.append(cur)
            continue
        prev_day = days[i - 1]
        prev = scores[scores["date"] == prev_day]
        model = _platt(prev["score"].to_numpy(dtype=float),
                       prev["is_laundering"].to_numpy(dtype=int))
        cur["calibrated"] = model.predict_proba(
            np.clip(cur["score"].to_numpy(dtype=float), -MARGIN_CLIP, MARGIN_CLIP)
            .reshape(-1, 1))[:, 1]
        parts.append(cur)
    return pd.concat(parts, ignore_index=True)


# --- Chains ------------------------------------------------------------------------


def build_chains(window: pd.Timedelta = CHAIN_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Walk account-contiguous transfer paths within ``window``.

    Returns ``(chain_root, chain_len)``, two arrays indexed by canonical-table row. The
    chain root is the canonical row of the first hop; ``chain_len == 1`` means the edge is
    not in a real chain.

    The walk is the same one the reference implementation describes - at each hop take the
    earliest edge leaving the hop's receiver, strictly after the hop and no more than
    ``window`` later - but it is done over integer arrays rather than dicts of strings.
    The previous version built a per-account adjacency of ``(Timestamp, id-string)`` tuples
    plus id-keyed dicts and was OOM-killed on this box at ~3 GB available; the canonical
    accounts are already integer-coded, so the adjacency is a sort plus ``searchsorted``.

    Deterministic because the successor search is over a stable sort of
    ``(from_account, ts_ns)`` and edges are walked in canonical-row order.
    """
    import pyarrow.parquet as pq

    tx = pq.read_table(CANONICAL_PATH, columns=["ts_ns", "from_account", "to_account"])
    ts = tx["ts_ns"].to_numpy().astype(np.int64, copy=False)
    src = tx["from_account"].to_numpy().astype(np.int64, copy=False)
    dst = tx["to_account"].to_numpy().astype(np.int64, copy=False)
    del tx
    gc.collect()

    n = len(ts)
    window_ns = np.int64(window.value)
    # Edges grouped by sender, earliest first. ``order`` maps sorted position -> row.
    order = np.lexsort((ts, src))
    src_sorted = src[order]
    ts_sorted = ts[order]
    # First position of each sender in the sorted order, plus a sentinel so that a sender
    # absent from the table gets an empty range.
    n_acct = int(src_sorted.max()) + 2
    starts = np.zeros(n_acct, dtype=np.int64)
    np.cumsum(np.bincount(src_sorted, minlength=n_acct - 1), out=starts[1:])

    # Earliest successor edge of each edge, or -1.
    dst_sorted = dst[order]
    next_pos = np.full(n, -1, dtype=np.int64)
    for rank, row in enumerate(order):
        to = dst_sorted[rank]
        t0 = ts_sorted[rank]
        lo = starts[to]
        hi = starts[to + 1]
        if lo >= hi:
            continue
        # first edge of ``to`` strictly after t0, then check the window
        pos = lo + bisect_right(ts_sorted[lo:hi], t0)
        if pos < hi and ts_sorted[pos] - t0 <= window_ns:
            next_pos[rank] = pos
    next_edge = order[next_pos]   # row of the successor edge, -1 where none

    root = np.full(n, -1, dtype=np.int64)
    chain_len = np.ones(n, dtype=np.int16)
    pos_of_row = np.empty(n, dtype=np.int64)
    pos_of_row[order] = np.arange(n, dtype=np.int64)

    for row in range(n):
        if root[row] != -1:
            continue
        path = [row]
        cur = row
        while True:
            nxt_row = int(next_edge[pos_of_row[cur]])
            if nxt_row < 0 or root[nxt_row] != -1 or nxt_row == row:
                break
            if nxt_row in path:
                break
            path.append(nxt_row)
            cur = nxt_row
        length = len(path)
        for r in path:
            root[r] = row
            chain_len[r] = length
    return root, chain_len


# --- Alert evaluation --------------------------------------------------------------


def evaluate_budgets(scored: pd.DataFrame,
                     budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> dict:
    """``scored`` must carry ``calibrated``, ``is_laundering``, ``chain_root``, ``chain_len``.

    ``chain_root`` is a dense integer code for the chain an edge belongs to (its canonical
    row of the first hop). Chains are compared by code, not by id string, so the whole
    evaluation stays on numpy arrays.
    """
    usable = scored[scored["calibrated"].notna()]
    days = sorted(usable["date"].unique())

    per_budget: dict[str, list[dict]] = {str(b): [] for b in budgets}
    thresholds = []
    for day in days:
        cur = usable[usable["date"] == day]
        n_cur = len(cur)
        pos = int(cur["is_laundering"].sum())
        y = cur["is_laundering"].to_numpy()
        root = cur["chain_root"].to_numpy()
        in_chain = cur["chain_len"].to_numpy() > 1
        # laundering edges that are in a real chain (len > 1)
        chained_pos = in_chain & (y == 1)
        chained_roots = root[chained_pos]
        n_chained_pos = int(chained_pos.sum())
        # One sort per day, reused for every budget.
        # Stable sort on (-score, row) so ties break the same way on every rebuild.
        order = np.lexsort((cur["row"].to_numpy(),
                            -cur["calibrated"].to_numpy()))
        # Accounts touched by any alerted edge, used for neighbourhood recall below.
        src = cur["from_account"].to_numpy()
        dst = cur["to_account"].to_numpy()
        for b in budgets:
            head = order[:b]                       # positions of the top-b rows
            tp = int(y[head].sum())
            alerted_roots = root[head][y[head] == 1]
            # chains with >=1 laundering edge that no alerted laundering edge touches
            lost_roots = np.setdiff1d(chained_roots, alerted_roots, assume_unique=False)
            lost_edges = int(np.isin(chained_roots, lost_roots).sum())
            # Neighbourhood recall: of the laundering edges this budget misses, how many
            # share an account with an edge this budget *did* alert that day. The chain
            # metric above is near-saturated on this data (most laundering edges have no
            # laundering successor within 24h), so this is the decision-useful measure of
            # "does an alert put the analyst next to the rest of the path".
            alerted_accts = np.union1d(src[head], dst[head])
            missed = np.ones(len(cur), dtype=bool)
            missed[head] = False
            missed_pos = (y == 1) & missed
            n_missed_pos = int(missed_pos.sum())
            near = ((np.isin(src, alerted_accts)) | (np.isin(dst, alerted_accts))) \
                & missed_pos
            n_near = int(near.sum())
            thr = float(cur["calibrated"].to_numpy()[head].min())
            per_budget[str(b)].append({
                "day": str(pd.Timestamp(day).date()),
                "n": int(n_cur),
                "positives": pos,
                "budget": b,
                "threshold": thr,
                "alerts": int(len(head)),
                "true_positives": tp,
                "precision": float(tp / max(len(head), 1)),
                "recall": float(tp / pos) if pos else None,
                "chained_positives": n_chained_pos,
                "chains_with_laundering": int(len(np.unique(chained_roots))),
                "chains_fully_missed": int(len(lost_roots)),
                "chained_edges_lost": lost_edges,
                "missed_positives": n_missed_pos,
                "missed_positives_near_alert": n_near,
                "neighbourhood_recall": (float(n_near / n_missed_pos)
                                         if n_missed_pos else None),
            })
            thresholds.append({"day": str(pd.Timestamp(day).date()), "budget": b,
                               "threshold": thr})
    return {"per_budget": per_budget,
            "thresholds": thresholds,
            "days": [str(pd.Timestamp(d).date()) for d in days]}


def summarise(per_budget: dict[str, list[dict]]) -> dict:
    out = {}
    for b, rows in per_budget.items():
        df = pd.DataFrame(rows)
        n_alerted = int(df["alerts"].sum())
        tp = int(df["true_positives"].sum())
        pos = int(df["positives"].sum())
        out[b] = {
            "n_days": int(len(df)),
            "alerts_total": n_alerted,
            "true_positives_total": tp,
            "positives_total": pos,
            "precision_pooled": float(tp / n_alerted) if n_alerted else None,
            "recall_pooled": float(tp / pos) if pos else None,
            "precision_mean": float(df["precision"].mean()),
            "recall_mean": float(df["recall"].mean()),
            "precision_min": float(df["precision"].min()),
            "precision_max": float(df["precision"].max()),
            "chained_positives_total": int(df["chained_positives"].sum()),
            "chained_edges_lost_total": int(df["chained_edges_lost"].sum()),
            "chains_with_laundering": int(df["chains_with_laundering"].sum()),
            "chains_fully_missed": int(df["chains_fully_missed"].sum()),
            "missed_positives_total": int(df["missed_positives"].sum()),
            "missed_positives_near_alert_total": int(
                df["missed_positives_near_alert"].sum()),
            "neighbourhood_recall_pooled": (
                float(df["missed_positives_near_alert"].sum()
                      / df["missed_positives"].sum())
                if df["missed_positives"].sum() else None),
        }
    return out


# --- Main --------------------------------------------------------------------------


def _rss_mb() -> float:
    """Resident set size in MB, from /proc/self/status (Linux only)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    args = p.parse_args(argv)

    if not SCORES_PATH.exists():
        print(f"missing {SCORES_PATH}; run src/models/model_ladder.py --rung 2")
        return 2

    scores = pd.read_parquet(SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    print(f"loaded {len(scores):,} scores over {scores['date'].nunique()} days "
          f"(RSS {_rss_mb():.0f} MB)")

    scored = calibrated_scores_by_day(scores)
    del scores
    gc.collect()
    print(f"calibrated by prior day (RSS {_rss_mb():.0f} MB)")

    chain_root, chain_len = build_chains()
    n_chains = int(len(np.unique(chain_root)))
    print(f"chains built: {n_chains:,} chains, "
          f"{int((chain_len > 1).sum()):,} edges in multi-hop chains "
          f"(RSS {_rss_mb():.0f} MB)")

    # ``row`` in the scores is the canonical-table row index; chain arrays are indexed by
    # the same position, so attach by integer gather - no merge on ids needed.
    rows = scored["row"].to_numpy()
    # Recode chain roots to dense 0..k-1 so the per-day chain comparisons are int codes.
    roots_present = np.unique(chain_root[rows])
    code_map = np.full(len(chain_root), -1, dtype=np.int64)
    code_map[roots_present] = np.arange(len(roots_present), dtype=np.int64)
    scored["chain_root"] = code_map[chain_root[rows]]
    scored["chain_len"] = chain_len[rows].astype(np.int64)
    # Neighbourhood recall needs the accounts each edge touches; read them in canonical
    # row order and gather by the same ``row`` index (canonical table is in that order).
    accts = pq.read_table(CANONICAL_PATH, columns=["from_account", "to_account"])
    scored["from_account"] = accts["from_account"].to_numpy()[rows]
    scored["to_account"] = accts["to_account"].to_numpy()[rows]
    del accts, chain_root, chain_len, rows, code_map, roots_present
    gc.collect()
    print(f"chains attached (RSS {_rss_mb():.0f} MB)")

    result = evaluate_budgets(scored, tuple(args.budgets))
    summary = summarise(result["per_budget"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"budgets": list(args.budgets), "days": result["days"],
               "summary": summary, "per_budget": result["per_budget"],
               "chain_window": str(CHAIN_WINDOW)}
    (OUT_DIR / "alerting.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(result["thresholds"]).to_parquet(
        OUT_DIR / "alert_thresholds.parquet", index=False)
    (OUT_DIR / "alerting.md").write_text(render_md(payload))

    print()
    for b, s in summary.items():
        print(f"budget {b:>5}: precision {s['precision_pooled']:.4f}  "
              f"recall {s['recall_pooled']:.4f}  TP {s['true_positives_total']}  "
              f"neighbourhood recall {s['neighbourhood_recall_pooled']:.4f} "
              f"({s['missed_positives_near_alert_total']}"
              f"/{s['missed_positives_total']} missed positives sit next to an alert)")
    print(f"\nwrote {OUT_DIR / 'alerting.md'}")
    return 0


def render_md(p: dict) -> str:
    s = p["summary"]
    lines = [
        "# Stage 5 - Alerting policy",
        "",
        "Thresholds come from a **budget** (alerts an analyst can review per day), not a",
        "probability cutoff. Within each validation day the top-N rows by calibrated Rung 2",
        "score are alerted; the Nth score is that day's threshold. The calibrator is",
        "fitted on the previous validation day, so 2022-09-03 (no earlier day) is excluded",
        "from every budget number.",
        "",
        f"Chain definition: account-contiguous transfers (`a -> b -> c`), consecutive hops",
        f"at most {p['chain_window']} apart. `chained edges lost` counts laundering edges on",
        "a chain where this budget alerts **none** of the chain's edges.",
        "",
        "**The operative metric is neighbourhood recall.** Only 178 of 3,565 laundering",
        "edges have a laundering successor within 24h, so most 'chains' are a laundering hop",
        "followed by ordinary transfers and the chain metric saturates. Neighbourhood recall",
        "asks the question an analyst actually has: of the laundering edges this budget",
        "misses, what share share an account with an edge it *did* alert that day? A high",
        "value means the alert queue puts the analyst next to the rest of the path even when",
        "it does not flag it directly.",
        "",
        "## Pooled over usable days",
        "",
        "| Budget | alerts | true positives | precision | recall | neighbourhood recall | chained lost |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in sorted(s, key=int):
        d = s[b]
        lines.append(
            f"| {b} | {d['alerts_total']:,} | {d['true_positives_total']} "
            f"| {d['precision_pooled']:.4f} | {d['recall_pooled']:.4f} "
            f"| {d['neighbourhood_recall_pooled']:.4f} "
            f"({d['missed_positives_near_alert_total']}/{d['missed_positives_total']}) "
            f"| {d['chained_edges_lost_total']} |")
    lines += ["", "## Per day", ""]
    for b in sorted(p["per_budget"], key=int):
        lines += [f"### Budget {b}", "",
                  "| day | n | pos | threshold | TP | precision | recall | nbr recall | chained lost |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in p["per_budget"][b]:
            lines.append(
                f"| {r['day']} | {r['n']:,} | {r['positives']} | {r['threshold']:.6f} "
                f"| {r['true_positives']} | {r['precision']:.4f} | {r['recall']:.4f} "
                f"| {r['neighbourhood_recall']:.4f} "
                f"({r['missed_positives_near_alert']}/{r['missed_positives']}) "
                f"| {r['chained_edges_lost']}/{r['chained_positives']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
