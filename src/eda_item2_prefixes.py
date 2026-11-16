"""Stage 2, item 2: ID-prefix scan at 4 and 8 characters.

The earlier EDA scanned 6-character prefixes by frequency. Frequency alone is not a
signal - the simulator may allocate IDs densely, so the biggest prefix is just the biggest
bank. This ranks prefixes by **behavioural anomaly** instead:

- ``fanout_lift``: median distinct counterparties per account with this prefix, divided
  by the same median for accounts without it. A prefix whose accounts reach far more
  distinct people than normal is a fabricated-entity marker.
- ``launder_rate_pct``: share of edges touched by a prefix-account that are labelled.
- ``launder_lift``: that rate divided by the overall edge base rate.

A prefix is recommended for exclusion when it is anomalous on *behaviour* even if it is
not the most frequent - which is what rule 6 in Shared Context warns about (the 10042B
family is large but must never become a model input).

Run::

    /home/dan/miniconda3/envs/learn_env/bin/python src/eda_item2_prefixes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.measure_daily_table import TRANS_PATH  # noqa: E402

MIN_ACCOUNTS = 20
"""Below this many accounts a prefix's stats are too noisy to rank."""

ANOMALY_LIFT = 2.0
"""A prefix is flagged when fanout or laundering lift reaches this multiple."""


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["Account", "Account.1", "Is Laundering"],
        dtype={"Account": "str", "Account.1": "str"},
    )
    return df.rename(columns={"Account": "from_account", "Account.1": "to_account", "Is Laundering": "is_laundering"})


def account_table(edges: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct account with fan-out and laundering exposure.

    Fan-out counts *distinct* counterparties (item 1's finding: raw counts are padded).
    Laundering exposure counts edges in either direction that are labelled.
    """
    out = edges.groupby("from_account", observed=True).agg(
        out_edges=("to_account", "size"),
        out_partners=("to_account", "nunique"),
        out_launder=("is_laundering", "sum"),
    )
    inn = edges.groupby("to_account", observed=True).agg(
        in_edges=("from_account", "size"),
        in_partners=("from_account", "nunique"),
        in_launder=("is_laundering", "sum"),
    )
    accts = out.join(inn, how="outer").fillna(0)
    accts["partners"] = accts["out_partners"] + accts["in_partners"]
    accts["edges"] = accts["out_edges"] + accts["in_edges"]
    accts["launder_edges"] = accts["out_launder"] + accts["in_launder"]
    return accts


def scan(
    accts: pd.DataFrame,
    base_rate: float,
    width: int,
    min_accounts: int = MIN_ACCOUNTS,
) -> pd.DataFrame:
    """Rank every ``width``-character prefix by behavioural anomaly."""
    prefix = accts.index.str[:width]
    global_median_partners = accts["partners"].median()

    rows = []
    for pfx, group in accts.groupby(prefix, observed=True):
        if len(group) < min_accounts:
            continue
        med_partners = group["partners"].median()
        edge_count = group["edges"].sum()
        launder_edges = group["launder_edges"].sum()
        rows.append(
            {
                "prefix": pfx,
                "accounts": len(group),
                "edges": edge_count,
                "median_partners": med_partners,
                "fanout_lift": med_partners / global_median_partners if global_median_partners else np.nan,
                "launder_rate_pct": launder_edges / edge_count * 100 if edge_count else np.nan,
                "launder_lift": (launder_edges / edge_count / base_rate) if edge_count else np.nan,
                "pct_accts_laundering": (group["launder_edges"] > 0).mean() * 100,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["anomaly_score"] = np.maximum(out["fanout_lift"], out["launder_lift"])
    return out.sort_values("anomaly_score", ascending=False)


def spread_check(accts: pd.DataFrame, prefix_series: pd.Series, pfx: str) -> dict:
    """Is a prefix's laundering spread across its accounts or driven by a few?

    Returns the share of its laundering edges held by the top 1% of its accounts. Near
    100% means the prefix is a handful of injected accounts, not a systemic pattern.
    """
    group = accts[prefix_series == pfx]
    launder = group["launder_edges"].sort_values(ascending=False)
    total = launder.sum()
    if total == 0:
        return {"share_top1pct": np.nan, "n_laundering_accts": 0}
    k = max(1, int(np.ceil(len(group) * 0.01)))
    return {
        "share_top1pct": float(launder.iloc[:k].sum() / total * 100),
        "n_laundering_accts": int((launder > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=TRANS_PATH)
    args = parser.parse_args()

    edges = load_edges(args.path)
    base_rate = float(edges["is_laundering"].mean())
    accts = account_table(edges)
    print(f"accounts: {len(accts):,}   distinct edges: {len(edges):,}   base rate: {base_rate * 100:.4f}%")
    print(f"id lengths: {accts.index.str.len().value_counts().to_dict()}")

    # Every account ID is exactly 9 characters, so an 8-character prefix is unique per
    # account: the scan degenerates into 705,903 single-account groups, each with one
    # value for every statistic. It measures nothing and is not run. The note asks for
    # "4 and 8"; 4 is the meaningful width, 8 is recorded as degenerate.
    print("\nwidth 8: SKIPPED - IDs are 9 chars, so 8-char prefixes are unique per account.")

    flagged: list[str] = []
    for width in (4,):
        prefix_series = accts.index.str[:width]
        table = scan(accts, base_rate, width)
        print(f"\n--- top 15 prefixes by anomaly, width={width} (of {len(table)} with >={MIN_ACCOUNTS} accts) ---")
        cols = ["prefix", "accounts", "fanout_lift", "launder_rate_pct", "launder_lift", "pct_accts_laundering"]
        print(table[cols].head(15).round(2).to_string(index=False))

        for _, row in table.iterrows():
            if row["anomaly_score"] >= ANOMALY_LIFT:
                sp = spread_check(accts, prefix_series, row["prefix"])
                flagged.append(row["prefix"])
                print(
                    f"  FLAG {row['prefix']:8s} fanout x{row['fanout_lift']:.1f}  "
                    f"launder x{row['launder_lift']:.1f}  "
                    f"top1% holds {sp['share_top1pct']:.0f}% of laundering edges "
                    f"({sp['n_laundering_accts']} accts)"
                )

    print(f"\nflagged prefixes ({len(flagged)}): {sorted(set(flagged))}")


if __name__ == "__main__":
    main()
