"""Stage 2, item 1: near-duplicate analysis.

Question the EDA left open: ~80% of rows share ``(from, to, amount)`` with another row.
Are these recurring payments or simulator padding? The decision drives whether velocity
features count raw transactions or distinct relationships.

Measures, as the note demands:

1. distribution of repeat counts for each ``(from_account, to_account, amount)`` key
2. spacing of timestamps within a repeating group - regular spacing means scheduled
   payments, random spacing means padding
3. laundering rate among repeated vs non-repeated rows
4. whether USD 0.00 amounts are involved
5. effect of the raw-vs-distinct choice on one concrete velocity feature
   (``txn_count`` per account)

The amount key uses ``Amount Paid`` in its own currency, not ``amount_usd``: the saved fx
tables are a Stage 2 deliverable and this item must not depend on them. Currency is part
of the key for the same reason - 100 USD and 100 GBP are not the same payment.

Run::

    /home/dan/miniconda3/envs/learn_env/bin/python src/eda_item1_duplicates.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.measure_daily_table import TRANS_PATH, load_transactions  # noqa: E402

KEY = ["from_account", "to_account", "Amount Paid", "Payment Currency"]


def load_with_amounts(path: Path) -> pd.DataFrame:
    """Load transactions plus the raw paid amount and its currency.

    ``load_transactions`` drops the amount columns, so this reads them again with the
    same header handling. Kept separate to avoid touching the Stage 1 helper.
    """
    df = pd.read_csv(
        path,
        usecols=[
            "Timestamp",
            "Account",
            "Account.1",
            "Amount Paid",
            "Payment Currency",
            "Is Laundering",
        ],
        dtype={"Account": "str", "Account.1": "str"},
        parse_dates=["Timestamp"],
    )
    return df.rename(
        columns={
            "Account": "from_account",
            "Account.1": "to_account",
            "Is Laundering": "is_laundering",
        }
    )


def repeat_counts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per key with its repeat count, laundering count and amount."""
    grouped = df.groupby(KEY, observed=True).agg(
        n=("is_laundering", "size"),
        launder=("is_laundering", "sum"),
    )
    return grouped.reset_index()


def spacing_stats(df: pd.DataFrame, min_repeats: int = 3, sample_groups: int = 20000) -> dict:
    """Timestamp spacing within repeating groups.

    A scheduled recurring payment shows a *regular* gap (e.g. exactly 1 day, every day).
    Simulator padding shows gaps with no structure. We measure the coefficient of
    variation of the within-group gaps: near 0 = regular, large = irregular.

    Only groups with at least ``min_repeats`` occurrences are considered, since a pair of
    rows gives a single gap and no notion of regularity.
    """
    counts = df.groupby(KEY, observed=True).size()
    eligible = counts[counts >= min_repeats].index
    if len(eligible) == 0:
        return {"n_groups": 0}

    if len(eligible) > sample_groups:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(eligible), size=sample_groups, replace=False)
        eligible = eligible[pick]

    sub = df.set_index(KEY).loc[eligible].reset_index()
    sub = sub.sort_values(KEY + ["Timestamp"])
    gap_secs = sub.groupby(KEY, observed=True)["Timestamp"].diff().dt.total_seconds()
    # One representative CV per group: std/mean of its gaps.
    per_group = sub.assign(gap=gap_secs).groupby(KEY, observed=True)["gap"].agg(
        ["mean", "std", "count"]
    )
    per_group = per_group.dropna(subset=["mean"])
    per_group = per_group[per_group["mean"] > 0]
    per_group["cv"] = per_group["std"].fillna(0) / per_group["mean"]

    return {
        "n_groups": int(len(per_group)),
        "cv_median": float(per_group["cv"].median()),
        "cv_p10": float(per_group["cv"].quantile(0.10)),
        "cv_p90": float(per_group["cv"].quantile(0.90)),
        "groups_cv_below_0.01": float((per_group["cv"] < 0.01).mean()),
        "median_gap_hours": float(per_group["mean"].median() / 3600),
        "median_repeats": float(per_group["count"].median() + 1),
    }


def laundering_by_repeat(df: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Laundering rate by repeat-count bucket, and repeated vs not."""
    merged = df.merge(counts[KEY + ["n"]], on=KEY, how="left")
    buckets = pd.cut(
        merged["n"],
        bins=[0, 1, 2, 5, 10, 100, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-100", "100+"],
    )
    out = (
        merged.groupby(buckets, observed=True)
        .agg(rows=("is_laundering", "size"), positives=("is_laundering", "sum"))
        .assign(pos_rate_pct=lambda d: d["positives"] / d["rows"] * 100)
    )
    return out


def zero_amounts(df: pd.DataFrame) -> dict:
    """How much of the file is USD/paid 0.00, and is it laundered."""
    zero = df["Amount Paid"] == 0
    return {
        "zero_rows": int(zero.sum()),
        "zero_pct": float(zero.mean() * 100),
        "zero_pos_rate_pct": float(df.loc[zero, "is_laundering"].mean() * 100) if zero.any() else 0.0,
        "nonzero_pos_rate_pct": float(df.loc[~zero, "is_laundering"].mean() * 100),
        "zero_amounts_distinct": int(df.loc[zero, "Amount Paid"].nunique()),
    }


def velocity_effect(df: pd.DataFrame, counts: pd.DataFrame) -> dict:
    """How much a raw-count velocity feature differs from a distinct-relationship one.

    If padding is common, ``raw count`` and ``distinct (from,to,amount) count`` diverge
    sharply, and the raw version would let the simulator inflate a node's activity.
    """
    raw = pd.concat([df["from_account"], df["to_account"]], ignore_index=True).value_counts()
    distinct = (
        pd.concat(
            [counts[["from_account"]].rename(columns={"from_account": "acct"}),
             counts[["to_account"]].rename(columns={"to_account": "acct"})],
            ignore_index=True,
        )
        .groupby("acct", observed=True)
        .size()
    )
    common = raw.index.intersection(distinct.index)
    raw_c = raw.loc[common].astype(float)
    dis_c = distinct.loc[common].astype(float)
    return {
        "accounts": int(len(common)),
        "raw_mean": float(raw_c.mean()),
        "distinct_mean": float(dis_c.mean()),
        "ratio_mean": float((raw_c / dis_c).mean()),
        "p99_raw_over_distinct": float((raw_c / dis_c).quantile(0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=TRANS_PATH)
    args = parser.parse_args()

    df = load_with_amounts(args.path)
    print(f"rows: {len(df):,}   distinct keys: {df[KEY].drop_duplicates().shape[0]:,}")

    counts = repeat_counts(df)
    print("\n--- repeat count distribution (per key) ---")
    print(counts["n"].value_counts().sort_index().head(15).to_string())
    print(f"keys: {len(counts):,}   rows in keys appearing once: {int((counts['n'] == 1).sum()):,}")
    print(f"share of rows in keys with n>=2: {(counts.loc[counts['n'] >= 2, 'n'].sum() / len(df)) * 100:.2f}%")

    print("\n--- timestamp spacing within repeating groups ---")
    for k, v in spacing_stats(df).items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")

    print("\n--- laundering rate by repeat bucket ---")
    print(laundering_by_repeat(df, counts).round(4).to_string())

    print("\n--- zero amounts ---")
    for k, v in zero_amounts(df).items():
        print(f"  {k:24s} {v}")

    print("\n--- raw vs distinct velocity counts ---")
    for k, v in velocity_effect(df, counts).items():
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
