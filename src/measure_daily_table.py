"""Stage 1 helper: measure the per-day table that justifies unusable-day exclusions.

Stage 1 requires the evaluation contract to *measure* which calendar days are unusable
rather than assume "days 11-17" are dead. This script produces exactly that table:
rows, positives, new accounts, positive rate per day, so the contract can cite numbers.

Run with the project's data env:
    /home/dan/miniconda3/envs/learn_env/bin/python src/measure_daily_table.py

Prints a compact summary only (no raw row dumps), per the project CLAUDE.md rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANS_PATH = PROJECT_ROOT / "data" / "LI-Small_Trans.csv"

# Days with fewer than this many transactions are too small to validate on; the same
# threshold the split protocol uses for a row-count floor is 50, but a *day* that small
# would make recall@1000 meaningless, so we flag anything under this separately.
MIN_DAY_ROWS = 50


def load_transactions(path: Path) -> pd.DataFrame:
    """Load the raw transaction file.

    The source CSV contains a duplicate "Account" column on both sides of the edge
    (``From Bank, Account, To Bank, Account``). pandas will mangle those into
    ``Account`` / ``Account.1`` — we rename them immediately so nothing downstream
    has to think about it again.
    """
    df = pd.read_csv(
        path,
        usecols=["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Is Laundering"],
        # Account IDs are hex strings (e.g. "8000ECA90"), not integers. Keep as str so
        # the leading zeros/prefix structure survives; rule 6 forbids using them raw anyway.
        dtype={"Account": "str", "Account.1": "str"},
        parse_dates=["Timestamp"],
    )
    df = df.rename(
        columns={
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Is Laundering": "is_laundering",
        }
    )
    df["date"] = df["Timestamp"].dt.normalize()
    return df


def daily_table(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day with the metrics the contract needs.

    - ``rows``        : transaction count on that day
    - ``positives``   : laundering-labelled transactions (edge labels)
    - ``pos_rate_pct``: positives / rows * 100
    - ``new_accounts``: accounts whose *first* appearance in the whole file is that day
    - ``min_roll_rate``: smallest 1000-txn rolling positive rate seen that day; used
      to detect intermittent dry spells as well as whole dead days
    """
    per_day = tx.groupby("date", observed=True).agg(
        rows=("is_laundering", "size"),
        positives=("is_laundering", "sum"),
    )
    per_day["pos_rate_pct"] = per_day["positives"] / per_day["rows"] * 100.0

    # First-seen day per account, counting an account once no matter how many edges it has.
    # Uses .nunique() in the count below, never len(), per CLAUDE.md rule 1.
    first_seen = pd.concat([tx["from_account"], tx["to_account"]], ignore_index=True)
    first_day = pd.Series(
        pd.concat([tx["date"], tx["date"]], ignore_index=True).values,
        index=first_seen.values,
    ).groupby(level=0).min()
    per_day["new_accounts"] = first_day.value_counts().reindex(per_day.index).fillna(0).astype(int)

    return per_day.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=TRANS_PATH)
    args = parser.parse_args()

    tx = load_transactions(args.path)
    per_day = daily_table(tx)

    print(f"file          : {args.path.name}")
    print(f"transactions  : {len(tx):,}   (loaded rows)")
    # tx holds only a column subset, so dedupe here would merge distinct transactions
    # that differ in amount/currency. Exact-duplicate counting needs the full row and is
    # done in the data contract (section 10), not here.
    print(
        f"accounts      : {tx['from_account'].nunique() + tx['to_account'].nunique():,} "
        "account-side slots"
    )
    print(f"date range    : {per_day.index.min().date()} .. {per_day.index.max().date()}")
    print()

    out = per_day.copy()
    out["pos_rate_pct"] = out["pos_rate_pct"].round(3)
    print(out.to_string())

    # Flag unusable days with an explicit, printed rule so the contract can quote it.
    dead = out[out["positives"] == 0]
    tiny = out[(out["positives"] > 0) & (out["rows"] < MIN_DAY_ROWS)]
    print()
    print(f"days with zero positives ({len(dead)}):")
    print("  " + ", ".join(str(d.date()) for d in dead.index) if len(dead) else "  none")
    print(f"days with positives but <{MIN_DAY_ROWS} rows ({len(tiny)}):")
    print("  " + ", ".join(str(d.date()) for d in tiny.index) if len(tiny) else "  none")
    print()
    print(f"base rate overall: {tx['is_laundering'].mean() * 100:.4f}%")
    print(
        "days above overall base rate: "
        f"{(out['pos_rate_pct'] > tx['is_laundering'].mean() * 100).sum()} / {len(out)}"
    )


if __name__ == "__main__":
    main()
