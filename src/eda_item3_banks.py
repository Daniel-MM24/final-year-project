"""Stage 2, item 3: bank check - fuzzy grouping and laundering involvement.

The earlier EDA flagged names like "Crytpo Bank #57" as possible synthetic noise. A raw
count of such names proves nothing: the simulator generates many banks, so obscure names
are common. This script tests *involvement*, and tests it at the right unit.

1. Group bank names fuzzily: strip a trailing ``#N`` suffix, normalise case/whitespace,
   and check whether the base name of a questionable bank is a known real bank.
2. Compute the laundering rate of edges touching each bank and compare with the global
   edge base rate using a one-sided binomial test. A bank is involved only if its rate is
   significantly *above* base rate, not merely non-zero.
3. Check the account-level spread: if laundering edges at a bank are all concentrated in
   one or two accounts, the bank is a coincidence, not an actor.

Bonferroni-correct across the banks tested, since 27,652 banks at alpha=0.05 will produce
~1,380 false positives by chance alone.

Run::

    /home/dan/miniconda3/envs/learn_env/bin/python src/eda_item3_banks.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.measure_daily_table import TRANS_PATH  # noqa: E402

ACCOUNTS_PATH = PROJECT_ROOT / "data" / "LI-Small_accounts.csv"
ALPHA = 0.05
MIN_EDGES = 50
"""Banks below this many edges are not tested; the binomial test has no power."""

SUFFIX = re.compile(r"\s*#\s*\d+\s*$")
TYPOS = {"crytpo": "crypto", "banque": "bank", "banck": "bank"}


def normalise(name: str) -> str:
    """Strip ``#N``, lowercase, collapse whitespace, and fold known typos."""
    s = SUFFIX.sub("", str(name)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    for wrong, right in TYPOS.items():
        s = s.replace(wrong, right)
    return s


def bank_edges(accounts: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Edges credited to the bank of their sending account.

    Attribution is on the sender only: the source file identifies the bank of the
    account holding the transaction, and double-counting both sides would double the
    apparent rate for banks that appear on both ends of many edges.
    """
    # 4 account numbers appear in two banks each (rows 364085/569902 and two more); all
    # four carry fabricated-prefix Entity IDs. Drop them rather than let a duplicate index
    # silently decide which bank gets the edge.
    accounts = accounts.drop_duplicates(subset="Account Number", keep=False)
    bank_of = accounts.set_index("Account Number")["Bank Name"]
    df = edges.copy()
    df["bank"] = df["from_account"].map(bank_of)
    return df.dropna(subset=["bank"])


def test_banks(df: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    g = df.groupby("bank", observed=True).agg(
        edges=("is_laundering", "size"), positives=("is_laundering", "sum")
    )
    g = g[g["edges"] >= MIN_EDGES]
    g["rate"] = g["positives"] / g["edges"]
    g["lift"] = g["rate"] / base_rate
    n_tests = len(g)
    alpha_adj = ALPHA / max(n_tests, 1)
    g["p_value"] = [
        binomtest(int(p), int(n), base_rate, alternative="greater").pvalue
        for p, n in zip(g["positives"], g["edges"])
    ]
    g["significant"] = g["p_value"] < alpha_adj
    return g.sort_values("p_value")


def spread(df: pd.DataFrame, bank: str) -> dict:
    """Share of a bank's laundering edges held by its top 1% of accounts."""
    sub = df[df["bank"] == bank]
    per_acct = sub.groupby("from_account", observed=True)["is_laundering"].sum().sort_values(ascending=False)
    total = per_acct.sum()
    if total == 0:
        return {"share_top1pct": np.nan, "laundering_accts": 0, "accts": int(len(per_acct))}
    k = max(1, int(np.ceil(len(per_acct) * 0.01)))
    return {
        "share_top1pct": float(per_acct.iloc[:k].sum() / total * 100),
        "laundering_accts": int((per_acct > 0).sum()),
        "accts": int(len(per_acct)),
    }


def known_real_banks() -> set[str]:
    """Base names that look like real institutions - the comparison set.

    Taken from the frequent banks in this file; a bank whose normalised base name is in
    this set is an ordinary bank, not a synthetic one.
    """
    return {
        "first bank of danbury",
        "savings bank of madison",
        "national bank of columbus",
        "savings bank of the north",
        "first bank of the west",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=TRANS_PATH)
    args = parser.parse_args()

    accounts = pd.read_csv(ACCOUNTS_PATH, dtype=str)
    edges = pd.read_csv(
        args.path,
        usecols=["Account", "Account.1", "Is Laundering"],
        dtype={"Account": str, "Account.1": str},
    ).rename(columns={"Account": "from_account", "Account.1": "to_account", "Is Laundering": "is_laundering"})

    base_rate = float(edges["is_laundering"].mean())
    print(f"edges: {len(edges):,}   base rate: {base_rate * 100:.4f}%")

    df = bank_edges(accounts, edges)
    print(f"edges with a known sending bank: {len(df):,} ({len(df) / len(edges) * 100:.1f}%)")

    # 1. suffix structure
    names = accounts["Bank Name"]
    has_suffix = names.str.contains(SUFFIX, regex=True)
    print(f"\nbank names with #N suffix: {has_suffix.mean() * 100:.1f}% of accounts")
    real = known_real_banks()
    base_name = names.map(normalise)
    print(f"accounts at a known real bank: {(base_name.isin(real)).mean() * 100:.2f}%")

    # 2. per-bank significance
    res = test_banks(df, base_rate)
    print(f"\nbanks tested (>= {MIN_EDGES} edges): {len(res)}   Bonferroni alpha: {ALPHA / len(res):.2e}")
    print(f"significant above base rate: {int(res['significant'].sum())}")

    print("\n--- top 15 banks by significance ---")
    cols = ["edges", "positives", "rate", "lift", "p_value"]
    top = res.head(15).copy()
    top["rate"] = (top["rate"] * 100).round(3)
    top["lift"] = top["lift"].round(2)
    print(top[cols].to_string())

    print("\n--- do significant banks have #N names? ---")
    sig = res[res["significant"]]
    if len(sig):
        sig_suffix = names[names.isin(sig.index)].str.contains(SUFFIX, regex=True)
        print(f"significant banks: {len(sig)}   with #N suffix: {sig_suffix.mean() * 100:.0f}%")
        print("\nspread check on top 5 significant banks:")
        for bank in sig.head(5).index:
            sp = spread(df, bank)
            print(
                f"  {bank[:40]:40s} top1% holds {sp['share_top1pct']:.0f}% "
                f"({sp['laundering_accts']}/{sp['accts']} accts have any laundering)"
            )
    else:
        print("  none")

    # 3. baseline comparison: are ordinary banks equally "involved"?
    ordinary = res[res.index.isin(names[base_name.isin(real)].unique())]
    if len(ordinary):
        print("\n--- known real banks, same test ---")
        o = ordinary.head(5).copy()
        o["rate"] = (o["rate"] * 100).round(3)
        o["lift"] = o["lift"].round(2)
        print(o[["edges", "positives", "rate", "lift", "p_value", "significant"]].to_string())


if __name__ == "__main__":
    main()
