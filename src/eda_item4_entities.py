"""Stage 2, item 4: Entity ID check.

The earlier EDA asked whether multi-account entities concentrate laundering. A high
count of accounts per entity is not itself suspicious - legitimate corporations hold many
accounts. The test is conditional: *given* that an entity has many accounts, is the
laundering rate of its accounts higher?

Measures:

1. distribution of accounts per entity
2. laundering rate in accounts belonging to single- vs multi-account entities
3. correlation between entity size and proportion of its accounts touching laundering
4. intra-entity transactions: edges where sender and receiver share an Entity ID, and
   their laundering rate. Moving money between one's own accounts is a classic
   structuring signal, so this doubles as a prospective feature check.

Run::

    /home/dan/miniconda3/envs/learn_env/bin/python src/eda_item4_entities.py
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

ACCOUNTS_PATH = PROJECT_ROOT / "data" / "LI-Small_accounts.csv"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = pd.read_csv(ACCOUNTS_PATH, dtype=str)
    edges = pd.read_csv(
        TRANS_PATH,
        usecols=["Account", "Account.1", "Is Laundering"],
        dtype={"Account": str, "Account.1": str},
    ).rename(columns={"Account": "from_account", "Account.1": "to_account", "Is Laundering": "is_laundering"})
    return accounts, edges


def account_laundering(edges: pd.DataFrame) -> pd.Series:
    """Labelled edges touching each account, either direction."""
    out = edges.groupby("from_account", observed=True)["is_laundering"].sum()
    inn = edges.groupby("to_account", observed=True)["is_laundering"].sum()
    total = out.add(inn, fill_value=0)
    edge_count = (
        edges.groupby("from_account", observed=True).size()
        .add(edges.groupby("to_account", observed=True).size(), fill_value=0)
    )
    return (total / edge_count).rename("launder_share")


def entity_size_table(accounts: pd.DataFrame, accts: pd.DataFrame) -> pd.DataFrame:
    """One row per entity: number of accounts it holds and how many are laundering-exposed."""
    df = accounts.drop_duplicates(subset="Account Number").copy()
    df["launder_share"] = df["Account Number"].map(accts).fillna(0.0)
    df["exposed"] = df["launder_share"] > 0
    return df.groupby("Entity ID", observed=True).agg(
        n_accounts=("Account Number", "size"),
        n_exposed=("exposed", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    accounts, edges = load()
    base_rate = float(edges["is_laundering"].mean())
    print(f"edges: {len(edges):,}   base rate: {base_rate * 100:.4f}%")

    accts = account_laundering(edges)
    ents = entity_size_table(accounts, accts)
    print(f"entities: {len(ents):,}   accounts: {int(ents['n_accounts'].sum()):,}")
    print(f"accounts per entity: mean {ents['n_accounts'].mean():.2f}  median {ents['n_accounts'].median():.0f}  max {ents['n_accounts'].max()}")

    print("\n--- accounts-per-entity distribution ---")
    buckets = pd.cut(
        ents["n_accounts"], bins=[0, 1, 2, 5, 10, 50, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-50", "50+"],
        include_lowest=True,
    )
    dist = ents.groupby(buckets, observed=True).agg(
        entities=("n_accounts", "size"), accounts=("n_accounts", "sum")
    )
    dist["entity_share_pct"] = (dist["entities"] / len(ents) * 100).round(2)
    print(dist.to_string())

    print("\n--- laundering exposure by entity size (account-weighted) ---")
    grouped = ents.groupby(buckets, observed=True).agg(
        accounts=("n_accounts", "sum"),
        exposed_accounts=("n_exposed", "sum"),
        entities_with_exposure=("n_exposed", lambda s: int((s > 0).sum())),
        entities=("n_accounts", "size"),
    )
    grouped["exposed_acct_pct"] = (grouped["exposed_accounts"] / grouped["accounts"] * 100).round(3)
    grouped["entities_with_exposure_pct"] = (grouped["entities_with_exposure"] / grouped["entities"] * 100).round(2)
    print(grouped[["accounts", "exposed_accounts", "exposed_acct_pct", "entities_with_exposure_pct"]].to_string())

    single = ents[ents["n_accounts"] == 1]
    multi = ents[ents["n_accounts"] > 1]
    print(
        f"\nsingle-account entities: {len(single):,}  exposed accts {single['n_exposed'].sum() / single['n_accounts'].sum() * 100:.3f}%\n"
        f"multi-account entities : {len(multi):,}  exposed accts {multi['n_exposed'].sum() / multi['n_accounts'].sum() * 100:.3f}%"
    )
    corr = ents["n_accounts"].corr(ents["n_exposed"] / ents["n_accounts"], method="spearman")
    print(f"spearman(entity size, share of accounts exposed): {corr:.3f}")

    print("\n--- intra-entity transactions ---")
    ent_of = accounts.drop_duplicates(subset="Account Number").set_index("Account Number")["Entity ID"]
    from_ent = edges["from_account"].map(ent_of)
    to_ent = edges["to_account"].map(ent_of)
    intra = from_ent.notna() & to_ent.notna() & (from_ent == to_ent)
    print(f"intra-entity edges: {int(intra.sum()):,} ({intra.mean() * 100:.2f}% of matched edges)")
    if intra.any():
        intra_rate = float(edges.loc[intra, "is_laundering"].mean())
        inter_rate = float(edges.loc[from_ent.notna() & to_ent.notna() & ~intra, "is_laundering"].mean())
        print(f"  laundering rate intra: {intra_rate * 100:.4f}%   inter: {inter_rate * 100:.4f}%   lift: {intra_rate / inter_rate:.2f}x")

    print("\n--- top 10 entities by exposed-account count ---")
    top = ents.sort_values("n_exposed", ascending=False).head(10).copy()
    top["exposed_pct"] = (top["n_exposed"] / top["n_accounts"] * 100).round(1)
    print(top.to_string())


if __name__ == "__main__":
    main()
