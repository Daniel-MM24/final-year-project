"""Stage 1 reproduction: rerun the node-level LightGBM baseline through ``eval_protocol``.

What this proves
----------------
Stage 1 asked that the protocol reproduce the existing node-level mean PR-AUC of 0.0408.
The original run lives in ``baseline_lightgbm.py`` and scored only 4 of 10 planned folds
(folds 2022-09-07 .. 2022-09-10), because the earlier folds had no positive accounts.

This script:

1. rebuilds the inputs from raw data + the saved fx tables (the original script depended
   on notebook state), including ``amount_usd`` exactly as ``docs/data_contract.md`` s3
   defines it;
2. builds **as-of** node features per fold - every feature uses only transactions strictly
   before the fold's training cutoff, so nothing leaks;
3. fits LightGBM on the training nodes and scores the validation-day nodes;
4. scores each fold through ``eval_protocol.score_fold`` and aggregates with
   ``summarise_folds``.

Reproducing the number exactly is not the goal - the goal is showing ``eval_protocol``
produces the same *kind* of measurement from the same folds. A close match (within fold
ordering and magnitude) confirms the protocol; an exact match is not expected because the
original features were built in a notebook with unspecified details.

Note on units: this is a **node-level** baseline. It is run only to validate the
protocol. The project's headline metric is edge-level.

Run::

    /home/dan/miniconda3/envs/learn_env/bin/python src/reproduce_baseline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_protocol import Fold, score_fold, summarise_folds  # noqa: E402
from src.measure_daily_table import TRANS_PATH  # noqa: E402

FX_USD = PROJECT_ROOT / "eda_output" / "fx_usd_rate.csv"
FX_BTC = PROJECT_ROOT / "eda_output" / "fx_btc_daily_rate.csv"
MIN_TRAIN_DAYS = 5
EMBARGO_DAYS = 1
TARGET_FOLDS = ["2022-09-07", "2022-09-08", "2022-09-09", "2022-09-10"]


def load_edges(path: Path) -> pd.DataFrame:
    """Load transactions with amount_usd computed per the data contract (s3)."""
    df = pd.read_csv(
        path,
        usecols=[
            "Timestamp", "Account", "Account.1", "Amount Paid", "Payment Currency",
            "Amount Received", "Receiving Currency", "Is Laundering",
        ],
        dtype={"Account": "str", "Account.1": "str"},
        parse_dates=["Timestamp"],
    ).rename(
        columns={
            "Account": "from_account", "Account.1": "to_account",
            "Amount Paid": "amount_paid", "Payment Currency": "currency_paid",
            "Amount Received": "amount_received", "Receiving Currency": "currency_received",
            "Is Laundering": "is_laundering",
        }
    )
    df["date"] = df["Timestamp"].dt.normalize()

    usd_rate = pd.read_csv(FX_USD, index_col=0)["usd_rate"]
    btc_rate = pd.read_csv(FX_BTC, index_col=0)["btc_usd_rate"]
    btc_rate.index = pd.to_datetime(btc_rate.index)

    def to_usd(amount: pd.Series, currency: pd.Series, dates: pd.Series) -> pd.Series:
        is_btc = currency == "Bitcoin"
        fiat = amount * currency.map(usd_rate)
        btc = amount * dates.map(btc_rate)
        return np.where(is_btc, btc, fiat)

    df["amount_paid_usd"] = to_usd(df["amount_paid"], df["currency_paid"], df["date"])
    df["amount_received_usd"] = to_usd(df["amount_received"], df["currency_received"], df["date"])
    return df


def node_table(edges: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct node with lifetime label and first-seen date."""
    first = edges.groupby("from_account", observed=True)["date"].min()
    first_to = edges.groupby("to_account", observed=True)["date"].min()
    first_seen = pd.concat([first, first_to]).groupby(level=0).min()

    launder_out = edges.groupby("from_account", observed=True)["is_laundering"].sum()
    launder_in = edges.groupby("to_account", observed=True)["is_laundering"].sum()
    node_launder = launder_out.add(launder_in, fill_value=0)

    nodes = pd.DataFrame({"date_first_seen": first_seen, "node_launder": node_launder.fillna(0)})
    nodes["is_laundering_node"] = (nodes["node_launder"] > 0).astype(int)
    return nodes


def as_of_features(edges: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per-node features from transactions strictly before ``cutoff`` only.

    Contract s9: no whole-file aggregates. Every column here is built from ``hist``.
    Fan-out/fan-in count **distinct** partners (contract s6.2), not raw rows.
    """
    hist = edges[edges["date"] < cutoff]
    out = hist.groupby("from_account", observed=True).agg(
        out_degree=("to_account", "size"),
        out_unique=("to_account", "nunique"),
        out_amount=("amount_paid_usd", "sum"),
        out_amount_mean=("amount_paid_usd", "mean"),
        out_amount_std=("amount_paid_usd", "std"),
    )
    inn = hist.groupby("to_account", observed=True).agg(
        in_degree=("from_account", "size"),
        in_unique=("from_account", "nunique"),
        in_amount=("amount_received_usd", "sum"),
        in_amount_mean=("amount_received_usd", "mean"),
        in_amount_std=("amount_received_usd", "std"),
    )
    feats = out.join(inn, how="outer").fillna(0.0)
    feats["degree"] = feats["out_degree"] + feats["in_degree"]
    feats["net_flow"] = feats["in_amount"] - feats["out_amount"]
    feats["flow_ratio"] = feats["in_amount"] / (feats["out_amount"] + 1.0)
    feats["fan_out"] = feats["out_unique"]
    feats["fan_in"] = feats["in_unique"]
    feats["fan_ratio"] = (feats["fan_in"] + 1.0) / (feats["fan_out"] + 1.0)
    return feats


FEATURE_COLS = [
    "out_degree", "out_unique", "out_amount", "out_amount_mean", "out_amount_std",
    "in_degree", "in_unique", "in_amount", "in_amount_mean", "in_amount_std",
    "degree", "net_flow", "flow_ratio", "fan_out", "fan_in", "fan_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=TRANS_PATH)
    args = parser.parse_args()

    edges = load_edges(args.path)
    nodes = node_table(edges)
    days = sorted(edges["date"].unique())
    print(f"edges {len(edges):,}  nodes {len(nodes):,}  days {len(days)}")
    print(f"node base rate: {nodes['is_laundering_node'].mean() * 100:.4f}%")

    results = []
    for day_str in TARGET_FOLDS:
        val_day = pd.Timestamp(day_str)
        first_usable = pd.Timestamp(days[0]) + pd.Timedelta(days=MIN_TRAIN_DAYS)
        if val_day < first_usable:
            print(f"{day_str}: skipped, not enough train days")
            continue
        train_cutoff = val_day - pd.Timedelta(days=EMBARGO_DAYS)
        # Nodes split by first-seen date, as in the original baseline: train on nodes
        # that appeared at least EMBARGO_DAYS before the validation day.
        train_idx = nodes.index[nodes["date_first_seen"] <= train_cutoff - pd.Timedelta(days=1)]
        val_idx = nodes.index[nodes["date_first_seen"] == val_day]
        if len(val_idx) == 0:
            print(f"{day_str}: no validation nodes")
            continue

        feats = as_of_features(edges, train_cutoff)
        X_tr = feats.reindex(train_idx).fillna(0.0)[FEATURE_COLS]
        y_tr = nodes.loc[train_idx, "is_laundering_node"]
        X_va = feats.reindex(val_idx).fillna(0.0)[FEATURE_COLS]
        y_va = nodes.loc[val_idx, "is_laundering_node"]

        # eval_protocol is written for edge-level masks over one frame. A node-level fold
        # is already a materialised split, so wrap just the validation nodes in a frame
        # whose masks select everything; the scoring code path is then identical to the
        # edge protocol and the aggregation below is the same.
        node_frame = pd.DataFrame({"is_laundering": y_va.values}, index=val_idx)
        fold = Fold(
            name=f"fold-{day_str}",
            validation_day=val_day,
            train_mask=pd.Series(True, index=node_frame.index),
            validation_mask=pd.Series(True, index=node_frame.index),
            train_max_date=train_cutoff - pd.Timedelta(days=1),
        )

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", verbose=-1, random_state=0,
        )
        model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_va)[:, 1]

        res = score_fold(fold, scores, labels=node_frame["is_laundering"])
        results.append(res)
        mark = "" if not res.skipped else f" [skipped: {res.reason}]"
        print(
            f"{day_str}: train {len(train_idx):,}  val {len(val_idx):,}  "
            f"pos {res.n_positives}  base {res.base_rate * 100:.3f}%  "
            f"PR-AUC {res.pr_auc:.4f}{mark}"
        )

    summary = summarise_folds(results)
    print("\n--- summary ---")
    print(f"folds used: {summary['n_used']} / {summary['n_folds']}   skipped: {summary['skipped']}")
    print(f"node-level mean PR-AUC: {summary['pr_auc_mean']:.4f}  (std {summary['pr_auc_std']:.4f})")
    print(f"original baseline      : 0.0408  (std 0.0178, 4 of 10 folds)")
    print(f"weighted lift: {summary['lift_weighted']:.2f}")


if __name__ == "__main__":
    main()
