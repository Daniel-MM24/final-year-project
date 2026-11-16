"""
Tabular baseline: LightGBM on node-level features, evaluated fold-by-fold on a
rolling-origin split with a 1-day embargo.

Purpose: decide whether the planned RGCN/GAT edge classifier in PyTorch Geometric is
actually justified. If this baseline lands within a couple of PR-AUC points of the GNN,
the graph structure isn't adding much beyond degree/fan-out/amount-variability, and the
PyG build-out should be deprioritised.

Assumes this runs in the same notebook session as your EDA, after Section 9 (`nodes`,
`tx`, `dormant`) and Section 7 (`amount_usd`, `amount_received_usd`) already exist.

CRITICAL: every feature below is rebuilt from scratch for each fold, using only
transactions dated on or before that fold's cutoff. The global `nodes` table built in
Section 9 aggregates over the ENTIRE file - using it directly as features in a temporal
CV leaks each account's future behaviour into earlier folds' training data and inflates
every fold's PR-AUC. This was verified empirically: on a small fixture, one fold's PR-AUC
dropped from a perfect 1.0000 to 0.7818 once features were rebuilt as-of the correct
cutoff instead of sliced from the global table.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

FEATURES = ["out_amount_std", "in_amount", "fan_in", "fan_out",
            "cross_bank_frac", "currency_crossing"]
TARGET = "is_laundering_node"


def build_features_asof(tx_all, as_of_date, account_ids):
    """Node-level features using ONLY transactions dated <= as_of_date."""
    hist = tx_all[tx_all["date"] <= as_of_date]

    out_agg = (hist.groupby("from_account", observed=True)
                    .agg(out_amount_std=("amount_usd", "std"),
                         fan_out=("to_account", "nunique"))
                    .reindex(account_ids).fillna(0))
    in_agg = (hist.groupby("to_account", observed=True)
                   .agg(in_amount=("amount_received_usd", "sum"),
                        fan_in=("from_account", "nunique"))
                   .reindex(account_ids).fillna(0))
    cb = (hist.groupby("from_account", observed=True)["cross_bank"]
              .mean().reindex(account_ids).fillna(0).rename("cross_bank_frac"))

    # account-level currency-crossing flag (Section 6.1's finding: 0.16% of laundering-
    # linked accounts receive in one currency and pay out in another across their history,
    # invisible to the per-leg currency_mismatch flag)
    recv_cur = (hist.assign(currency_received=lambda d: d["currency_received"].astype(str))
                    .groupby("to_account", observed=True)["currency_received"]
                    .agg(lambda s: frozenset(s.dropna())))
    paid_cur = (hist.assign(currency_paid=lambda d: d["currency_paid"].astype(str))
                    .groupby("from_account", observed=True)["currency_paid"]
                    .agg(lambda s: frozenset(s.dropna())))
    crossing = pd.Series(0, index=account_ids, dtype="int8", name="currency_crossing")
    for a in recv_cur.index.intersection(paid_cur.index).intersection(account_ids):
        if recv_cur[a] and paid_cur[a] and not (recv_cur[a] & paid_cur[a]):
            crossing[a] = 1

    return pd.concat([out_agg, in_agg, cb, crossing], axis=1)


def rolling_origin_splits(node_df, embargo_days=1, min_train_days=5):
    days = sorted(node_df["date_first_seen"].dropna().unique())
    splits = []
    for k in range(min_train_days, len(days) - embargo_days):
        train_end = days[k]
        val_day = days[k + embargo_days]
        train_accts = node_df.index[node_df["date_first_seen"] <= train_end]
        val_accts = node_df.index[node_df["date_first_seen"] == val_day]
        val_accts = val_accts.difference(train_accts)
        splits.append((train_end, val_day, train_accts, val_accts))
    return splits


def run_baseline(tx, nodes, embargo_days=1, min_train_days=5):
    tx = tx.copy()
    tx["cross_bank"] = tx["from_bank"].astype(str) != tx["to_bank"].astype(str)

    splits = rolling_origin_splits(nodes, embargo_days=embargo_days, min_train_days=min_train_days)
    print(f"{len(splits)} rolling folds\n")

    fold_scores, last_model = [], None
    for i, (train_end, val_day, train_idx, val_idx) in enumerate(splits, 1):
        train_idx = train_idx.intersection(nodes.index)
        val_idx = val_idx.intersection(nodes.index)
        if len(val_idx) == 0 or nodes.loc[val_idx, TARGET].sum() == 0:
            print(f"fold {i}: skipped (no positives in validation slice)")
            continue

        train_feat = build_features_asof(tx, train_end, train_idx)
        val_feat = build_features_asof(tx, val_day, val_idx)

        X_train, y_train = train_feat[FEATURES], nodes.loc[train_idx, TARGET]
        X_val, y_val = val_feat[FEATURES], nodes.loc[val_idx, TARGET]

        pos, neg = y_train.sum(), (1 - y_train).sum()
        spw = neg / max(pos, 1)   # see calibration caveat below before using raw predict_proba
        model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    scale_pos_weight=spw, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_scores)
        base_rate = y_val.mean()
        fold_scores.append(pr_auc)
        last_model = model
        print(f"fold {i}: train_end={train_end.date()} val_day={val_day.date()} "
              f"| train n={len(train_idx):,} val n={len(val_idx):,} "
              f"| val base rate={base_rate*100:.3f}% | PR-AUC={pr_auc:.4f} "
              f"(lift={pr_auc/max(base_rate,1e-9):.1f}x over random)")

    if fold_scores:
        print(f"\nmean PR-AUC across {len(fold_scores)} folds: {np.mean(fold_scores):.4f} "
              f"(std {np.std(fold_scores):.4f}, range {min(fold_scores):.4f}-{max(fold_scores):.4f})")
        print("\nfeature importance (last fold):")
        print(pd.Series(last_model.feature_importances_, index=FEATURES)
                .sort_values(ascending=False).to_string())
    else:
        print("no folds produced a scoreable result - check min_train_days / embargo_days "
              "against how many distinct 'first-seen' days your data actually has.")

    return fold_scores


if __name__ == "__main__":
    # Wire this up to your existing notebook state:
    #   from baseline_lightgbm import run_baseline
    #   run_baseline(tx, nodes, embargo_days=1, min_train_days=5)
    print(__doc__)
