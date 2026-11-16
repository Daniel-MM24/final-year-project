"""Stage 4 model ladder: rules -> LightGBM -> GNN gate.

Run from the project root with the learn_env interpreter::

    python src/models/model_ladder.py --rung 0

Each rung exports validation-day scores in the same rolling-origin folds, so the rungs
can be compared on identical days with identical metrics. Scores are written as
``results/ladder/<rung>_scores.parquet`` and a summary as ``<rung>_summary.json``.

Rung 0 is deterministic rules: no fitting, no labels. It exists to give every learned
model a floor, and to prove the harness scores a model with known weaknesses correctly.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_protocol import (  # noqa: E402
    make_splits,
    score_fold,
    summarise_folds,
)
from src.features.build_edge_features import FEATURE_COLS, FMT_COLS  # noqa: E402

FEATURE_PATH = PROJECT_ROOT / "eda_output" / "features" / "edge_features_v1.parquet"
OUT_DIR = PROJECT_ROOT / "results" / "ladder"

# Keys and label only. Account codes are deliberately excluded: they are object dtype
# in the parquet and dominate memory, and no rung needs them.
LOAD_COLS = ["date", "is_laundering"]


# --- Rung 0: deterministic rules ---------------------------------------------------

def rung0_rule_scores(frame: pd.DataFrame) -> np.ndarray:
    """Hand-written risk score in [0, 1]; higher = more suspicious.

    Three signals, each individually defensible from the EDA:
      * amount z-score against the sender's own history (and a large absolute fallback
        for senders with no history, where the z-score is undefined);
      * broad fan-out: a sender paying many distinct counterparties;
      * cross-bank movement.
    Weights are round numbers, not tuned - tuning a rules rung against the labels would
    make it a weak learner and destroy its purpose as a floor.
    """
    hist = frame["from_has_history"].to_numpy() == 1
    sd = frame["from_amount_std"].to_numpy()
    mean = frame["from_amount_mean"].to_numpy()
    amt = np.expm1(frame["log_amount_usd"].to_numpy())

    z = np.zeros(amt.shape[0], dtype=np.float64)
    valid = hist & (sd > 0)
    z[valid] = (amt[valid] - mean[valid]) / sd[valid]
    z = np.clip(z, 0.0, 10.0) / 10.0

    fan = frame["from_fan_out"].to_numpy().astype(np.float64)
    fan_score = np.clip(fan, 0.0, 20.0) / 20.0
    cross = frame["is_cross_bank"].to_numpy().astype(np.float64)

    return 0.6 * z + 0.3 * fan_score + 0.1 * cross


# Rung 1 uses every Stage 3 feature *except* the 1-hop neighbour aggregates, which are
# held back for Rung 2 so the ladder measures what the graph neighbourhood is worth.
HOP_COLS = [c for c in FEATURE_COLS if "_hop_" in c]
RUNG1_COLS = [c for c in FEATURE_COLS if c not in HOP_COLS]
RUNG2_COLS = list(FEATURE_COLS)

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 2,
    "verbose": -1,
    # Bounded memory on the 7 GB dev box: coarser bins and an explicit histogram budget
    # instead of the default, which materialises a float64 binned copy of the window.
    # max_bin=31 keeps the binned dataset at 1 byte/value on the largest fold.
    "max_bin": 63,
    # force_col_wise: LightGBM's row-wise histogram buffer on a 5.1M-row training window
    # pushed peak RSS to 6.4 GB and the kernel killed the run. Column-wise peaks at 2.5 GB.
    "force_col_wise": True,
    # max_delta_step bounds each tree's contribution to the margin. Without it the
    # weighted log-odds run away (|margin| ~ 1e5 on a handful of rows), which destroys
    # both ranking and calibration.
    "max_delta_step": 1.0,
}
NUM_BOOST_ROUND = 300


# Full neg/pos (~2,034) weighting diverges: every leaf applies a large weighted update,
# the gradient does not vanish on near-inseparable classes, and over 300 rounds the
# margin compounds to a median |margin| of ~2.5e6. Sigmoid output hides this by
# pinning every row at 1.0. sqrt(neg/pos) (~45) keeps the imbalance correction while
# bounding the update size; it is the standard choice for weighted boosting.
POS_WEIGHT_POWER = 0.5
# No training-window bound by default. The dataset spans 2022-09-01..09-17, so every
# fold already trains on <= 7 days; the knob stays for callers with a longer history.
TRAIN_WINDOW_DAYS: int | None = None
# A log-odds beyond this is a divergence signal, not a ranking. Weights are only
# interpretable in roughly +-20 nats.
MARGIN_SANITY = 50.0


def _pos_weight(pos: int, neg: int) -> float:
    if pos <= 0:
        return 1.0
    return float((neg / pos) ** POS_WEIGHT_POWER)


def _lightgbm_scorer(cols: list[str], matrix: np.ndarray, feature_cols: list[str]):
    """Fit on each fold's training window and score its validation day.

    Scores are the **raw margin** (the log-odds, ``predict(raw_score=True)``), not the
    sigmoid output. With ``scale_pos_weight = neg/pos`` the weighted log-odds for a
    mixed leaf are ~log(2034) ~ 7.6, so the sigmoid saturates at exactly 1.0 for a large
    block of rows and ties hundreds of thousands of them - recall@k then measures the
    tie order, not the model's ranking. The raw margin keeps the ordering the model
    actually learned at full resolution. It is not a probability; calibration (see
    ``src/models/calibration.py``) is what turns it into one.
    """
    import lightgbm as lgb

    def score_fold_lgb(frame: pd.DataFrame, train_mask: pd.Series,
                       val_mask: pd.Series, model_path: Path | None = None
                       ) -> tuple[np.ndarray, list[float]]:
        # Memory discipline on the 7 GB box. frame[cols].to_numpy() copies 720 MB, and
        # doing it twice (once for train, once for validation) plus LightGBM's raw-data
        # copy peaks around 2.8 GB and the kernel kills it. Build the matrix once,
        # index both sides from it, and free it before the boosting loop.
        tr = train_mask.to_numpy()
        va = val_mask.to_numpy()
        feat, x = feature_cols, matrix
        idx = [feat.index(c) for c in cols]
        val_x = x[np.ix_(np.flatnonzero(va), idx)].copy()
        x_train = x[np.ix_(np.flatnonzero(tr), idx)].copy()
        y = frame["is_laundering"].to_numpy()[tr]
        pos = int(y.sum())
        neg = int(y.size - pos)
        params = dict(LIGHTGBM_PARAMS, scale_pos_weight=_pos_weight(pos, neg))
        train_set = lgb.Dataset(x_train, label=y, free_raw_data=True)
        model = lgb.train(params, train_set, num_boost_round=NUM_BOOST_ROUND)
        del train_set, x_train
        gc.collect()
        if model_path is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path))
        scores = model.predict(val_x, raw_score=True)
        imp = model.feature_importance(importance_type="gain")
        return scores, imp

    return score_fold_lgb


def saturation_report(scores: np.ndarray) -> dict:
    """How many rows share their score with a large tie block.

    A healthy ranking has almost no large ties; the sigmoid-saturated model had 20-93%
    of rows on a single value. This is recorded in every summary so the failure cannot
    reappear silently.
    """
    vals, counts = np.unique(scores, return_counts=True)
    n = scores.shape[0]
    biggest = int(counts.max())
    max_abs = float(np.abs(scores).max())
    return {
        "n": int(n),
        "n_distinct": int(vals.shape[0]),
        "largest_tie": biggest,
        "largest_tie_frac": float(biggest / n),
        "max_abs_score": max_abs,
        "diverged": bool(max_abs > MARGIN_SANITY),
    }


# Rung 0 is label-free and needs no matrix; rungs 1-2 are built after loading, since
# the float32 matrix is passed explicitly (stashing it in frame.attrs made pandas copy
# it on every column access and OOM'd the box).
RUNG_SCORERS = {
    "0": ("rung0_rules", lambda: rung0_rule_scores),
    "1": ("rung1_lgbm", lambda m, c: _lightgbm_scorer(RUNG1_COLS, m, c)),
    "2": ("rung2_lgbm_plus_hop", lambda m, c: _lightgbm_scorer(RUNG2_COLS, m, c)),
}


# --- Loading -----------------------------------------------------------------------


def load_frame(path: Path) -> pd.DataFrame:
    """Read the feature table row-group by row-group into float32 columns.

    A single ``pd.read_parquet`` of ~34 columns decodes every column at once and peaks
    around 4 GB on this 6.9M-row table, which OOM-kills the 7 GB dev box. Accumulating
    per row group bounds the peak to the final float32 frame (~1 GB) plus one 385k-row
    chunk. Account codes are skipped entirely: they are object dtype and no rung uses
    them (scores are keyed by row index).
    """
    import pyarrow.parquet as pq

    feat_cols = list(FEATURE_COLS) + list(FMT_COLS)
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    n_feat = len(feat_cols)
    mat = np.empty((n, n_feat), dtype=np.float32)
    dates = np.empty(n, dtype="datetime64[ns]")
    labels = np.empty(n, dtype=np.int8)

    at = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["date", "is_laundering"] + feat_cols)
        k = t.num_rows
        for j, c in enumerate(feat_cols):
            col = t.column(c).to_numpy(zero_copy_only=False)
            mat[at:at + k, j] = col.astype(np.float32, copy=False)
        dates[at:at + k] = t.column("date").to_numpy()
        labels[at:at + k] = t.column("is_laundering").to_numpy()
        at += k
        del t
        del col
        gc.collect()
    if at != n:
        raise AssertionError(f"row groups covered {at:,} of {n:,} rows")

    frame = pd.DataFrame(mat, columns=feat_cols, copy=False)
    frame["date"] = pd.to_datetime(dates)
    frame["is_laundering"] = labels
    return frame, mat, feat_cols


# --- Shared runner -----------------------------------------------------------------

def window_mask(frame: pd.DataFrame, fold, days: int | None) -> pd.Series:
    """Restrict a fold's training mask to the ``days`` most recent days before the embargo.

    The full-history LightGBM fit peaks at 6.4 GB on this 7.8 GB box (measured, exit 137
    when run alongside the session), so the ladder trains on a bounded recent window.
    The window is still strictly earlier than the validation day - the as-of rule and
    the 1-day embargo are unchanged - it only drops very old days from training.
    """
    if days is None:
        return fold.train_mask
    dates = frame["date"]
    cutoff = fold.train_mask & (dates >= fold.validation_day - pd.Timedelta(days=days + 1))
    if not cutoff.any():
        return fold.train_mask
    return cutoff


def run_rung(name: str, scorer, frame: pd.DataFrame,
             save_models: bool = False, train_days: int | None = TRAIN_WINDOW_DAYS,
             matrix: np.ndarray | None = None,
             feature_cols: list[str] | None = None) -> dict:
    """Score one rung across all folds and write scores + summary."""
    labels = frame["is_laundering"]
    folds = make_splits(frame)

    results = []
    score_parts = []
    importances = []
    saturations = {}
    for fold in folds:
        # Rung 0 is label-free and takes only the validation frame; learned rungs need
        # the training window too. Build the copy only where it is used - a .loc copy of
        # the full column set per fold adds up over 8 folds.
        if getattr(scorer, "__name__", "") == "rung0_rule_scores":
            val = frame.loc[fold.validation_mask]
            scores = scorer(val)
            del val
        else:
            model_path = None
            if save_models:
                model_path = (OUT_DIR / "models"
                              / f"{name}_fold-{fold.validation_day.date()}.txt")
            scores, imp = scorer(frame, window_mask(frame, fold, train_days),
                                 fold.validation_mask, model_path=model_path)
            importances.append(imp)
        # Record ties every fold: a saturated score makes recall@k measure tie order.
        saturations[str(fold.validation_day.date())] = saturation_report(scores)
        results.append(score_fold(fold, scores, full_labels=labels))
        score_parts.append(pd.DataFrame({
            "date": fold.validation_day,
            "fold": fold.name,
            "row": np.flatnonzero(fold.validation_mask.to_numpy()),
            "score": scores,
            "is_laundering": labels[fold.validation_mask].to_numpy(),
        }))

    summary = summarise_folds(results)
    summary["rung"] = name
    summary["saturation"] = saturations
    summary["train_window_days"] = train_days
    if importances:
        cols = RUNG1_COLS if name == "rung1_lgbm" else RUNG2_COLS
        mean_imp = np.mean(np.vstack(importances), axis=0)
        share = mean_imp / mean_imp.sum() if mean_imp.sum() else mean_imp
        summary["feature_importance"] = sorted(
            ({"feature": c, "gain_share": float(s)} for c, s in zip(cols, share)),
            key=lambda d: -d["gain_share"])
        summary["features"] = cols
        summary["lightgbm_params"] = LIGHTGBM_PARAMS
        summary["num_boost_round"] = NUM_BOOST_ROUND
    summary["folds"] = [
        {
            "name": r.name,
            "day": str(r.validation_day.date()),
            "skipped": r.skipped,
            "reason": r.reason,
            "n_validation": r.n_validation,
            "n_positives": r.n_positives,
            "base_rate": r.base_rate,
            "pr_auc": r.pr_auc,
            "lift": r.lift,
            "recall_at_k": r.recall_at_k,
        }
        for r in results
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(score_parts, ignore_index=True).to_parquet(
        OUT_DIR / f"{name}_scores.parquet", index=False)
    (OUT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print_summary(summary: dict) -> None:
    print(f"rung {summary['rung']}: {summary['n_used']} used / "
          f"{summary['n_skipped']} skipped of {summary['n_folds']} folds")
    if summary["skipped"]:
        print("  skipped:", summary["skipped"])
    print(f"  PR-AUC {summary['pr_auc_mean']:.5f}"
          f" +/- {summary['pr_auc_std']:.5f} (std over {summary['n_used']} folds)")
    print(f"  weighted lift {summary['lift_weighted']:.3f}")
    if "feature_importance" in summary:
        print("  top features by gain share:")
        for d in summary["feature_importance"][:10]:
            print(f"    {d['feature']:<24}{d['gain_share']:6.2%}")
    for k, v in sorted(summary["recall_at_k"].items()):
        print(f"  recall@{k}: {v:.4f}")
    if summary["low_history"]:
        print("  low-history folds:", summary["low_history"])
    if summary.get("saturation"):
        worst = max(summary["saturation"].items(),
                    key=lambda kv: kv[1]["largest_tie_frac"])
        print(f"  worst score tie: {worst[1]['largest_tie']:,} rows "
              f"({worst[1]['largest_tie_frac']:.1%}) on {worst[0]}")
        div = [d for d, s in summary["saturation"].items() if s["diverged"]]
        if div:
            top = max(summary["saturation"].items(),
                      key=lambda kv: kv[1]["max_abs_score"])
            print(f"  !! DIVERGED on {len(div)} fold(s); max |score| "
                  f"{top[1]['max_abs_score']:,.0f} on {top[0]} — scores are not a ranking")
    print()
    print(f"  {'fold':<16}{'n':>9}{'pos':>6}{'base':>10}{'PR-AUC':>10}{'lift':>8}")
    for f in summary["folds"]:
        if f["skipped"]:
            print(f"  {f['name']:<16}{f['n_validation']:>9,}{f['n_positives']:>6}"
                  f"   skipped: {f['reason']}")
        else:
            print(f"  {f['name']:<16}{f['n_validation']:>9,}{f['n_positives']:>6}"
                  f"{f['base_rate']:>10.5f}{f['pr_auc']:>10.5f}{f['lift']:>8.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", default="0", choices=sorted(RUNG_SCORERS),
                        help="which rung to run")
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--save-models", action="store_true",
                        help="write each fold's LightGBM booster to results/ladder/models/")
    args = parser.parse_args(argv)

    if not args.features.exists():
        print(f"missing {args.features}; run src/features/build_edge_features.py")
        return 2

    name, make_scorer = RUNG_SCORERS[args.rung]
    frame, mat, feat_cols = load_frame(args.features)
    scorer = make_scorer() if args.rung == "0" else make_scorer(mat, feat_cols)
    print(f"loaded {len(frame):,} edges, {int(frame['is_laundering'].sum()):,} positives")

    summary = run_rung(name, scorer, frame, save_models=args.save_models,
                       matrix=mat, feature_cols=feat_cols)
    _print_summary(summary)
    print(f"wrote {OUT_DIR / (name + '_scores.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
