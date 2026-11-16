"""Unit tests for the Stage 1 evaluation protocol.

Each test maps to one clause of ``docs/evaluation_contract.md``. The important ones are
not the happy paths but the guards:

- ``test_no_validation_day_in_train``      contract s2: no leakage across the embargo
- ``test_embargo_day_in_neither_split``    contract s2: the embargo really is empty
- ``test_single_class_fold_never_averaged`` contract s2/s3: skipped folds cannot move
                                            the headline mean
- ``test_too_few_rows_fold_skipped``       contract s2: the <50-row floor

Run with::

    /home/dan/miniconda3/envs/learn_env/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval_protocol import (
    EMBARGO_DAYS,
    MIN_VALIDATION_ROWS,
    Fold,
    FoldResult,
    is_fold_usable,
    make_splits,
    recall_at_k,
    score_fold,
    summarise_folds,
)


# --- Fixtures ----------------------------------------------------------------------


def make_frame(start: str = "2022-09-01", days: int = 10, rows_per_day: int = 60) -> pd.DataFrame:
    """A synthetic transaction frame with a date and a label column.

    Positives are placed deterministically every 10th row so each day has a stable
    number of positives; ``rows_per_day`` above the 50-row floor by default.
    """
    dates = pd.date_range(start, periods=days, freq="D").repeat(rows_per_day)
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "is_laundering": (np.arange(n) % 10 == 0).astype(int),
            "row_id": np.arange(n),
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    """Ten days, 60 rows each, 6 positives per day. Excludes the simulator tail days."""
    return make_frame()


# --- Split protocol (contract s2) --------------------------------------------------


def test_one_fold_per_usable_day(frame):
    """One fold for each day that is neither excluded nor lacking train history."""
    folds = make_splits(frame)
    days = [f.validation_day.date().isoformat() for f in folds]
    # 09-01/02 lack >= MIN_TRAIN_DAYS of embargoed history; 09-10 is the last steady day.
    assert days == [f"2022-09-{d:02d}" for d in range(3, 11)]


def test_no_validation_day_in_train(frame):
    """Contract s2: every training row is strictly before the validation day.

    This is the leakage test the note demands. It is an assertion on the data, not on
    the mask construction, so a future change to ``make_splits`` that lets a validation
    row into training fails here.
    """
    folds = make_splits(frame)
    for fold in folds:
        train_dates = frame.loc[fold.train_mask, "date"]
        if train_dates.empty:
            continue
        assert train_dates.max() < fold.validation_day
        assert (train_dates < fold.validation_day).all()


def test_embargo_day_in_neither_split(frame):
    """Contract s2: the day before the validation day is in neither train nor validation."""
    folds = make_splits(frame)
    for fold in folds:
        embargo_day = fold.validation_day - pd.Timedelta(days=EMBARGO_DAYS)
        assert not frame.loc[fold.train_mask, "date"].eq(embargo_day).any()
        assert not frame.loc[fold.validation_mask, "date"].eq(embargo_day).any()


def test_train_max_date_respects_embargo(frame):
    """The recorded ``train_max_date`` must be at least embargo+1 days behind."""
    for fold in make_splits(frame):
        assert fold.train_max_date <= fold.validation_day - pd.Timedelta(
            days=EMBARGO_DAYS + 1
        )


def test_excluded_tail_days_not_produced():
    """Contract s5: 2022-09-11..17 are never validation days, even with the data present."""
    tx = make_frame(start="2022-09-01", days=16, rows_per_day=60)
    days = {f.validation_day.date().isoformat() for f in make_splits(tx)}
    assert not any(d >= "2022-09-11" for d in days)


def test_tail_day_exclusion_holds_even_at_usable_size():
    """A tail day with plenty of rows is still excluded - size is not the criterion."""
    tx = make_frame(start="2022-09-01", days=17, rows_per_day=500)
    days = {f.validation_day.date().isoformat() for f in make_splits(tx)}
    assert "2022-09-15" not in days


def test_splits_ascending(frame):
    days = [f.validation_day for f in make_splits(frame)]
    assert days == sorted(days)


def test_low_history_flag(frame):
    """Contract s5: the first steady-state fold trains on a single day and is flagged."""
    folds = make_splits(frame)
    by_day = {f.validation_day.date().isoformat(): f for f in folds}
    # Folds start at 09-03 because 09-01/02 lack enough embargoed history; 09-03's train
    # window is 09-01 alone, which is thin, so it is reported separately.
    assert by_day["2022-09-03"].low_history is True
    assert by_day["2022-09-04"].low_history is False


# --- Usability (contract s2) -------------------------------------------------------


def _fold_from(frame: pd.DataFrame, day: str) -> Fold:
    day_ts = pd.Timestamp(day)
    return Fold(
        name=day,
        validation_day=day_ts,
        train_mask=frame["date"] < day_ts,
        validation_mask=frame["date"] == day_ts,
        train_max_date=frame.loc[frame["date"] < day_ts, "date"].max(),
    )


def test_single_class_fold_rejected(frame):
    """A validation day with one label value is not usable."""
    tx = frame.copy()
    day = pd.Timestamp("2022-09-05")
    tx.loc[tx["date"] == day, "is_laundering"] = 1
    fold = _fold_from(tx, "2022-09-05")
    usable, reason = is_fold_usable(fold, tx["is_laundering"])
    assert (usable, reason) == (False, "single_class")


def test_too_few_rows_fold_skipped():
    """Below MIN_VALIDATION_ROWS rows the fold is skipped and never scored."""
    tx = make_frame(days=6, rows_per_day=60)
    day = pd.Timestamp("2022-09-05")
    keep = (tx["date"] < day) | (tx["date"] > day)
    mostly_gone = tx[keep].copy()
    small = tx[tx["date"] == day].iloc[: MIN_VALIDATION_ROWS - 1].copy()
    tx_small = pd.concat([mostly_gone, small], ignore_index=True)
    fold = _fold_from(tx_small, "2022-09-05")
    usable, reason = is_fold_usable(fold, tx_small["is_laundering"])
    assert not usable
    assert reason == f"too_few_rows:{MIN_VALIDATION_ROWS - 1}"


def test_empty_train_fold_rejected():
    tx = make_frame(days=6)
    day = tx["date"].min()
    fold = Fold(
        name="first",
        validation_day=day,
        train_mask=pd.Series(False, index=tx.index),
        validation_mask=tx["date"] == day,
        train_max_date=pd.NaT,
    )
    usable, reason = is_fold_usable(fold, tx["is_laundering"])
    assert (usable, reason) == (False, "empty_train")


# --- Metrics (contract s3) ---------------------------------------------------------


def test_recall_at_k_perfect_order():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.05, 0.01])
    assert recall_at_k(labels, scores, 2) == 1.0
    assert recall_at_k(labels, scores, 1) == 0.5


def test_recall_at_k_k_clamped_to_rows():
    labels = np.array([1, 0, 0])
    scores = np.array([0.3, 0.2, 0.1])
    assert recall_at_k(labels, scores, 1000) == 1.0


def test_recall_at_k_no_positives_is_nan():
    assert np.isnan(recall_at_k(np.zeros(5), np.arange(5.0), 2))


def test_recall_at_k_ties_are_deterministic():
    """Identical scores must rank by row order, so recall is reproducible."""
    labels = np.array([0, 1, 0, 1])
    scores = np.zeros(4)
    # Stable argsort keeps rows 0,1 -> captures the row-1 positive only.
    assert recall_at_k(labels, scores, 2) == 0.5


def test_score_fold_metrics(frame):
    folds = {f.validation_day.date().isoformat(): f for f in make_splits(frame)}
    fold = folds["2022-09-05"]
    y = frame.loc[fold.validation_mask, "is_laundering"].to_numpy()
    # A model that ranks all positives first gets PR-AUC 1.0 and recall@k = 1.0.
    scores = np.where(y == 1, 1.0, 0.0) + 1e-9 * np.arange(len(y))
    res = score_fold(fold, scores, labels=frame["is_laundering"])
    assert not res.skipped
    assert res.pr_auc == pytest.approx(1.0)
    assert res.n_positives == int(y.sum())
    assert res.recall_at_k[100] == 1.0


def test_score_fold_rejects_wrong_length(frame):
    """Scoring with full-length scores instead of the validation slice must fail loudly."""
    fold = make_splits(frame)[0]
    with pytest.raises(ValueError, match="validation mask"):
        score_fold(fold, np.zeros(len(frame)), labels=frame["is_laundering"])


def test_skipped_fold_has_nan_not_zero(frame):
    fold = make_splits(frame)[0]
    tx = frame.copy()
    tx.loc[fold.validation_mask, "is_laundering"] = 0
    res = score_fold(fold, np.zeros(int(fold.validation_mask.sum())), labels=tx["is_laundering"])
    assert res.skipped
    assert res.reason == "single_class"
    assert np.isnan(res.pr_auc)


# --- Aggregation (contract s3) -----------------------------------------------------


def _result(name: str, pr_auc: float, n_pos: int, skipped: bool = False, reason: str = "") -> FoldResult:
    return FoldResult(
        name=name,
        validation_day=pd.Timestamp("2022-09-05"),
        skipped=skipped,
        reason=reason,
        n_validation=100,
        n_positives=n_pos,
        base_rate=n_pos / 100,
        pr_auc=pr_auc,
        lift=pr_auc / (n_pos / 100) if n_pos else float("nan"),
        recall_at_k={100: pr_auc, 1000: 1.0},
    )


def test_single_class_fold_never_averaged():
    """The headline mean must be identical with or without a skipped fold present.

    This is the regression test for "never average skipped folds in". If someone
    changes ``summarise_folds`` to iterate all results, the two means diverge and this
    fails.
    """
    used = [_result("a", 0.2, 10), _result("b", 0.4, 10)]
    with_skipped = used + [_result("c", float("nan"), 30, skipped=True, reason="single_class")]

    s_clean = summarise_folds(used)
    s_with = summarise_folds(with_skipped)

    assert s_with["pr_auc_mean"] == pytest.approx(s_clean["pr_auc_mean"])
    assert s_with["pr_auc_mean"] == pytest.approx(0.3)
    assert s_with["n_used"] == 2
    assert s_with["n_skipped"] == 1
    assert s_with["skipped"] == {"single_class": 1}


def test_pr_auc_std_is_population():
    used = [_result("a", 0.2, 10), _result("b", 0.4, 10)]
    s = summarise_folds(used)
    assert s["pr_auc_std"] == pytest.approx(0.1)


def test_lift_weighted_by_positives():
    """A low-base-rate fold cannot inflate the aggregate lift."""
    # a: pr_auc .2, base .1 -> lift 2.0, 10 positives
    # b: pr_auc .05, base .01 -> lift 5.0, 1 positive
    used = [_result("a", 0.2, 10), _result("b", 0.05, 1)]
    s = summarise_folds(used)
    expected = (10 * 2.0 + 1 * 5.0) / 11
    assert s["lift_weighted"] == pytest.approx(expected)
    assert s["lift_weighted"] < 5.0  # naive mean of lifts would be 3.5


def test_recall_weighted_by_positives():
    used = [_result("a", 0.2, 10), _result("b", 0.4, 30)]
    s = summarise_folds(used)
    assert s["recall_at_k"][100] == pytest.approx((10 * 0.2 + 30 * 0.4) / 40)


def test_summary_with_no_usable_folds():
    s = summarise_folds([_result("x", float("nan"), 0, skipped=True, reason="empty_validation")])
    assert s["n_used"] == 0
    assert np.isnan(s["pr_auc_mean"])
    assert s["pr_auc_std"] != s["pr_auc_std"]  # nan
    assert s["skipped"] == {"empty_validation": 1}


def test_low_history_listed(frame):
    folds = make_splits(frame)
    results = []
    for f in folds:
        y = frame.loc[f.validation_mask, "is_laundering"].to_numpy()
        results.append(score_fold(f, np.where(y == 1, 1.0, 0.0), labels=frame["is_laundering"]))
    s = summarise_folds(results)
    # Only the first fold (09-03, single-day train window) is reported separately.
    assert s["low_history"] == ["fold-2022-09-03"]
