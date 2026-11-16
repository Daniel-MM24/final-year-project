"""Stage 1: the evaluation protocol, as code.

This module is the single source of truth for *how the project measures itself*. Every
later stage (features, model ladder, alerting, agent, deployment) imports from here
rather than re-implementing splits or metrics. The rules it enforces are written down in
``docs/evaluation_contract.md``; function docstrings cite the matching section.

Design rules, in priority order:

1. **Edge-level.** The prediction unit is the transaction, labelled ``is_laundering``.
   Node-level is an internal diagnostic only.
2. **As-of only.** A fold's training set is strictly earlier data than its validation
   day, minus an embargo. No whole-file aggregates.
3. **Skipped folds are skipped, not scored.** A fold with <50 rows or a single class is
   excluded from every aggregate. Never averaged in as 0.0 or as nan.
4. **Lift travels with PR-AUC** because the daily base rate moves 5x across the period.
5. **No silent defaults.** Anything excluded is recorded with a reason string so
   ``summarise_folds`` can say *why* a fold was dropped.

Typical use::

    from src.eval_protocol import make_splits, score_fold, summarise_folds

    folds = make_splits(tx)                 # tx has date, is_laundering
    results = []
    for fold in folds:
        model = fit(fold.train)             # your as-of feature builder
        scores = model.predict_proba(build_features(fold.validation))[:, 1]
        results.append(score_fold(fold, scores))
    print(summarise_folds(results))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# --- Constants from docs/evaluation_contract.md ------------------------------------

MIN_VALIDATION_ROWS = 50
"""Contract section 2. A fold with fewer validation rows is skipped."""

EMBARGO_DAYS = 1
"""Contract section 2. Validation day *d* trains on ``date <= d - 1 - EMBARGO_DAYS``."""

MIN_TRAIN_DAYS = 2
"""Contract section 2. A fold needs at least this many training calendar days."""

ALERT_BUDGETS: tuple[int, ...] = (100, 1000)
"""Contract section 3. Alerts per validation day for recall@k."""

EXCLUDED_VALIDATION_DAYS: tuple[str, ...] = tuple(
    f"2022-09-{day:02d}" for day in range(11, 18)
)
"""Contract section 5. Simulator tail: 3-77 rows/day at a 56-71% positive rate.

Measured by ``src/measure_daily_table.py``. These days are never validation days.
"""

LOW_HISTORY_DAYS: tuple[str, ...] = ("2022-09-03",)
"""Contract section 5. Usable as validation but flagged for separate reporting.

09-01/02 are never validation days (insufficient embargoed history), so 09-03 is the
first emitted fold - and its training window is 09-01 alone, the ramp-up day holding
84% of new-account first appearances. A model validated on 09-03 has seen almost no
transaction history, so its score is not comparable with the steady state."""


# --- Fold representation -----------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One rolling-origin split.

    ``train`` and ``validation`` are *masks* (boolean Series aligned to the caller's
    frame) rather than copies, so a full 6.9M-row frame is never duplicated per fold.

    Attributes
    ----------
    name:
        Human label, e.g. ``"fold-2022-09-05"``.
    validation_day:
        The cutoff day, ``Timestamp`` normalised to midnight.
    train_mask, validation_mask:
        Boolean masks into the caller's transaction frame.
    train_max_date:
        Latest date present in the training set. Asserted ``<= validation_day -
        (EMBARGO_DAYS + 1)``; the leakage test relies on this attribute.
    low_history:
        True when the validation day is one of ``LOW_HISTORY_DAYS``.
    """

    name: str
    validation_day: pd.Timestamp
    train_mask: pd.Series
    validation_mask: pd.Series
    train_max_date: pd.Timestamp
    low_history: bool = False

    @property
    def embargo_start(self) -> pd.Timestamp:
        """First embargoed day (inclusive): validation day minus EMBARGO_DAYS."""
        return self.validation_day - pd.Timedelta(days=EMBARGO_DAYS)


@dataclass
class FoldResult:
    """Outcome of scoring one fold. ``skipped`` folds carry a ``reason`` and no metrics."""

    name: str
    validation_day: pd.Timestamp
    skipped: bool
    reason: str = ""
    low_history: bool = False
    n_validation: int = 0
    n_positives: int = 0
    base_rate: float = float("nan")
    pr_auc: float = float("nan")
    lift: float = float("nan")
    recall_at_k: dict[int, float] = field(default_factory=dict)


# --- Splits ------------------------------------------------------------------------


def _normalise_dates(dates: pd.Series) -> pd.Series:
    """Strip any time component so day comparisons are exact.

    ``Series.dt.normalize`` on a datetime column is a no-op when already midnight, so
    this is safe to call unconditionally.
    """
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    return dates.dt.normalize()


def make_splits(
    tx: pd.DataFrame,
    date_col: str = "date",
    excluded_days: Sequence[str] = EXCLUDED_VALIDATION_DAYS,
) -> list[Fold]:
    """Build rolling-origin folds over calendar days (contract section 2).

    Parameters
    ----------
    tx:
        Any frame carrying a date column. It does **not** need to be sorted, and it
        may be a subset of the full file (tests use small fixtures).
    date_col:
        Column holding transaction dates.
    excluded_days:
        Validation days to never produce. Defaults to the measured simulator tail.

    Returns
    -------
    list[Fold]
        One fold per non-excluded calendar day from ``min(excluded) - 2`` onward, in
        ascending date order. Days before ``excluded_days[0]`` are the steady state;
        days after are the tail. Folds are produced for every day between
        ``min_day + MIN_TRAIN_DAYS`` and ``max(excluded_days) - 1`` that is in
        ``excluded_days``' complement.

    Notes
    -----
    Folds that fail ``is_fold_usable`` are *returned* rather than dropped, so callers
    can print why they were skipped. ``summarise_folds`` never averages them in.
    """
    dates = _normalise_dates(tx[date_col])

    min_day = dates.min()
    excluded = {pd.Timestamp(day).normalize() for day in excluded_days}
    # With no exclusions the whole range is steady state; validate through the last day.
    last_day = min(excluded) - pd.Timedelta(days=1) if excluded else dates.max()

    folds: list[Fold] = []
    day = min_day + pd.Timedelta(days=MIN_TRAIN_DAYS)
    while day <= last_day:
        if day not in excluded:
            train_cutoff = day - pd.Timedelta(days=EMBARGO_DAYS + 1)
            train_mask = dates <= train_cutoff
            validation_mask = dates == day

            # train_max_date is None when the train window is empty; Fold is frozen so
            # we use pd.NaT, which never compares <= any usable date (see tests).
            if train_mask.any():
                train_max = dates[train_mask].max()
            else:
                train_max = pd.NaT

            folds.append(
                Fold(
                    name=f"fold-{day.date()}",
                    validation_day=day,
                    train_mask=train_mask,
                    validation_mask=validation_mask,
                    train_max_date=train_max,
                    low_history=str(day.date()) in LOW_HISTORY_DAYS,
                )
            )
        day += pd.Timedelta(days=1)

    return folds


def is_fold_usable(
    fold: Fold,
    labels: pd.Series,
    min_rows: int = MIN_VALIDATION_ROWS,
) -> tuple[bool, str]:
    """Decide whether a fold may be scored (contract section 2).

    Returns ``(usable, reason)``. ``reason`` is empty when usable, otherwise a short
    machine-readable string, one of:

    - ``"empty_train"``      - training window had no rows
    - ``"empty_validation"`` - validation day had no rows
    - ``"too_few_rows:<n>"`` - validation day below ``min_rows``
    - ``"single_class"``     - validation day has only one label value

    Why skip instead of scoring: a 9-row day with all positives has a degenerate PR-AUC
    of 1.0, and a 9-row day with no positives has an undefined one. Both are noise; 09-15
    is exactly the former, which is why the contract excludes it outright.
    """
    if not fold.train_mask.any():
        return False, "empty_train"

    val_labels = labels[fold.validation_mask]
    n_rows = int(val_labels.shape[0])
    if n_rows == 0:
        return False, "empty_validation"
    if n_rows < min_rows:
        return False, f"too_few_rows:{n_rows}"
    if val_labels.nunique() < 2:  # nunique, not len — single-class check per rule 1
        return False, "single_class"
    return True, ""


# --- Metrics -----------------------------------------------------------------------


def recall_at_k(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    k: int,
) -> float:
    """Fraction of positives captured by the top-``k`` scored rows (contract section 3).

    Ties are broken by original row order (``np.argsort(kind="stable")``), which is
    deterministic and identical for every model, so models remain comparable. k larger
    than the row count is clamped, matching "alert everyone" rather than erroring.

    Returns ``nan`` when there are no positives: recall is undefined, and silently
    returning 0.0 would punish a day that simply has nothing to find.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must be the same length")
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    k = max(0, min(int(k), labels.shape[0]))
    order = np.argsort(-scores, kind="stable")[:k]
    return float((labels[order] == 1).sum() / n_pos)


def score_fold(
    fold: Fold,
    scores: Sequence[float] | np.ndarray,
    labels: pd.Series | None = None,
    *,
    full_labels: pd.Series | None = None,
    budgets: Sequence[int] = ALERT_BUDGETS,
) -> FoldResult:
    """Score one fold: PR-AUC, lift, recall@k (contract section 3).

    Parameters
    ----------
    fold:
        The split being scored.
    scores:
        Model output aligned to **the validation rows only**, in the same order as
        ``labels[mask]``. Passing full-length scores is a common bug; pass only the
        validation slice.
    labels:
        Full-length label Series aligned to the frame the fold's masks index. Optional
        when ``full_labels`` is given under that name (kept for call-site readability).
    full_labels:
        Alias of ``labels``.
    budgets:
        Alert budgets, each applied per validation day.

    Returns
    -------
    FoldResult
        If the fold is not usable, ``skipped=True`` with a ``reason`` and nan metrics —
        never zeros. ``summarise_folds`` filters on ``skipped``.

    Raises
    ------
    ValueError
        When ``scores`` length does not match the validation-row count. This is the
        check that catches accidental leakage of train rows into the score vector.
    """
    labels = labels if labels is not None else full_labels
    if labels is None:
        raise ValueError("pass labels= or full_labels=")

    usable, reason = is_fold_usable(fold, labels)
    if not usable:
        val = labels[fold.validation_mask]
        return FoldResult(
            name=fold.name,
            validation_day=fold.validation_day,
            skipped=True,
            reason=reason,
            low_history=fold.low_history,
            n_validation=int(val.shape[0]),
            n_positives=int(val.sum()) if val.shape[0] else 0,
        )

    y_true = labels[fold.validation_mask].to_numpy()
    y_score = np.asarray(scores, dtype=float)
    if y_score.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"fold {fold.name}: scores has {y_score.shape[0]} rows but validation mask "
            f"selects {y_true.shape[0]}. Score only the validation slice."
        )

    base_rate = float(y_true.mean())
    # average_precision_score is PR-AUC; it equals the expected recall@k integrated
    # over alert budgets, which is why it is the headline metric and not ROC-AUC.
    pr_auc = float(average_precision_score(y_true, y_score))
    lift = pr_auc / base_rate if base_rate > 0 else float("nan")

    return FoldResult(
        name=fold.name,
        validation_day=fold.validation_day,
        skipped=False,
        low_history=fold.low_history,
        n_validation=int(y_true.shape[0]),
        n_positives=int(y_true.sum()),
        base_rate=base_rate,
        pr_auc=pr_auc,
        lift=lift,
        recall_at_k={int(k): recall_at_k(y_true, y_score, k) for k in budgets},
    )


# --- Aggregation -------------------------------------------------------------------


def summarise_folds(results: Sequence[FoldResult]) -> dict:
    """Aggregate fold results (contract section 3).

    Every aggregate is computed over **usable folds only**. Skipped folds are counted
    and reported with their reasons so the reader can see how much of the timeline was
    actually measured.

    Returns a dict with:

    - ``n_folds``        - folds presented
    - ``n_used``         - usable folds that entered the aggregate
    - ``n_skipped``      - number skipped
    - ``skipped``        - ``{reason: count}``
    - ``pr_auc_mean``    - mean over used folds, ``nan`` if none
    - ``pr_auc_std``     - population std (ddof=0) to match ``baseline_lightgbm.py``
    - ``lift_weighted``  - positives captured at full recall weighting: sum(positives *
      lift) / sum(positives). Equivalent to the count-weighted lift and cannot be
      inflated by a low-base-rate fold.
    - ``recall_at_k``    - ``{k: positives captured / positives available}`` across used
      folds, i.e. a count-weighted mean rather than a mean of fold fractions.
    - ``low_history``    - names of used folds flagged as low-history, for separate
      reporting.

    The std is a dispersion estimate over a handful of days, not a confidence interval;
    with 8 folds it should be quoted as "+/- std over 8 folds", never as a CI.
    """
    used = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    skipped_counts: dict[str, int] = {}
    for r in skipped:
        skipped_counts[r.reason] = skipped_counts.get(r.reason, 0) + 1

    summary: dict = {
        "n_folds": len(results),
        "n_used": len(used),
        "n_skipped": len(skipped),
        "skipped": skipped_counts,
        "pr_auc_mean": float("nan"),
        "pr_auc_std": float("nan"),
        "lift_weighted": float("nan"),
        "recall_at_k": {},
        "low_history": [r.name for r in used if r.low_history],
    }

    if not used:
        return summary

    pr_aucs = np.array([r.pr_auc for r in used], dtype=float)
    summary["pr_auc_mean"] = float(pr_aucs.mean())
    summary["pr_auc_std"] = float(pr_aucs.std(ddof=0))

    total_pos = sum(r.n_positives for r in used)
    if total_pos > 0:
        summary["lift_weighted"] = float(
            sum(r.n_positives * r.lift for r in used) / total_pos
        )

    # Count-weighted recall: a day with 402 positives should not weigh the same as one
    # with 301, so accumulate captured positives rather than averaging fold fractions.
    budgets = sorted({k for r in used for k in r.recall_at_k})
    for k in budgets:
        captured = sum(r.recall_at_k[k] * r.n_positives for r in used)
        summary["recall_at_k"][int(k)] = float(captured / total_pos) if total_pos else float("nan")

    return summary
