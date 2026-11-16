# Evaluation Contract

Binding for every later stage. If a result is reported without conforming to this
document, it is not a project result. Measurements below come from
`src/measure_daily_table.py` on `data/LI-Small_Trans.csv` (6,924,049 rows,
2022-09-01 .. 2022-09-17). Data is synthetic; no result here is real-world evidence.

## 1. Prediction unit

**The transaction (edge), labelled `Is Laundering`.** Edge base rate 0.0515%.

The node label used by `baseline_lightgbm.py` (`is_laundering_node`, 0.7442%) is
**rejected as a headline metric**, for three reasons:

1. It is a *contaminated* label. An account inherits the label if it is on either end
   of any laundering edge, so clean counterparties of a single laundering payment are
   marked positive. The 14.4x rate gap over the edge base rate is that smearing.
2. It over-weights hub accounts. One laundering payment to a busy exchange marks the
   exchange positive; the edge view counts it once.
3. It is not the decision the system makes. Alerts are raised on transactions.

Node-level scoring is permitted **only** as an internal baseline (to judge whether the
graph adds signal over degree/fan-out) and must be labelled as such wherever reported.

## 2. Split protocol

Rolling origin over **calendar days**, with a one-day embargo.

For a validation day *d* (cutoff = 00:00 of *d*):

| Element | Rule |
|---|---|
| Train window | all transactions with `date <= d - 2 days` |
| Embargo day | `d - 1`. Excluded from both train and validation. Absorbs label/feature lag. |
| Validation day | transactions with `date == d` |
| Minimum validation size | 50 rows **and** >= 1 positive. Below this the fold is skipped. |
| Single-class folds | skipped, never scored, never averaged in. |
| Unusable days | measured in section 5, excluded from the validation-day list entirely. |
| Minimum train history | a fold needs >= 2 calendar days of training data; folds before that are skipped. |

Implemented by `make_splits()`; guarded by `is_fold_usable()`. The split is defined on
**rows and dates, not on account first-seen** — the baseline's account-level split is
superseded.

## 3. Metrics

Per fold:

- **PR-AUC** (`average_precision_score` on the edge label).
- **Lift** = PR-AUC / fold base rate. Required companion to PR-AUC because the base
  rate swings 0.020% -> 0.106% across validation days; raw PR-AUC is not comparable
  across folds.
- **Recall@k** for k in {100, 1000} alerts per validation day: fraction of that day's
  positives caught by the top-k scored transactions. Reported as a count-weighted mean
  across folds (a day with 402 positives should not weigh the same as one with 301).

Across folds, `summarise_folds()` reports:

- count of usable folds and count skipped, with reasons;
- **mean and std of PR-AUC over usable folds only** (population std, matching the
  baseline's `np.std`);
- **count-weighted mean lift**, computed as total positives captured / total expected
  under a random ranker, so low-base-rate folds cannot inflate it;
- a warning that the std with 4-8 folds is a rough dispersion estimate, not a CI.

Test-period (2022-09-01/02) folds are used for calibration only, never for headline
metrics (their base rate is 3-5x lower than the steady state).

## 4. Calibration policy

Scores from `scale_pos_weight`-trained models are **not probabilities** and must not be
thresholded directly.

1. Any probability used for alert-volume tuning is produced by **isotonic regression**
   (fallback: Platt scaling when the calibration fold has < 200 positives), fitted on
   the **immediately preceding usable validation fold** — earlier data only.
2. Calibrator inputs are the *raw* model scores on that earlier fold; the calibrator is
   then frozen and applied to the current fold.
3. The calibration fold itself is never used to fit or evaluate the same model.
4. Recall@k is computed on **ranks**, so it is calibration-independent. PR-AUC is also
   rank-based. Only any probability threshold (e.g. "alert if p > 0.5") requires the
   calibrated score.

## 5. Unusable days (measured, not assumed)

From the daily table:

- **2022-09-11 .. 2022-09-17 excluded.** 3-77 rows/day at a 56-71% positive rate —
  simulator tail dump, unrepresentative and too small to validate on. Folds whose
  validation day falls here are skipped. (09-11 itself has 77 rows so it would pass the
  50-row floor; it is excluded anyway because its base rate is ~1200x the steady state.)
- **2022-09-01 .. 2022-09-02 usable as validation but flagged.** They carry 84% of all
  new-account first appearances (599,154 on 09-01 alone). Any fold here has a train
  window with almost no behavioural history.
- **Effective validation days: 2022-09-02 .. 2022-09-10** — eight folds.

## 6. Known data caveats carried forward to Stage 2

- 6,924,049 rows but 6,787,653 distinct rows: ~136K exact duplicates, far more than the
  8 reported in `eda_output/eda_summary.json`. Deduplication rule must be fixed before
  features are built.
- 1,257,457 distinct account values in from/to slots vs 712,684 "active accounts" in
  the earlier EDA. Reconcile.
- Account IDs are hex strings with a shared `10042B...` prefix family; excluded from
  model input per rule 6.
- 804,477 self-loops (`from_account == to_account`) exist. Whether they count toward
  fan-out/velocity is decided in Stage 3, not here.
