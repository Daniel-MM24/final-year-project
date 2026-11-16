# Stage 4 - Calibration

Rung 1 is trained with `scale_pos_weight = neg/pos`, so its raw output ranks
well but its *level* is meaningless: the mean score is ~0.5 on every day while
the true base rate is ~0.1%. Any decision that needs a probability (expected
loss, alert thresholds in cost terms) must be calibrated first.

Every pair below fits the calibrator on an **earlier** validation day and applies
it to a **later** one - the same as-of discipline the folds use. No pair is
cherry-picked; all fit->hold pairs are in the table.

## Headline

| Metric | Brier |
|---|---|
| Raw score as probability (before) | 44.84675 |
| Platt, fitted on an earlier day      | **0.00059** |
| Isotonic, fitted on an earlier day   | 0.00059 |

Pooled over 28 fit->hold pairs, 16,822,063 scored rows. The raw-score Brier is dominated by the level mismatch, not by ranking quality - it is the number you get if you ship `predict_proba` uncalibrated. Calibration cuts it by ~430x. Platt and isotonic are indistinguishable in aggregate; Platt is lower in 19 of 28 pairs and never clearly higher, which is what 2 parameters vs a step function with ~300 positives per fit day would predict, but the difference is not large enough here to call isotonic overfit.

## Per-pair Brier

| Fit day | Hold day | n hold | pos | base | before | Platt | Isotonic |
|---|---|---|---|---|---|---|---|
| 2022-09-03 | 2022-09-04 | 282,476 | 291 | 0.00103 | 70.33779 | 0.00103 | 0.00103 |
| 2022-09-03 | 2022-09-05 | 657,397 | 370 | 0.00056 | 39.12732 | 0.00056 | 0.00056 |
| 2022-09-03 | 2022-09-06 | 657,170 | 380 | 0.00058 | 43.10636 | 0.00057 | 0.00058 |
| 2022-09-03 | 2022-09-07 | 658,504 | 368 | 0.00056 | 52.47937 | 0.00055 | 0.00056 |
| 2022-09-03 | 2022-09-08 | 657,938 | 402 | 0.00061 | 49.20687 | 0.00060 | 0.00061 |
| 2022-09-03 | 2022-09-09 | 891,573 | 367 | 0.00041 | 38.39394 | 0.00041 | 0.00041 |
| 2022-09-03 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00105 | 0.00106 |
| 2022-09-04 | 2022-09-05 | 657,397 | 370 | 0.00056 | 39.12732 | 0.00056 | 0.00056 |
| 2022-09-04 | 2022-09-06 | 657,170 | 380 | 0.00058 | 43.10636 | 0.00057 | 0.00057 |
| 2022-09-04 | 2022-09-07 | 658,504 | 368 | 0.00056 | 52.47937 | 0.00055 | 0.00055 |
| 2022-09-04 | 2022-09-08 | 657,938 | 402 | 0.00061 | 49.20687 | 0.00060 | 0.00060 |
| 2022-09-04 | 2022-09-09 | 891,573 | 367 | 0.00041 | 38.39394 | 0.00042 | 0.00042 |
| 2022-09-04 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00105 | 0.00105 |
| 2022-09-05 | 2022-09-06 | 657,170 | 380 | 0.00058 | 43.10636 | 0.00057 | 0.00057 |
| 2022-09-05 | 2022-09-07 | 658,504 | 368 | 0.00056 | 52.47937 | 0.00055 | 0.00055 |
| 2022-09-05 | 2022-09-08 | 657,938 | 402 | 0.00061 | 49.20687 | 0.00060 | 0.00060 |
| 2022-09-05 | 2022-09-09 | 891,573 | 367 | 0.00041 | 38.39394 | 0.00041 | 0.00041 |
| 2022-09-05 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00105 | 0.00105 |
| 2022-09-06 | 2022-09-07 | 658,504 | 368 | 0.00056 | 52.47937 | 0.00055 | 0.00055 |
| 2022-09-06 | 2022-09-08 | 657,938 | 402 | 0.00061 | 49.20687 | 0.00060 | 0.00060 |
| 2022-09-06 | 2022-09-09 | 891,573 | 367 | 0.00041 | 38.39394 | 0.00041 | 0.00041 |
| 2022-09-06 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00104 | 0.00104 |
| 2022-09-07 | 2022-09-08 | 657,938 | 402 | 0.00061 | 49.20687 | 0.00060 | 0.00060 |
| 2022-09-07 | 2022-09-09 | 891,573 | 367 | 0.00041 | 38.39394 | 0.00041 | 0.00041 |
| 2022-09-07 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00104 | 0.00104 |
| 2022-09-08 | 2022-09-09 | 891,573 | 367 | 0.00041 | 38.39394 | 0.00041 | 0.00041 |
| 2022-09-08 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00105 | 0.00106 |
| 2022-09-09 | 2022-09-10 | 282,877 | 301 | 0.00106 | 46.77661 | 0.00104 | 0.00105 |

## Reliability table (holding day 09-10)

The last validation day, 2022-09-10 (282,877 rows, 301 positives), scored by each earlier fit day. A calibrated
model should have `mean_pred` close to `observed` in every bin. **In practice every row lands in the first bin** - the calibrated probabilities are all below
0.10 - so calibration is only verified in the small-probability regime. The high
end of the table is untested because the folding never produces confident rows.
Do not extrapolate these probabilities to a wider alert budget without refitting.

Fitted on 2022-09-09 (Platt coef 1.098, intercept -3.970); applied to 2022-09-10:

| Bin | lo | hi | n | mean predicted | observed |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.10 | 282,858 | 0.00055 | 0.00106 |
| 1 | 0.10 | 0.20 | 19 | 0.12048 | 0.05263 |

## Recommendation

- Use **Platt** calibration for any downstream probability: it is fitted on an
  earlier day, it takes 2 parameters, and it is stable on ~300 positives.
- Do **not** use isotonic at this data scale.
- Do **not** compare an uncalibrated Rung 1 score to a probability anywhere;
  `results/calibration/calibration.json` holds the final Platt coefficients per
  fit day for reproducibility.

## Reproducing

```
python src/models/model_ladder.py --rung 1
python src/models/calibration.py
```
