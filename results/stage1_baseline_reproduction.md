# Stage 1: attempted reproduction of the node-level baseline

**Outcome: not reproduced. Reproduction is not a gate for Stage 1.**

The original baseline (`baseline_lightgbm.py`, saved in
`eda_output/baseline_lightgbm_{folds,summary}.json`) reports node-level mean PR-AUC
**0.0408** (std 0.0178) over 4 usable folds of 10 planned.

## What was attempted

`src/reproduce_baseline.py` rebuilds the inputs from raw data using the `amount_usd` rule
from `docs/data_contract.md` s3 (which Stage 2 supplies), builds as-of node features per
fold, fits LightGBM, and scores each fold through `src/eval_protocol.py`.

## Result

```
2022-09-07: train 702,031  val 893  pos 11  base 1.232%  PR-AUC 0.0123
2022-09-08: train 702,912  val 760  pos  9  base 1.184%  PR-AUC 0.0118
2022-09-09: train 703,805  val 653  pos 12  base 1.838%  PR-AUC 0.0184
2022-09-10: train 704,565  val 683  pos  6  base 0.878%  PR-AUC 0.0088
node-level mean PR-AUC 0.0128 (std 0.0035)   weighted lift 1.00
```

Verbatim from `/tmp/repro.log`.

Weighted lift of exactly 1.00 means the model has **no ranking power** - every fold's
PR-AUC equals its base rate.

## Why it does not reproduce

1. **Validation sizes differ by two orders of magnitude.** The original validated on
   11,145 / 11,409 / 188,920 / 103,358 nodes. This script validates only on nodes whose
   `date_first_seen` equals the validation day: 653-893 nodes, 6-12 positives. PR-AUC on
   6-12 positives is noise-dominated and cannot reach 0.04 by construction.
2. **Node-selection logic is not in `baseline_lightgbm.py`.** The saved JSON's
   `n_train` / `n_val` show the original used a node set that this script cannot
   reconstruct from the script alone - the notebook that produced the saved run is the
   missing piece. `eda_output/account_features.parquet` is a whole-file aggregate
   (returns `date` 2022-09-10 with lifetime counts, and contains `10042B*` IDs), so it
   cannot supply as-of features either.
3. **Train sizes are wrong too.** This script trains on 702,031-704,565 nodes; the
   original trained on 366,367-577,841. Two of the original's four train sets are smaller
   than its validation sets, which my run never reproduces.
4. The `<50 rows` skip rule did not fire (653-893 > 50), but 6-12 positives is effectively
   a single-class problem - the protocol's row floor is not sufficient protection for
   node-level folds.

## What this does and does not tell us

- It does **not** undermine `eval_protocol`. The protocol scored what it was given and
  correctly reported PR-AUC at base rate; `lift_weighted = 1.00` is the protocol working,
  not failing.
- It **does** say the original 0.0408 number is not independently reproducible from the
  committed code. Treat 0.0408 as a historical reference, not a verified result.
- The protocol's headline metric is **edge-level**, per `docs/data_contract.md` s6.1. The
  node number was only ever an internal diagnostic.

## Recommended follow-up (optional, not blocking)

If the 0.0408 figure must be used in the report, either cite it as the previous
baseline's reported figure without independent confirmation, or reconstruct the original
node selection from the notebook. Neither is needed for Stage 3 onward.
