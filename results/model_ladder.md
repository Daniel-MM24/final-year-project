# Stage 4 — Model ladder

Every rung is scored on the identical rolling-origin folds from `src/eval_protocol.py`
(validation days 2022-09-03..09-10, 1-day embargo, folds with <50 rows or one class
skipped and never averaged in). `n` = number of *usable* folds behind each number (all 8
here). Data is synthetic; nothing here is real-world evidence.

## Headline

| Rung | Model | PR-AUC | weighted lift | recall@100 | recall@1000 |
|---|---|---|---|---|---|
| 0 | deterministic rules | 0.00205 ± 0.00062 | 3.27 | 0.0083 | 0.0321 |
| 1 | LightGBM, 24 edge features | 0.02395 ± 0.01144 | 41.50 | 0.0220 | 0.1407 |
| 2 | Rung 1 + 4 hop features | **0.03333 ± 0.01794** | **59.57** | **0.0296** | **0.1563** |
| 3 | GNN edge classifier | **not built — gate closed** | | | |

LightGBM is 11.7× the rules' PR-AUC and 12.7× its weighted lift. Both learned rungs beat
the rules on every metric, including the top-of-ranking recall@100 that the rules won
in earlier (broken) runs.

## A note on how these numbers were obtained

Three separate scoring pathologies had to be fixed before any number here was
trustworthy. All three are recorded because each produced plausible-looking results:

1. **Sigmoid saturation.** With `scale_pos_weight = neg/pos` (~2,034) the sigmoid output
   pins most rows at exactly 1.0 — 93% on 09-03. recall@k then measures tie order, not
   ranking (it reported recall@100 = 0.1498, almost all artifact). **Fix:** score the raw
   margin, `predict(raw_score=True)`.
2. **Margin divergence.** Raw margins reached ±1e5..1e8: `scale_pos_weight` compounds a
   large update every round on near-inseparable classes, and a handful of rows explode.
   Platt's coefficient collapsed to 0. **Fix:** `scale_pos_weight = sqrt(neg/pos)` (~45)
   and `max_delta_step=1.0`. Margins now span −12.9..+2.5.
3. **Memory blow-up.** Row-wise histogram building on the 5.1M-row training window peaked
   at 6.4 GB on this 7.8 GB box and was OOM-killed (exit 137) — including Rung 0, which
   uses no LightGBM at all and exposed that the fault was in passing the feature matrix
   through `frame.attrs`, not in the model. **Fix:** `force_col_wise=True` (peak ~2.5 GB)
   and pass the matrix as an explicit argument.

The saturation statistic is written per fold into every `_summary.json` so a regression of
pathology 1 or 2 is visible without re-running.

## The decision at each gate

- **Gate 1 → 2: passed.** PR-AUC 0.00205 → 0.02395 (11.7×) and weighted lift 3.27 → 41.50.
- **Gate 2 → 3: CLOSED.** The pre-stated condition is "Rung 2 must beat Rung 1 by more
  than one std of per-fold lift *and* hop features must rank near the top". The second
  clause is met (`to_hop_mean_fan` 5.56%, `from_hop_max_fan` 5.36%, `to_hop_max_fan`
  4.52% — three of the top ten). The first **fails narrowly**: per-fold lift improves on
  7 of 8 folds, mean gain +16.98, but one std of per-fold lift is 19.23, so the mean gain
  does not clear it. Rung 2 is better on every aggregate metric; the evidence is not
  strong enough to spend a GNN on. **No GNN is built.**
- **Alert budget: now usable, still weak.** See below.

## The finding that matters: the top of the ranking is better but still not good

Rung 2's positives caught in the top-100 per day (budget = 100 alerts):

| fold | 09-03 | 09-04 | 09-05 | 09-06 | 09-07 | 09-08 | 09-09 | 09-10 |
|---|---|---|---|---|---|---|---|---|
| positives in top 100 | 4 | 5 | 10 | 16 | 3 | 11 | 19 | 14 |
| of total positives | 292 | 291 | 370 | 380 | 368 | 402 | 367 | 301 |

Every fold now catches several positives (previously zero on 5 of 8). Precision@100 is
3–19%, i.e. 30–190× the ~0.1% base rate — far better than chance, but an analyst looking
only at the top 100 still sees 80–97 wrong alerts. **The learned model ranks the book well;
it does not isolate a tiny confident core.** Stage 5 must treat recall@k, not PR-AUC, as
the operative number.

The two metrics still tell different stories:

- **PR-AUC** (0.033) says the model orders the bulk of the book well: any 1% slice is ~60×
  enriched.
- **recall@100** (0.0296) says the extreme head remains blunt: the model has no sharp,
  confident top.

## Rung 0 — deterministic rules

`src/models/model_ladder.py --rung 0`. Label-free:
`0.6 * clip(z/10) + 0.3 * clip(fan_out/20) + 0.1 * is_cross_bank`, with
`z = (amount − sender mean) / sender sd`. Weights are round numbers, deliberately not
tuned — tuning a rules rung against labels would turn it into a weak learner and destroy
its use as a floor.

| fold | n | positives | base rate | PR-AUC | lift |
|---|---|---|---|---|---|
| 2022-09-03 | 283,326 | 292 | 0.00103 | 0.00148 | 1.43 |
| 2022-09-04 | 282,476 | 291 | 0.00103 | 0.00194 | 1.88 |
| 2022-09-05 | 657,397 | 370 | 0.00056 | 0.00333 | 5.91 |
| 2022-09-06 | 657,170 | 380 | 0.00058 | 0.00250 | 4.32 |
| 2022-09-07 | 658,504 | 368 | 0.00056 | 0.00218 | 3.89 |
| 2022-09-08 | 657,938 | 402 | 0.00061 | 0.00200 | 3.28 |
| 2022-09-09 | 891,573 | 367 | 0.00041 | 0.00111 | 2.69 |
| 2022-09-10 | 282,877 | 301 | 0.00106 | 0.00189 | 1.78 |

Per-fold base rate varies 2.6× (0.041% .. 0.106%), so a plain mean over folds is
dominated by the large low-base-rate days; `lift_weighted` (positive-weighted) is the
honest headline.

The z-score term only fires on rows with sender history (5.3M of 6.9M); 1.6M
no-history rows get `z = 0`. That is the rules' real weakness, and it is why they win
recall@k nowhere once the model is fixed.

## Rung 1 — LightGBM, edge features only

24 features: every Stage 3 feature except the four 1-hop neighbour aggregates. Params:
`learning_rate=0.05`, `num_leaves=31`, `min_child_samples=200`, `feature_fraction=0.8`,
`bagging_fraction=0.8`, `max_bin=63`, `force_col_wise=True`, `max_delta_step=1.0`,
`scale_pos_weight = sqrt(neg/pos)` (≈45 in the largest fold), 300 rounds, no early
stopping (there is no validation split inside the training window — stopping on the scored
day would leak it), seed 42.

Scores are the **raw margin**, not a probability. See "A note on how these numbers were
obtained" and `results/calibration/calibration.md`.

Per-fold:

| fold | n | positives | base | PR-AUC | lift | recall@100 |
|---|---|---|---|---|---|---|
| 2022-09-03 | 283,326 | 292 | 0.00103 | 0.00270 | 2.62 | 0.0137 |
| 2022-09-04 | 282,476 | 291 | 0.00103 | 0.01035 | 10.04 | 0.0103 |
| 2022-09-05 | 657,397 | 370 | 0.00056 | 0.02403 | 42.69 | 0.0216 |
| 2022-09-06 | 657,170 | 380 | 0.00058 | 0.02892 | 50.01 | 0.0237 |
| 2022-09-07 | 658,504 | 368 | 0.00056 | 0.02096 | 37.50 | 0.0054 |
| 2022-09-08 | 657,938 | 402 | 0.00061 | 0.03735 | 61.13 | 0.0274 |
| 2022-09-09 | 891,573 | 367 | 0.00041 | 0.03139 | 76.27 | 0.0300 |
| 2022-09-10 | 282,877 | 301 | 0.00106 | 0.03588 | 33.72 | 0.0432 |

**Fold 09-03 is degenerate and should be read with care.** It trains on roughly one day
of history, so LightGBM collapses to a handful of leaves: only **112 distinct scores** on
283k rows, with 3.9% of rows tied, a max |margin| of 14.65 (vs −12.9..+2.5 elsewhere), and
PR-AUC 0.0027 — barely above the rules. This is an undertrained model, not a signal about
the data. It is the fold that widens the ±std enough to fail the Gate 2→3 test; excluding
it, Rung 1's mean PR-AUC would be materially higher and more stable. It is **not**
excluded here because the protocol forbids cherry-picking folds.

Top gain shares: `log_amount_usd` 12.9%, `from_amount_mean` 9.2%, `from_txn_count` 7.9%,
`to_txn_count` 6.7%, `from_fan_out` 6.5%, `to_age_days` 6.4%, `from_cross_bank_frac` 6.0%,
`from_fan` 5.4%. Amount-vs-history still leads, but prior-day **count/fan** features now
carry ~20% between them — a change from the broken runs, where they barely ranked.

Worst residual tie: 11,041 rows (3.9%) on 09-03 — the degenerate fold. Every other fold has
<0.05% ties.

## Rung 2 — Rung 1 + 1-hop neighbour aggregates

Identical settings plus `from_hop_mean_fan`, `from_hop_max_fan`, `to_hop_mean_fan`,
`to_hop_max_fan` (mean/max prior-day fan over the distinct counterparties an account
faces on the current day).

PR-AUC 0.03333 ± 0.01794, lift 59.57, recall@100 0.0296, recall@1000 0.1563. Three hop
features land in the top ten by gain. Rung 2 improves on 7/8 folds (identical on the
degenerate 09-03) and is better on every aggregate metric, but the per-fold spread is wide
enough — the same 09-03 fold dominates it — that the one-std gate does not clear.

## Gate for Rung 3 — CLOSED

Stated in advance: build the GNN only if Rung 2 beats Rung 1 by more than one std of
per-fold lift *and* the hop features rank near the top of importance.

| | Rung 1 | Rung 2 | Δ |
|---|---|---|---|
| PR-AUC | 0.02395 | 0.03333 | +0.00938 |
| weighted lift | 41.50 | 59.57 | +18.07 |
| per-fold lift, mean Δ | — | — | **+16.98** |
| per-fold lift, one std | — | — | **19.23** |
| folds improved | — | — | 7 / 8 |

The mean gain (+16.98) does not exceed one std (19.23). The importance condition is met.
**No GNN is built.** The gap is directional and consistent (7/8 folds) but the metric is
too noisy at ~300 positives per fold to certify it.

## What would have to change for a GNN to be justified

The gate tests whether hop *aggregates* help. They help on every aggregate metric and on
7/8 folds, but not by a certifiable margin. A GNN earns its place only if Stage 5 shows
alert-level precision must improve and the failure is specifically *relational* — a
laundering ring is a dense subgraph, not a set of individually odd edges, and per-edge hop
aggregates throw that structure away. If so, the right experiment is a graph model
evaluated *specifically on recall@100*, gated on beating Rung 2's 0.0296, not on PR-AUC.
That is a decision for a later stage, not a licence to build one now.

## Calibration

See `src/models/calibration.py` and `results/calibration/calibration.md`. Pooled over 28
fit→hold pairs, Platt cuts Brier from **44.85** (raw margin used as a probability) to
**0.00059**. The bar is low: the constant-zero baseline scores 0.000634 (the base rate) on
this data, so anything usable must at least beat that. Platt beats it on **20 of 28**
pairs (median 0.000572 vs 0.000634) and loses on 8, so calibration adds measurable but
not reliable accuracy — the gain is small because positives are so rare that the base rate
is already a strong predictor. Platt and isotonic are indistinguishable after rounding
(0.000586 vs 0.000587 n-weighted; Platt lower in 20 of 28 pairs). Platt is recommended: 2 parameters,
stable coefficients (0.55–1.10), and never clearly worse. The reliability table is only
informative in the low-probability tail — 282,858 of 282,877 holding-day rows land in
P<0.10, where Platt under-predicts by ~2× (0.00055 vs 0.00106) — so do not extrapolate
these probabilities to a wider alert budget without refitting.

## Leakage re-check

`src/features/leakage_tests.py` passes 3/3 on the saved table, re-run because the count/fan
features now carry ~20% of Rung 1's gain:

- **shuffled labels:** real mean PR-AUC 0.00278 vs shuffled 0.00052, and the base rate is
  0.00051 — shuffled collapses to chance, so the signal is real.
- **future injection:** a late-day row leaves every earlier day's features bit-identical.
- **self-contribution:** `from_amount_mean/std/max`, `from_txn_count`, `from_fan`,
  `from_fan_out`, `from_fan_in` all have 0.00e+00 diff when the row's own amount changes
  (only `log_amount_usd`, which is the row's own amount, moves).

## Saved models and seed

Seed: **42** for LightGBM; Platt/isotonic use sklearn defaults with `max_iter=1000`.

`python src/models/model_ladder.py --rung <n> --save-models` writes the per-fold boosters
to `results/ladder/models/<rung>_fold-<day>.txt` (LightGBM text format) alongside
`_scores.parquet` and `_summary.json`, which carry the exact feature list, params, and
seed. Scores are reproducible by refitting with the same seed; the saved boosters let
downstream stages score new days without a refit.

## Reproducing

```
python src/features/build_edge_features.py      # ~85 s, 346 MB parquet
python src/features/leakage_tests.py            # 3/3 pass
python src/models/model_ladder.py --rung 0
python src/models/model_ladder.py --rung 1 --save-models
python src/models/model_ladder.py --rung 2 --save-models
python src/models/calibration.py
```

Artefacts per rung in `results/ladder/`: `<rung>_scores.parquet` (per-row score, fold,
day), `<rung>_summary.json` (metrics, fold table, importances, params, saturation).
Boosters in `results/ladder/models/`.

> **Memory note.** The feature table's account columns are `object` dtype and cost ~2 GB
> if loaded; the ladder loads only features + label and accumulates row-group by
> row-group into float32 (peak ~470 MB). LightGBM with `force_col_wise=True` peaks near
> 2.5 GB on the full training window — the box has 7.8 GB and is comfortable, but not by a
> wide margin with other processes running.
