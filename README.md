# AML transaction-graph project

An end-to-end anti-money-laundering pipeline built on the IBM AML synthetic
dataset: as-of edge-level features, a rolling-origin evaluation protocol, a model
ladder with a pre-stated gate for graph models, and calibrated risk scores.

> **The data is synthetic.** Every number in this repository is measured on
> simulator output. Nothing here is evidence about real-world laundering
> behaviour.

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Problem definition and evaluation protocol | complete |
| 2 | EDA consolidation and data contract | complete |
| 3 | As-of feature engineering (edge level) | complete |
| 4 | Model ladder (rules → LightGBM → GNN gate) | complete |
| 5 | Alerting and evidence packaging | not started |
| 6 | RAG layer | not started |
| 7 | Case-draft agent | not started |
| 8 | Layered evaluation | not started |
| 9 | Deployment and monitoring | not started |

## Results so far

Edge-level classification: a transaction is flagged as laundering or not. Trained
on rolling-origin folds over validation days 2022-09-03 .. 2022-09-10 with a
1-day embargo; folds with fewer than 50 rows or a single class are skipped and
never averaged in.

| Rung | Model | PR-AUC | weighted lift | recall@100 | recall@1000 |
|---|---|---|---|---|---|
| 0 | deterministic rules | 0.00205 ± 0.00062 | 3.27 | 0.0083 | 0.0321 |
| 1 | LightGBM, 24 edge features | 0.02395 ± 0.01144 | 41.50 | 0.0220 | 0.1407 |
| 2 | + 4 hop-neighbourhood features | **0.03333 ± 0.01794** | **59.57** | **0.0296** | **0.1563** |
| 3 | GNN edge classifier | not built — gate closed | | | |

**The GNN gate is closed.** Rung 2 improves on Rung 1 on 7 of 8 folds, but the
mean per-fold lift gain (+16.98) does not exceed one standard deviation of
per-fold lift (19.23). The evidence is directional and consistent, not
certifiable at ~300 positives per fold.

**The operative number is recall@100, not PR-AUC.** At an alert budget of 100 per
day the best model catches 3–19 of ~300 positives. It ranks the whole book well
(lift ~60×) but does not isolate a small high-confidence core. Full discussion,
including the three scoring bugs that had to be fixed before these numbers were
trustworthy, is in [`results/model_ladder.md`](results/model_ladder.md).

Calibration is fitted on an earlier validation day and applied to a later one.
Platt scaling cuts Brier from 44.85 (raw margin used as a probability) to
0.00059; see [`results/calibration/calibration.md`](results/calibration/calibration.md).

## Layout

```
src/eval_protocol.py            rolling-origin folds, embargo, PR-AUC / recall@k
src/features/                   edge-feature build and leakage tests
src/models/model_ladder.py      rungs 0-2, per-fold scoring, saturation guard
src/models/calibration.py       Platt / isotonic, fit on earlier day, score later
src/eda_item*.py                the five EDA open items (Stage 2)
tests/                          protocol tests (24 passing)
docs/                           evaluation contract, data contract, feature manifest
results/                        per-stage write-ups and summaries
```

## Data and prerequisites

The raw dataset is **not tracked** (it is ~800 MB). Expected inputs under `data/`:

- `LI-Small_Trans.csv` and the associated account/pattern files
- `_trans_canonical.parquet`, `_trans_vocab.json` — canonicalised tables built
  from the CSVs

Derived tables are rebuilt from scripts and are also untracked
(`eda_output/features/`, score parquets, boosters). See `.gitignore`.

Python environment: `learn_env` (3.10) with `pandas`, `pyarrow`, `lightgbm`,
`scikit-learn`, `pytest`. No Polars.

## Reproducing

```bash
python src/measure_daily_table.py               # daily table used by the folds
python src/eda_item1_duplicates.py              # EDA items 1-4
python src/eda_item2_prefixes.py
python src/eda_item3_banks.py
python src/eda_item4_entities.py

python src/features/build_edge_features.py      # ~85 s, 346 MB parquet
python src/features/leakage_tests.py            # 3/3 pass

python src/models/model_ladder.py --rung 0
python src/models/model_ladder.py --rung 1 --save-models
python src/models/model_ladder.py --rung 2 --save-models
python src/models/calibration.py

pytest tests/                                   # 24 tests
```

Each rung writes `results/ladder/<rung>_scores.parquet` and
`<rung>_summary.json`; `--save-models` additionally writes one LightGBM booster
per fold to `results/ladder/models/`. Seed is 42.

## Project rules

These are enforced throughout the code and checked by the leakage tests:

1. Count distinct items with `.unique()` / `.nunique()`, never `len()`.
2. Use `amount_usd` from the saved FX tables, never raw amounts across currencies.
3. `.astype(float)` after category `.map()`; `observed=True` in groupby.
4. Features must be as-of — strictly earlier data only, no whole-file aggregates.
5. Skip validation folds with <50 rows or one class; never average them in.
6. Never use raw `account_id` or the `10042B` prefix family as model input.
7. Report PR-AUC and recall at the alert budget, on rolling-origin splits with a
   1-day embargo.
8. Data is synthetic; never present results as real-world evidence.

## Licence

See [`LICENSE`](LICENSE).
