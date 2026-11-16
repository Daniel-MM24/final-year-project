# State - last updated 2026-11-16

- **Stages 1-2: complete.** `docs/evaluation_contract.md`, `src/eval_protocol.py`, `tests/test_eval_protocol.py` (24/24). 8 folds validate 09-03..09-10; 5/5 EDA items answered, `docs/data_contract.md`. Baseline 0.0408 **not reproduced** (0.0128, lift 1.00) - not a gate.
- Env: `/home/dan/miniconda3/envs/learn_env/bin/python` (only env with pandas+lgbm+sklearn+pyarrow); run from project root.
- Key facts: edge base rate 0.0515% (3,565 pos); node 0.7514% (5,304/705,903). Repeated edges are padding, not recurrence (14x depleted). Never use `fx_pair_rates.csv` for Bitcoin.
- **Stage 3: complete.** `eda_output/features/edge_features_v1.parquet` (6,924,049 rows x 38 cols, 34 features) via `src/features/build_edge_features.py`. `rel_count_prior` dropped (corr 1.0 with `from_fan_out`). Leakage tests 3/3 pass (`src/features/leakage_tests.py`). Manifest: `docs/feature_manifest.json`.
- **Stage 4: complete.** Rungs 0-2 in `src/models/model_ladder.py`, calibration in `src/models/calibration.py`. Results: `results/model_ladder.md`, `results/calibration/calibration.md`. PR-AUC / weighted lift / recall@100: Rung 0 (rules) 0.00205/3.27/0.0083; Rung 1 (LightGBM, 24 feats) 0.02395/41.50/0.0220; Rung 2 (+4 hop feats) 0.03333/59.57/0.0296. Boosters in `results/ladder/models/` (16 files).
- Earlier Stage 4 numbers were invalid (sigmoid saturation, margin divergence, OOM); all three fixes documented in `results/model_ladder.md`. Fold 09-03 is degenerate (112 distinct scores) and drives the variance; not excluded, per protocol.
- **Rung 3 GNN gate CLOSED** again: Rung 2 improves 7/8 folds, mean lift +16.98 vs one std 19.23. Calibration: Platt 44.85 -> 0.00059 Brier, beats constant-zero on 20/28 pairs, reliable only below P<0.10. Leakage 3/3.
- Next: Stage 5 (alerting and evidence packaging). Operative metric is recall@k, not PR-AUC.
