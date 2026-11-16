# AML project vault

Obsidian vault at `vault/`. Each note is a self-contained prompt for one stage.
Read ONLY the stage you are asked to work on. Never scan the vault.

All deliverable paths in the Stage notes (`docs/`, `src/`, `results/`, `eval/`,
`ops/`, `corpus/`, `notebooks/`) are relative to the project root.

## Always-on rules (full detail in vault/Shared Context.md)
1. Count distinct items with .unique()/.nunique(), never len().
2. Use amount_usd from the SAVED fx tables; never raw amounts across currencies.
3. Category dtype: .astype(float) after .map(); observed=True in groupby.
4. Features must be as-of (strictly earlier data only). No whole-file aggregates.
5. Skip validation folds with <50 rows or one class. Never average them in.
6. Never use raw account_id or the 10042B prefix family as model input.
7. Metrics: PR-AUC and recall at alert budget. Rolling-origin splits, 1-day embargo.
8. Data is synthetic. Do not present results as real-world evidence.

## Map (open a file only when its trigger applies)
| File | Read when |
|---|---|
| state/STATE.md | Start of every session. Always read first. |
| vault/Shared Context.md | You need dataset facts, known bugs, or hard rules. |
| vault/Stage 1- Problem definition and evaluation protocol.md | Working on evaluation protocol, folds, metrics, calibration. |
| vault/Stage 2 - EDA consolidation and data contract.md | Working on EDA open items or the data contract. |
| vault/Stage 3 - As-of feature engineering (edge level).md | Building features or leakage tests. |
| vault/Stage 4 - Model ladder.md | Running the model ladder (rules, LightGBM, GNN gate). |
| vault/Stage 5 - Alerting and evidence packaging.md | Building alerts or evidence packages. |
| vault/Stage 6 - The RAG layer.md | Building text retrieval or structured tools. |
| vault/Stage 7 - The agent.md | Building the case-draft agent. |
| vault/Stage 8 - Layered evaluation.md | Building the evaluation harness. |
| vault/Stage 9 - Deployment and monitoring.md | Shadow mode or monitoring. |

## Existing code and artifacts
- Notebooks (root): `l1-small_accoints.ipynb`, `IBM_AML_LI_Small_EDA.ipynb`,
  `IBM_AML_LI_Small_EDA(1).ipynb`, `AML_Baseline_LightGBM.ipynb`.
- `baseline_lightgbm.py` — node-level baseline that produced 0.0408.
- `data/` — raw CSVs. `eda_output/` — saved fx tables, account_features, baselines.

## Working rules
- Notes use [[wikilinks]]; they are not followed automatically. Use the paths above.
- Read notes with an offset/limit or grep for a heading when you need one section.
- Do not open data/, eda_output/*.parquet or notebooks. Use scripts that print summaries.
- Big exploration (many files, large outputs): delegate to a subagent and keep only its summary.
- Make outputs summaries: print df.describe() or folded tables, not raw dumps.
- When a stage finishes, update state/STATE.md (10 lines max), then stop.
