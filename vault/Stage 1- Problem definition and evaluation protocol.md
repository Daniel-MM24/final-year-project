
TASK: Write the project's evaluation contract as a short spec plus a reusable Python
module, before any further modelling.

Decide and document, with reasoning, the following:
1. Prediction unit. Default: the transaction (edge), label "Is Laundering". Explain why the
   derived node label (is_laundering_node, 0.74% positive) is a weaker, noisier proxy that
   also labels clean counterparties, and restrict it to exploratory use.
2. Split protocol. Rolling-origin with a 1-day embargo. Specify the exact rules for:
   train window, embargo day, validation day, minimum validation size (50), single-class
   folds (skip), and which calendar days are excluded as unusable (inspect the daily table;
   do not assume days 11-17, measure it).
3. Metrics. PR-AUC per fold, mean and std across usable folds, lift over each fold's own
   base rate, and recall at fixed alert budgets (for example top 100 and top 1,000 alerts
   per validation day). State how metrics are aggregated across folds with very different
   base rates.
4. Calibration policy. scale_pos_weight distorts probabilities. Specify how scores will be
   recalibrated (isotonic or Platt, fitted on a held-out earlier fold) before any
   threshold is chosen.

DELIVERABLES
- docs/evaluation_contract.md (max 2 pages).
- src/eval_protocol.py with: make_splits(), is_fold_usable(), score_fold(),
  recall_at_k(), summarise_folds(). Include unit tests that fail if a single-class fold
  is averaged in, or if a validation row's date is on or before the train cutoff.

ACCEPTANCE
- Tests pass. The module reproduces the existing node-level mean of 0.0408 when run on the
  same folds, to within rounding. If it does not, explain the difference before continuing.