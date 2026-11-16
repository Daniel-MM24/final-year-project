TASK: Run the model ladder in order. Each rung must beat the previous one on the SAME folds
before the next is built. Use src/eval_protocol.py from Stage 1 and the features from Stage 3.

RUNG 0: Rules. Implement two or three simple deterministic flags (for example very high
sender fan-out, an unusually large amount for the sender's history). Report PR-AUC and
recall at the alert budgets. This is the floor.

RUNG 1: LightGBM, edge level, target "Is Laundering". Use the Stage 3 transaction, sender
and receiver features, but NOT the neighbour aggregates yet. Use scale_pos_weight for
training. Report per-fold PR-AUC (usable folds only), mean and std, lift over each fold's
base rate, recall at budget, and mean feature importance. This is the number a GNN must
beat. (The existing node-level 0.0408 is NOT comparable: different label, different
population, different base rate. Do not compare them.)

RUNG 2: Rung 1 plus the 1-hop neighbour aggregates. Report the delta over Rung 1 with a
confidence interval (bootstrap across validation rows within each fold). Say explicitly
whether the delta is distinguishable from noise.

RUNG 3 (GATED): Build a GNN edge classifier (RGCN or GAT in PyTorch Geometric) ONLY if
Rung 2 leaves headroom. State the gate in advance: proceed only if you can argue, from the
Rung 1 and 2 results, that graph structure could add more than the noise level. If the gate
fails, stop and write why. If it passes, use temporal neighbour sampling that respects the
as-of rule, the same folds, and the same metrics.

CALIBRATION
For the best model, fit isotonic or Platt calibration on an earlier fold, apply it to a
later fold, and report a reliability table and Brier score before and after. Do not report
raw predict_proba from a scale_pos_weight model as a probability.

DELIVERABLES
- results/model_ladder.md: a single table of rungs x metrics, with the usable-fold count
  next to every number, and the decision at each gate.
- Saved models, the exact feature list, and the seed.

ACCEPTANCE
- Every number is reproducible from a script. No fold with fewer than 50 rows or a single
  class is included. Negative results (a rung that does not help) are reported, not hidden.