TASK: Turn model scores into investigator-ready alerts with a self-contained evidence package.

ALERT POLICY
- Choose the alert threshold from an alert BUDGET (alerts per day an investigation team can
  review), not from a probability cutoff. Make the budget a parameter. For budgets of 50,
  200 and 1,000 alerts per day, report precision, recall and the number of true positives
  caught, using calibrated scores on usable validation folds.
- Report what is lost when the budget is tight, for example how many laundering edges
  belong to a chain where only one edge is alerted.

EVIDENCE PACKAGE (one JSON document per alert, schema-validated)
- The transaction: ids, timestamp, USD amount, original amount and currency, payment
  format, banks.
- Model output: calibrated score, rank within the day, top SHAP drivers (feature, value,
  direction, magnitude), computed from the as-of features only.
- Context: sender and receiver as-of history (counts, amounts, fan-in/out, account age) and
  the 1-hop and 2-hop neighbourhood subgraph, capped to a size limit with a "truncated" flag.
- Detected structure: cycles of length 3-6 through either account, fan-out and fan-in star
  indicators, chains of transfers within a short window. Self-loops are excluded.
- Provenance: feature version, model version, data cutoff time. No field may use data later
  than the alert's own timestamp.

DELIVERABLES
- src/evidence.py with build_evidence(transaction_id) and a JSON Schema.
- A sample of 20 packages and a script that validates them.
- A leakage test: build the package for an old transaction twice, once with the full file
  loaded and once truncated to its own timestamp, and assert the two are identical.

ACCEPTANCE
- Packages are deterministic, schema-valid and under a size limit; the leakage test passes.
- Every SHAP claim in a package can be re-derived from the stored features.