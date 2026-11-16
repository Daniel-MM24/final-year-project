TASK: Build an evaluation harness that scores each layer separately, so a failure can be
located rather than only detected.

LAYER 1: Model. Already covered by the Stage 1 protocol. Re-run it as a regression gate: fail
if mean usable-fold PR-AUC or recall at budget drops beyond a tolerance you define.

LAYER 2: Retrieval. recall@5 and MRR on the Stage 6 test set; also a "should return nothing"
set of off-topic queries, scored on correct abstention.

LAYER 3: Narrative faithfulness. For each draft, extract every factual claim (numbers,
account ids, dates, typology assertions) and check it against its cited source:
- numeric claims: exact match against the evidence package or tool result;
- typology claims: the cited passage must actually say it (use an LLM judge, then verify
  the judge against 40 hand-labelled claims and report its agreement rate);
- uncited claims count as failures.
Report: claim-level faithfulness, fabricated-number rate, uncited-claim rate.

LAYER 4: Agent behaviour. Tool-selection correctness against expected tool lists on a set of
scripted alerts; abstention rate on deliberately under-evidenced alerts (target: abstains);
prompt-injection tests (target: 0 followed); boundary tests (target: 0 forbidden actions).

REPORTING
- One report with a table per layer, the sample size for each number, and a plain-language
  "what this does not show". Be explicit that this data has no real investigator narratives,
  so narrative quality against real compliance judgement is UNMEASURED, and say so.

DELIVERABLES
- eval/ with the harness, the labelled sets, and a single command that runs all layers.

ACCEPTANCE
- Re-running produces the same numbers (fixed seeds, logged model versions). Every metric
  states its n and its uncertainty.