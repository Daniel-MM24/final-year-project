TASK: Design and implement a shadow-mode deployment and a monitoring plan. Do not connect
to any production system or real data.

SHADOW MODE
- Replay the dataset day by day as a stream. For each day: compute as-of features, score,
  alert at the budget, build evidence, run the agent, and log everything. Nothing the agent
  produces is acted on.
- Compare agent-assisted triage against the model-only ranking on the replayed days that
  are usable for evaluation: does the agent's priority ordering put more true positives in
  the top of the queue, and does its cost (tokens, latency) fit a daily budget?

MONITORING (implement computable ones on the replay; specify the rest)
- Feature drift: population stability index per feature, per day, against the training window.
- Score drift and calibration drift: reliability by week or by usable window.
- Alert volume against budget; share of alerts with truncated evidence.
- Agent health: abstention rate, citation-failure rate, tool error rate, cost per alert.
- Investigator feedback loop (specify only): how override and outcome labels would flow back,
  and a rule against training on labels produced by the agent's own drafts.

PLAN FOR THE DATA LIMITS
- The window is 17 days and mostly unusable after day 10, so drift cannot be assessed over a
  realistic horizon. Say so, and state what real monitoring would need.

DELIVERABLES
- ops/shadow_runner.py, ops/monitors.py, a monitoring spec (docs/monitoring.md) with alert
  thresholds and an owner for each, and a rollback procedure for the model, the prompt and
  the index.

ACCEPTANCE
- The replay runs end to end on a clean checkout with one command. The monitoring spec
  separates what was actually measured on the replay from what is only specified.