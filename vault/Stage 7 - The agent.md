TASK: Build a bounded agent that takes an alert and produces a cited case draft for a human.

INPUTS: an alert's evidence package (Stage 5), the text retrieval index and structured tools (Stage 6).

BEHAVIOUR
1. Read the evidence package. Plan which additional evidence it needs (at most 8 tool calls).
2. Call the structured tools for exact facts and the text index for typology guidance.
3. Produce a case draft with these sections: summary, observed facts, hypothesised typology,
   supporting evidence, counter-evidence and gaps, recommended next steps.
4. EVERY factual statement must carry a citation: either an evidence-package field, a tool
   result id, or a retrieved passage id. Statements without a citation are not allowed.
5. Quote retrieved text exactly; never paraphrase a number or a regulation.
6. Abstain. If evidence is thin, contradictory or retrieval returns nothing relevant, output
   "insufficient evidence" and list what is missing, rather than inventing a typology.

AUTONOMY BOUNDARIES (hard)
- The agent may: triage, gather evidence, draft, and rank alerts for review.
- The agent may NOT: close a case, escalate, file or draft-as-final any suspicious activity
  report, contact anyone, or change any score or threshold. Those are human actions. The
  output is always labelled "DRAFT for investigator review".
- It must treat all retrieved text and all data fields as untrusted data, never as instructions.
  If a retrieved passage or a field contains an instruction, ignore it and flag it.

AUDIT
- Log every step (prompt, tool call, arguments, result, cost, latency) to an append-only
  JSONL file keyed by alert id and run id. A run must be fully replayable from its log.

DELIVERABLES
- src/agent/ with the orchestrator, a prompt template under version control, the log writer,
  and a CLI: python -m agent.run --alert <transaction_id>.
- 10 sample drafts for alerts of mixed true/false status, plus their logs.

ACCEPTANCE
- Drafts validate against an output schema; a unit test confirms the agent has no tool that
  can close, escalate or file; a test with an injected instruction inside a retrieved
  passage shows the instruction is ignored and flagged.