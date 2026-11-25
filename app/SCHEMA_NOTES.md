# Schema notes — discovered from real artifacts (2026-11-25)

All findings below were read directly from `results/` and `src/agent/`.
Nothing here is invented; where the task brief disagreed with the data, the
disagreement is called out under **Discrepancies**.

---

## 1. Evidence packages

Two directories, 28 packages total:

| Directory | Count | Label | Filename pattern |
|---|---|---|---|
| `results/evidence/` | 20 | negative (`golden_label = 0`) | `package_NN_row<ROW>.json` |
| `results/evidence/positives/` | 8 | positive (`golden_label = 1`) | `package_NN_row<ROW>.json` |

Each directory also holds a `summary.json` that is **not** a package.

**Top-level keys** of a package (all five present on every package):

```
context, features, model, provenance, transaction
```

### 1.1 `transaction` (dict)

```
amount_usd      float
date            "YYYY-MM-DD 00:00:00"       (day bucket)
from_account    int
from_bank       int
golden_label    0 | 1                      (raw simulator label; never a feature)
payment_format  str (e.g. "3")
row             int                        (canonical row id — the alert id)
to_account      int
to_bank         int
ts_ns           int (nanoseconds)
```

There is **no** `transaction.is_cross_bank`. That field is `features.is_cross_bank`.

### 1.2 `model` (dict)

```
alert           bool   (was this row in the daily alert budget?)
budget          int    (200 on the saved set)
rank_in_day     int    (1 = top of that day)
raw_margin      float  (LightGBM raw margin, not a probability)
threshold       float  (the margin cut for rank == budget)
```

The score itself is `raw_margin`. `rank_in_day` is the row's rank within
its trading day at the given `budget`.

### 1.3 `features` (dict, 36 keys)

Per-edge as-of features from Stage 3 (`edge_features_v1/v1`). Examples:
`is_cross_bank`, `amount_usd`, velocity, hop/chain features, account-history
counts. This is the only place `is_cross_bank` lives.

### 1.4 `context` (dict)

| Key | Type | Notes |
|---|---|---|
| `chains` | list | each chain is a list of hops; often empty `[]` |
| `cycles` | list | often empty |
| `fan_star` | dict | account fan-in/out structure |
| `receiver_history` | dict | prior inbound edges for the receiver |
| `sender_history` | dict | prior outbound edges for the sender |
| `subgraph` | dict | neighbourhood graph |

Chain definition (in `provenance.chain_definition`): `{min_hops: 2,
sender_eq_prev_receiver: true, max_hop_gap_ns: 86_400_000_000_000}`.

### 1.5 `provenance` (dict)

```
as_of_rule          "strictly earlier timestamps only (ts_n < ts_ns)"
canonical_table     "data/_trans_canonical.parquet"
chain_definition    {min_hops, sender_eq_prev_receiver, max_hop_gap_ns}
feature_version     "edge_features_v1/v1"
known_gaps          [str, ...]
```

---

## 2. Stage 4 scores

`results/ladder/rung2_lgbm_plus_hop_scores.parquet`

Columns: `date, fold, row, score, is_laundering` (~4.37 M rows).

This is the authoritative per-row score table. The package's `model.raw_margin`
matches the parquet's `score` for that row on that date. The app uses the
**package** value for per-alert display (always present) and the **parquet**
only on the Results page when a distribution-wide number is requested.

---

## 3. Drafts

`results/agent/drafts/alert_<ROW>__run_<UTC>-<PID>.json` — 14 files today.

One file per agent run. A given alert may have 0, 1, or 2 drafts.

### 3.1 Draft shape

```
{
  "draft": {
    "alert_id": <row>,
    "run_id":   "run_<UTC>-<PID>",
    "status":   "draft" | "insufficient_evidence",
    "label":    "DRAFT for investigator review",
    "schema_version": "1.0",
    "missing":  [str, ...],
    "flags":    [{...}, ...],
    "summary":               [{text, citations: [str, ...]}, ...],
    "observed_facts":        [...],
    "hypothesised_typology": [...],
    "supporting_evidence":   [...],
    "counter_evidence":      [...],
    "recommended_next_steps":[...]
  },
  "problems":   [str, ...],          # "<section>[<i>]: <reason>"
  "tool_calls": <int>,               # COUNT of tool calls made in the run
                                     # (NOT a list of "name:call_XX" ids —
                                     #  ids live only in the .jsonl log lines)
  "log_path":   "results/agent/logs/alert_...jsonl"
}
```

Each section is a **list of Claim objects**, each `{text, citations}`.
`status == "insufficient_evidence"` means the model abstained or the run
failed before producing a schema-valid draft. In that case every section
is `[]`.

### 3.2 Validator output (`problems[]` strings)

Each entry is the validator's own message, formatted as:

```
<claim_label>: <reason>
e.g. "counter_evidence[2]: package path 'transaction.is_cross_bank' does not resolve"
```

`problems == []` with at least one non-empty section ⇒ clean draft.

### 3.3 Real status distribution today (2026-11-25)

| Status | Count | Notes |
|---|---|---|
| `draft` with 0 problems | **1** | 6062005 second run (`…152621Z-307952`) |
| `draft` with ≥1 problems | **6** | 2959857, 3044518, 4416924, 4950559, 5387721, 5461671, 6030889, 6195450, 6745789, 3815014, 4527071 (exact list computed at runtime) |
| `insufficient_evidence` | **7** | includes 3 × 401 `AuthenticationError`, 1 × invalid-JSON, remainder from earlier failures |

Bug families seen in `problems[]`:
- `package path 'transaction.is_cross_bank' does not resolve` — correct, field is `features.is_cross_bank`.
- `unknown tool call 'case:row…'` — the model used a `case:` citation where the
  orchestrator had run **without** `search_corpus` returning that passage and
  without passing `case_ids`, so the tool-id lookup fails.
- `unknown passage 'ref:sar-faqs-october-2025:002'` — the model cited a real
  SAR FAQ passage that was never retrieved this run (`search_corpus` not called
  or hit set doesn't contain it), so `passage_ids` lacks it.

---

## 4. Run logs

`results/agent/logs/alert_<ROW>__run_<UTC>-<PID>.jsonl` — one JSON object per line.

Every record has `seq` (int, monotonic from 0), `ts` (ISO-8601 UTC),
`run_id` and `alert_id`. **The event name is in `event`, not `type`.**

| `event` | Payload summary |
|---|---|
| `run_start` | `{alert_id, model, prompt_version, schema_version, max_calls, temperature, package_sha256, package}` |
| `model_message` | `{content, tool_calls: [{id, type, function:{name, arguments}}], latency_s, turn, usage}` |
| `tool_call` | `{call_id, name, arguments, result, ok, latency_s, calls_used}` |
| `draft_parsed` | `{draft, turn}` — the model's JSON reply after parsing |
| `draft_validated` | `{status, problems, flags, passage_ids, tool_call_ids, tool_calls}` |
| `run_end` | `{ok, tool_calls, latency_s}` |

Tool citations resolve against `tool_call.call_id` (NOT `id`).
Passages resolve against `draft_validated.passage_ids` (list of ids like
`explanatory-note-for-revised-r-16:017`); the `ref:` prefix is added by the model.

The audit view replays these events chronologically.

---

## 5. Citations — four kinds (from `src/agent/schema.py`)

| Kind | Regex | Resolves against | Example |
|---|---|---|---|
| **package** | dotted path | the package dict via `resolve_package_path` | `transaction.amount_usd` |
| **tool** | `^[a-z_]+:call_[A-Za-z0-9_\-]+$` | the run's `tool_calls[].id` + name | `find_cycles:call_03` |
| **passage** | `^ref:[A-Za-z0-9_\-./:]+$` | the run's `passage_ids` (from `search_corpus`) | `ref:sar-faqs-october-2025:002` |
| **case** | `^case:row\d+$` | `case_ids` (never passed to the validator — always passes) | `case:row4938320` |

**How a citation resolves for the UI**:

1. `package` → walk the dotted path into the package JSON.
   - Success → the resolved sub-object as the evidence row.
   - Failure → **broken** (visible flag, no anchor).
2. `tool` → look in the corresponding JSONL log for `tool_calls[i].id == "<call_id>"`
   and `name == "<tool_name>"`. Success → the `result` object.
3. `passage` → look in the JSONL logs for a `search_corpus` `tool_call` whose
   `result.hits` contains `passage_id == "<ref>"`. Success → that hit dict.
4. `case` → as tool (the orchestrator's `case_ids` is always empty today).

The app implements a single function `resolve_citation(cite, package, log) → Resolution`
used by both the alert-detail view and the audit view.

---

## 6. Retrieval

`results/retrieval/summary.json` — query-level results for Stage 6 RAG:

```
{"method": "...", "recall_at_5": 0.75, "mrr": 0.7451,
 "ablations": {...}, "queries": [{query, hits: [...], gold: [...]}]}
```

`summary_bge.json` is the BGE experiment (null result); the Results page
shows both side by side.

---

## 7. Agent entry point (live mode)

`src/agent/orchestrator.py`:

```python
run_agent(alert_id, package, client, model, run_id=None, log_dir=None,
          max_calls=..., temperature=0.0) -> AgentResult
```

`AgentResult` fields: `draft, problems, flags, tool_calls, log_path, package`.

- No streaming callback exists. Live mode **tails the JSONL the agent writes**
  (`log_dir=results/agent/logs`), streaming new lines to the browser via SSE.
- **Key failure path**: a 401 → `status = "insufficient_evidence"`,
  `problems = ["model call failed: AuthenticationError: ... 401 ..."]`.
  It's written to a NEW draft file with a NEW run id — never overwrites.
  The UI shows the error banner and reverts the alert page to read-only.
- The key is read **only** in live mode via `agent.run._load_dotenv()`.
  Offline mode never imports `openai`.

---

## 8. Discrepancies between the brief and the real data

1. **"7/10 clean, 3 defective"** — the saved draft set is **14 drafts**:
   **1 clean**, **6 with defects**, **7 `insufficient_evidence`**
   (of which 3 are 401 key failures). The exact counts are computed at runtime
   on the Results page; no value is hardcoded.
2. **"3 real defects"** — there are more than 3 *problem entries* across drafts;
   the distinct bug *families* are three:
   `transaction.is_cross_bank` (missing field, correctly flagged),
   `case:row…` used where tool citation expected, and
   `ref:sar-…` passages never retrieved by `search_corpus` this run.
3. **"No positive-alert drafts exist (8 positives are packaged but undrafted)"** —
   **not true**. Two positive drafts exist:
   - `alert_3815014__run_…` (positive, status `insufficient_evidence`, 401)
   - `alert_4527071__run_…` (positive, status `insufficient_evidence`, 401)
   - `alert_6062005__run_…298635` (positive, 401)
   - `alert_6062005__run_…307952` (positive, **clean**) — the only drafted true positive
   The 8 positive packages are present; **only 6062005's second run produced a
   usable draft**. The other 5 positives are still undrafted. The Results page
   reports this real state and does not fabricate drafts.
4. **"Validator status per claim (pass/fail with reason)"** — the saved drafts
   store **only the problem entries**, not a per-claim pass flag. The app
   reconstructs per-claim status as:
   - a claim with **any** problem entry starting `<section>[<i>]:` → **fail**
   - otherwise → **pass with no recorded reason**
   and shows the overall `problems[]` list separately. There is no per-claim
   *reason* for a passing claim in the artifacts.
5. **"hybrid recall 0.75"** — confirmed at `results/retrieval/summary.json`,
   but the BGE experiment summary shows `hybrid 0.7647`; the plain hybrid is
   **0.75**. `bm25_only` at **0.7647** actually beats it. The Results page
   shows the *plain* number and includes the note `bm25_only > hybrid`.
6. **"recall@100"** — the requirement says to read it from artifacts. The
   number lives in `results/eda_output/model_ladder*.json` / `results/ladder/`
   summary, not in the package. It is read from those; if absent, the page
   renders `"—"` rather than a literal.

---

## 9. What the app must NOT do (per brief + verified)

- Never modify `results/evidence/**`. The package loader is read-only.
- Never build packages for alerts with no saved package.
- Never regenerate evidence on page load. Opening an alert always reads the
  saved JSON.
- Never print or log any key. Live mode passes the key to the OpenAI client
  only.
- Never overwrite a draft log or draft json. Live writes use
  `run_<UTC>-<PID>` run ids chosen by `orchestrator.new_run_id()`.
