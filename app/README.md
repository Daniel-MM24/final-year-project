# Alert Triage Viewer

A local read-first UI over the saved AML pipeline artifacts (Stage 4 scores →
Stage 5 evidence packages → Stage 7 agent drafts + validator output). Two modes,
one codebase:

- **Offline** (default): reads saved files only. No network, no model client, no key.
- **Live** (`--live`): the *Draft* button calls the real agent orchestrator for the
  currently-open alert, writes a **new** draft JSON and JSONL log under
  `results/agent/`, and re-validates. Existing drafts and packages are never touched.

Everything the app shows is **synthetic data from a single simulator run**.
The app is for human review; it does not make decisions and is not
production-ready.

## Requirements

- Python 3.10+ (use the project's conda env: `conda activate learn_env`)
- Install once: `pip install fastapi uvicorn jinja2 httpx pytest`
- For live mode only: `openai` (already installed for the agent)

## Run offline (default)

```bash
# from the project root
python -m app
```

Opens on <http://127.0.0.1:8000/>. No `.env` needed, no key needed, no network
calls made. This is the mode to demo on a locked-down machine.

## Run live

```bash
python -m app --live
```

`--live` enables the *Draft* button on every alert page. Clicking it:

1. Reads `OPENAI_API_KEY` from `.env` if present, else from the environment
   (only this endpoint reads it — the key is never logged or shown).
2. Calls `agent.orchestrator.run_agent(...)` on the saved package for that alert.
3. Writes a NEW draft under `results/agent/drafts/` and a NEW log under
   `results/agent/logs/`, with a fresh `run_<UTC>-<PID>` id.
4. Redirects to the alert page filtered to the new draft.

**Key precedence: `.env` is authoritative in the viewer.** A stale
`OPENAI_API_KEY` exported in a long-lived shell will *not* shadow the key the
project maintains — that shell variable is exactly how the earlier confusing
401s happened. (The batch agent in `src/agent/run.py` keeps the opposite
precedence, env first; only the viewer inverts it.) The topbar badge in live
mode shows which source won: `key: .env` or `key: environment`.

If the key is missing or rejected (401), the button shows a clear error banner
and the page stays usable.

## Bind address

By default binds `127.0.0.1`. Binding wider requires an explicit token:

```bash
python -m app --host 0.0.0.0 --token my-demo-token
```

Then open with `?token=my-demo-token` or send `X-Viewer-Token: my-demo-token`.
No HTTPS is set up; this is a speed bump, not an auth system. Deployment is
out of scope.

## Routes

| Route | What it shows |
|---|---|
| `/` | alert queue, 28 packages, filters by label / draft status |
| `/alert/{id}` | evidence package, score + rank, drafts with citation linking, validator status |
| `/alert/{id}/audit/{draft_id}` | JSONL tool-call replay for one draft |
| `/results` | honest headline numbers read from artifacts |

## Tests

```bash
pytest tests/ -q
```

Covers data loaders, citation resolution (package / tool / passage / case),
offline determinism (two loads → identical HTML), the offline Draft endpoint's
405 response, and the live-mode 401 path with a mocked OpenAI client.

## What this app does **not** do

- It does not regenerate, modify, or overwrite anything under `results/evidence/**`.
- It does not build packages for alerts that are not in the saved 28.
- It does not call a model in offline mode — the `openai` import is inside `app/live.py`,
  which is only imported when `--live` is passed.
- It does not implement analyst accept/reject/assign, a database, or deployment.

## Known limitations / discrepancies vs the original Stage brief

See [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md) for the discovered schemas and the list of
discrepancies between the original brief and the real artifact set (draft counts,
positive-draft existence, validator wording quirks).
