"""Live-mode drafting. Imported only when ``create_app(live=True)``.

Public surface:
  * :func:`run_live_draft(alert_id)` — runs the real orchestrator, catches
    auth/HTTP errors, writes a NEW draft + log under ``results/agent/`` with a
    fresh run id, and returns ``{draft_id, status, error}``.

The key is loaded from ``.env`` (via the agent's own ``_load_dotenv`` helper)
*only* here. The key never leaves this module — it is handed to the OpenAI
client and never logged or returned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from . import data

# Ensure ``src`` is importable so ``agent.*`` resolves without PYTHONPATH.
_SRC = data.PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _read_dotenv_key() -> str | None:
    """Return the OPENAI_API_KEY value from the project .env, or None."""
    env = data.PROJECT_ROOT / ".env"
    if not env.is_file():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if k.strip() == "OPENAI_API_KEY" and v:
            return v
    return None


def key_source() -> str | None:
    """Where live mode would take its key from: ".env", "environment", or None.

    The project .env is authoritative: a stale OPENAI_API_KEY exported in some
    long-lived shell must never shadow the key the project maintains. This is
    deliberately the OPPOSITE of agent.run._load_dotenv's precedence, which is
    kept for the batch agent.
    """
    if _read_dotenv_key():
        return ".env"
    if os.environ.get("OPENAI_API_KEY"):
        return "environment"
    return None


def _ensure_key() -> str:
    key = _read_dotenv_key()
    if key:
        os.environ["OPENAI_API_KEY"] = key  # make the SDK see the .env key
        return key
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY not set (env or .env)")


def _model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "deepseek-chat")


def run_live_draft(alert_id: int) -> dict[str, Any]:
    pkg = data.find_package(alert_id)
    if pkg is None:
        return {"draft_id": None, "error": "alert not in saved pack set"}

    try:
        key = _ensure_key()
    except RuntimeError as e:
        return {"draft_id": None, "error": str(e)}

    try:
        from openai import AuthenticationError, PermissionDeniedError
        from openai import OpenAI
    except Exception as e:  # pragma: no cover — dependency missing
        return {"draft_id": None, "error": f"openai client unavailable: {e}"}

    try:
        from agent.orchestrator import run_agent
    except Exception as e:  # pragma: no cover — agent missing
        return {"draft_id": None, "error": f"agent orchestrator unavailable: {e}"}
    try:
        from agent.run import _write_draft  # reuse the project's writer
    except Exception:
        _write_draft = None  # fall back to manual write below

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)

    try:
        result = run_agent(  # type: ignore[call-arg]
            alert_id=str(alert_id),
            package=pkg.raw,
            client=client,
            model=_model_name(),
            log_dir=data.LOGS_DIR,
        )
    except AuthenticationError as e:
        return {"draft_id": None, "error": f"401 — key rejected by provider: {e}"}
    except PermissionDeniedError as e:
        return {"draft_id": None, "error": f"permission denied: {e}"}
    except Exception as e:  # network/other
        return {"draft_id": None, "error": f"agent run failed: {type(e).__name__}: {e}"}

    # run_agent does NOT persist the draft; agent.run._write_draft does.
    try:
        draft_path = Path(_write_draft(result, str(alert_id)))
    except Exception as e:
        return {"draft_id": None, "error": f"draft written but not persisted: {type(e).__name__}: {e}"}

    # Our data layer cached the drafts index at import time. Bust it so the
    # redirect target can be found by the next request.
    data._drafts_index.cache_clear()

    status = getattr(result.draft, "status", None) if not isinstance(result.draft, dict) else result.draft.get("status")
    return {"draft_id": draft_path.stem, "status": status, "error": None}
