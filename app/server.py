"""FastAPI app factory and route handlers.

Offline is the default. Live mode adds the Draft endpoint (POST /alert/{id}/draft)
and a small streaming endpoint. Nothing in offline mode imports the model
client, the agent orchestrator, or the network stack beyond the web framework.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import data

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

BANNER = "Synthetic data, single simulator run."


# --------------------------------------------------------------------------- app factory


def create_app(*, live: bool = False, token: str | None = None) -> FastAPI:
    app = FastAPI(title="Alert Triage Viewer", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # Global template context: every page shows the banner + mode badge.
    templates.env.globals["BANNER"] = BANNER
    templates.env.globals["MODE"] = "LIVE" if live else "OFFLINE"
    templates.env.globals["key_source"] = _key_source if live else (lambda: None)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.state.live = live
    app.state.token = token
    app.state.templates = templates
    app.state.key_present = _key_present() if live else False
    app.state.key_source = _key_source() if live else None
    # Live-mode accessors are installed lazily so importing this module in
    # offline mode does not import the orchestrator or the model client.
    if live:
        from . import live as _live
        app.state.live_module = _live
    else:
        app.state.live_module = None

    # --------------------------------------------------------------------- token gate
    @app.middleware("http")
    async def _token_gate(request: Request, call_next: Callable):
        required = app.state.token
        if required:
            supplied = (
                request.headers.get("x-viewer-token")
                or request.query_params.get("token")
            )
            if supplied != required:
                return JSONResponse({"error": "token required"}, status_code=401)
        return await call_next(request)

    # --------------------------------------------------------------------- routes
    @app.get("/", response_class=HTMLResponse)
    def alert_queue(request: Request):
        pkgs = data.list_packages()
        rows = []
        for p in pkgs:
            drafts = data.drafts_for(p.alert_id)
            rows.append({
                "pkg": p,
                "drafts": drafts,
                "draft_summary": _draft_summary(drafts),
            })
        return app.state.templates.TemplateResponse(
            request,
            "queue.html",
            {
                "rows": rows,
                "live": app.state.live,
                "key_present": app.state.key_present,
            },
        )

    @app.get("/alert/{alert_id}", response_class=HTMLResponse)
    def alert_detail(request: Request, alert_id: int, draft: str | None = None):
        pkg = data.find_package(alert_id)
        if pkg is None:
            raise HTTPException(status_code=404, detail="alert not in saved pack set")
        drafts = data.drafts_for(alert_id)
        selected = None
        if draft:
            selected = data.find_draft(alert_id, draft)
            if selected is None:
                raise HTTPException(status_code=404, detail="draft not found for this alert")
        return app.state.templates.TemplateResponse(
            request,
            "detail.html",
            {
                "pkg": pkg,
                "drafts": drafts,
                "selected": selected,
                "view": _detail_view(pkg, selected),
                "live": app.state.live,
                "key_present": app.state.key_present,
            },
        )

    @app.get("/alert/{alert_id}/audit/{draft_id}", response_class=HTMLResponse)
    def audit_replay(request: Request, alert_id: int, draft_id: str):
        pkg = data.find_package(alert_id)
        if pkg is None:
            raise HTTPException(status_code=404, detail="alert not in saved pack set")
        dr = data.find_draft(alert_id, draft_id)
        if dr is None:
            raise HTTPException(status_code=404, detail="draft not found for this alert")
        log = data.load_log(dr.log_path) if dr.log_path else data.load_log(
            data.LOGS_DIR / (dr.path.stem + ".jsonl")
        )
        return app.state.templates.TemplateResponse(
            request,
            "audit.html",
            {
                "pkg": pkg,
                "dr": dr,
                "events": _audit_events(log),
                "live": app.state.live,
            },
        )

    @app.get("/results", response_class=HTMLResponse)
    def results(request: Request):
        return app.state.templates.TemplateResponse(
            request,
            "results.html",
            {
                "stats": data.draft_stats(),
                "retrieval": data.retrieval_summary(),
                "retrieval_bge": data.retrieval_bge_summary(),
                "calibration": data.calibration_summary(),
                "ladder": data.ladder_rung2(),
                "ladder_r1": data.ladder_rung1(),
                "drafts": data.all_drafts(),
                "packages": data.list_packages(),
                "live": app.state.live,
            },
        )

    # --------------------------------------------------------------------- live routes
    @app.post("/alert/{alert_id}/draft")
    def start_draft(request: Request, alert_id: int):
        if not app.state.live:
            raise HTTPException(status_code=405,
                                detail="offline mode: drafting needs --live and a working key")
        if app.state.live_module is None:
            raise HTTPException(status_code=500, detail="live module not loaded")
        pkg = data.find_package(alert_id)
        if pkg is None:
            raise HTTPException(status_code=404, detail="alert not in saved pack set")
        result = app.state.live_module.run_live_draft(alert_id)
        if result.get("draft_id"):
            return RedirectResponse(url=f"/alert/{alert_id}?draft={result['draft_id']}",
                                    status_code=303)
        return app.state.templates.TemplateResponse(
            request,
            "detail.html",
            {
                "pkg": pkg,
                "drafts": data.drafts_for(alert_id),
                "selected": None,
                "view": _detail_view(pkg, None),
                "live": True,
                "key_present": app.state.key_present,
                "live_error": result.get("error", "live draft failed"),
            },
            status_code=200,
        )

    return app


# --------------------------------------------------------------------------- helpers


def _key_present() -> bool:
    return _key_source() is not None


def _key_source() -> str | None:
    # In live mode the .env key is authoritative; import the live module for
    # the single source of truth on precedence.
    from . import live as _live
    return _live.key_source()


def _draft_summary(drafts: list[data.DraftFile]) -> dict[str, Any]:
    if not drafts:
        return {"status": "no draft", "tone": "muted", "has_content": False}
    if any(d.clean for d in drafts):
        n = sum(1 for d in drafts if d.clean)
        return {"status": f"clean ({n})", "tone": "ok", "has_content": True}
    if all(d.status != "draft" for d in drafts):
        return {"status": "no usable draft", "tone": "fail", "has_content": False}
    n = sum(1 for d in drafts if d.problems)
    return {"status": f"{n} defects", "tone": "warn", "has_content": True}


def _detail_view(pkg: data.PackageInfo, dr: data.DraftFile | None) -> dict[str, Any]:
    """Everything the template needs for the selected draft (or the package alone)."""
    view: dict[str, Any] = {
        "resolutions": [],          # list[Resolution] for anchors on the page
        "sections": [],             # [{name, claims: [{text, status, reasons, cites:[{cite, res}]}]}]
        "problem_total": 0,
        "clean_count": 0,
    }
    if dr is None:
        return view

    log = data.load_log(dr.log_path) if dr.log_path else []
    pm = data.problem_map(dr)
    seen: dict[str, data.Resolution] = {}

    for section in data.DRAFT_SECTIONS:
        claims = dr.draft.get(section) or []
        out_claims = []
        for i, c in enumerate(claims):
            reasons = pm.get((section, i), [])
            cites = []
            for cite in c.get("citations") or []:
                r = seen.get(cite) or data.resolve_citation(cite, pkg.raw, log)
                seen[cite] = r
                cites.append({"cite": cite, "res": r})
            out_claims.append({
                "text": c.get("text", ""),
                "status": "fail" if reasons else "pass",
                "reasons": reasons,
                "cites": cites,
            })
        view["sections"].append({"name": section.replace("_", " "), "raw_name": section,
                                 "claims": out_claims})

    view["resolutions"] = list(seen.values())
    view["problem_total"] = len(dr.problems)
    view["clean_count"] = sum(1 for s in view["sections"] for c in s["claims"] if c["status"] == "pass")
    return view


def _audit_events(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in log:
        t = rec.get("event", "?")
        summary = ""
        if t == "run_start":
            summary = (f"alert {rec.get('alert_id')} · model {rec.get('model')} · "
                       f"max_calls {rec.get('max_calls')}")
        elif t == "model_message":
            calls = rec.get("tool_calls") or []
            summary = f"{len(calls)} tool call(s) · {float(rec.get('latency_s') or 0):.2f}s"
        elif t == "tool_call":
            summary = f"{rec.get('name')}({_short(rec.get('arguments'))})"
            if not rec.get("ok", True):
                summary += " → call failed"
        elif t == "draft_parsed":
            summary = "model output parsed into a draft"
        elif t == "draft_validated":
            ps = rec.get("problems") or []
            summary = (f"status={rec.get('status')} · {len(ps)} problem(s) · "
                       f"{len(rec.get('tool_call_ids') or [])} tool ids · "
                       f"{len(rec.get('passage_ids') or [])} passage ids")
        elif t == "run_end":
            tc = rec.get("tool_calls")
            n = tc if isinstance(tc, int) else len(tc or [])
            summary = f"{n} total call(s) · ok={rec.get('ok')}"
        out.append({"seq": rec.get("seq"), "ts": rec.get("ts"), "type": t,
                    "summary": summary, "raw": rec})
    return out


def _short(x: Any) -> str:
    s = str(x) if x is not None else ""
    return s if len(s) <= 120 else s[:117] + "…"


def _ladder_numbers() -> dict[str, Any]:
    """The honest headline numbers, read from saved artifact files.

    Where a number lives only in a markdown note (e.g. lift 59.6), we look for a
    JSON next to it. If none exists, we return None so the template renders
    "—" rather than a hardcoded value.
    """
    out: dict[str, Any] = {}
    p = data.LADDER_DIR / "summary.json"
    if p.is_file():
        try:
            import json as _json
            out.update(_json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out
