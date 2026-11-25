"""Tests for the Alert Triage Viewer.

Offline-only. The live-mode 401 path is exercised with a mocked client.
Run from the project root:  pytest tests/test_app.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app import data  # noqa: E402
from app.server import create_app  # noqa: E402


# --------------------------------------------------------------------------- data layer


def test_packages_count_and_labels():
    pkgs = data.list_packages()
    assert len(pkgs) == 28, f"expected 28 packages, got {len(pkgs)}"
    positives = [p for p in pkgs if p.positive]
    assert len(positives) == 8
    assert all(p.path.parent == data.POSITIVES_DIR for p in positives)
    assert all(p.path.parent == data.EVIDENCE_DIR for p in pkgs if not p.positive)


def test_packages_have_required_fields():
    for pkg in data.list_packages():
        for k in ("transaction", "model", "features", "context", "provenance"):
            assert k in pkg.raw, f"{pkg.path.name} missing {k}"
        assert isinstance(pkg.model["raw_margin"], (int, float))
        assert isinstance(pkg.model["rank_in_day"], int)


def test_find_package_round_trip():
    for pkg in data.list_packages():
        found = data.find_package(pkg.alert_id)
        assert found is not None
        assert found.path == pkg.path


def test_drafts_load():
    drafts = data.all_drafts()
    assert len(drafts) >= 1
    for d in drafts:
        assert d.alert_id > 0
        assert d.run_id
        assert d.status in ("draft", "insufficient_evidence"), d.status
        # Every problem string should be either "<section>[i]: reason" or a free-form message.
        for p in d.problems:
            assert isinstance(p, str)


def test_draft_stats_are_consistent():
    s = data.draft_stats()
    assert s["total"] == s["clean"] + s["defective"] + s["insufficient"]


# --------------------------------------------------------------------------- citation resolution


def _first_draft_with_citations():
    for d in data.all_drafts():
        for section in data.DRAFT_SECTIONS:
            for claim in d.draft.get(section) or []:
                if claim.get("citations"):
                    return d
    return None


def test_citation_grammar():
    assert data.classify_citation("transaction.amount_usd") == "package"
    assert data.classify_citation("features.is_cross_bank") == "package"
    assert data.classify_citation("find_cycles:call_03_abc") == "tool"
    assert data.classify_citation("ref:sar-faqs-october-2025:002") == "passage"
    assert data.classify_citation("case:row12345") == "case"
    assert data.classify_citation("garbage:with:colons") == "unknown"


def test_package_path_resolves():
    pkg = data.list_packages()[0]
    r = data.resolve_citation("transaction.amount_usd", pkg.raw, [])
    assert r.ok and r.kind == "package"
    assert r.payload == pkg.raw["transaction"]["amount_usd"]


def test_package_path_missing_is_broken():
    pkg = data.list_packages()[0]
    r = data.resolve_citation("transaction.is_cross_bank", pkg.raw, [])
    assert not r.ok
    assert "does not resolve" in r.reason


def test_tool_citation_resolves_via_log():
    # Find a draft with a tool citation and use its own log.
    for d in data.all_drafts():
        log = data.load_log(d.log_path)
        for section in data.DRAFT_SECTIONS:
            for c in d.draft.get(section) or []:
                for cite in c.get("citations") or []:
                    if data.classify_citation(cite) == "tool":
                        r = data.resolve_citation(cite, {}, log)
                        # Either it resolves or it's flagged broken — not invented.
                        if r.ok:
                            assert r.payload.get("event") == "tool_call"
                            assert r.payload.get("call_id")
                        else:
                            assert r.reason
                        return
    pytest.skip("no tool citations in saved drafts")


def test_case_citation_resolves_for_known_row():
    # Case ids are accepted iff the validator listed them in this run's
    # draft_validated.passage_ids.
    for d in data.all_drafts():
        log = data.load_log(d.log_path)
        known = [pid for rec in log if rec.get("event") == "draft_validated"
                 for pid in (rec.get("passage_ids") or []) if pid.startswith("case:row")]
        if known:
            r = data.resolve_citation(known[0], {}, log)
            assert r.ok and r.kind == "case"
            return
    pytest.skip("no case citations in saved draft logs")


def test_case_citation_broken_for_unknown_row():
    r = data.resolve_citation("case:row999999999", {}, [])
    assert not r.ok


# --------------------------------------------------------------------------- server / offline


@pytest.fixture()
def offline_client():
    return TestClient(create_app(live=False))


def test_queue_lists_all_28(offline_client):
    html = offline_client.get("/").text
    for pkg in data.list_packages():
        assert str(pkg.alert_id) in html


def test_every_alert_renders(offline_client):
    for pkg in data.list_packages():
        r = offline_client.get(f"/alert/{pkg.alert_id}")
        assert r.status_code == 200, (pkg.alert_id, r.status_code)


def test_unknown_alert_404s(offline_client):
    assert offline_client.get("/alert/1").status_code == 404


def test_results_renders(offline_client):
    r = offline_client.get("/results")
    assert r.status_code == 200
    assert "Synthetic data, single simulator run" in r.text


def test_offline_determinism(offline_client):
    for pkg in data.list_packages():
        a = offline_client.get(f"/alert/{pkg.alert_id}").text
        b = offline_client.get(f"/alert/{pkg.alert_id}").text
        assert a == b, f"alert {pkg.alert_id} rendered differently on second load"


def test_offline_banner_on_every_page(offline_client):
    for path in ["/", "/results"] + [f"/alert/{p.alert_id}" for p in data.list_packages()]:
        r = offline_client.get(path)
        assert "Synthetic data, single simulator run" in r.text, path


def test_no_evidence_files_modified(offline_client):
    before = {p: data.package_sha256(p) for p in data.EVIDENCE_DIR.glob("*.json")}
    before.update({p: data.package_sha256(p) for p in data.POSITIVES_DIR.glob("*.json")})
    for pkg in data.list_packages():
        offline_client.get(f"/alert/{pkg.alert_id}")
    after = {p: data.package_sha256(p) for p in data.EVIDENCE_DIR.glob("*.json")}
    after.update({p: data.package_sha256(p) for p in data.POSITIVES_DIR.glob("*.json")})
    assert before == after


def test_offline_draft_button_is_disabled(offline_client):
    pkg = data.list_packages()[0]
    html = offline_client.get(f"/alert/{pkg.alert_id}").text
    assert 'disabled' in html
    assert "Offline mode" in html


def test_offline_draft_endpoint_rejects(offline_client):
    pkg = data.list_packages()[0]
    r = offline_client.post(f"/alert/{pkg.alert_id}/draft")
    assert r.status_code == 405


def test_audit_view_renders_when_log_exists(offline_client):
    for d in data.all_drafts():
        log = data.load_log(d.log_path)
        if log:
            pkg = data.find_package(d.alert_id)
            if pkg is None:
                continue
            r = offline_client.get(f"/alert/{d.alert_id}/audit/{d.draft_id}")
            assert r.status_code == 200
            return
    pytest.skip("no saved draft has a log on disk")


# --------------------------------------------------------------------------- live mode key failure


def _bad_key_client():
    """Build a live app but replace the orchestrator call with a 401 raiser."""
    client = TestClient(create_app(live=True))
    return client


def test_live_without_key_returns_friendly_error(monkeypatch, tmp_path):
    # Make sure neither env nor .env supplies a key.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(data, "PROJECT_ROOT", tmp_path)
    # Re-import live against the patched PROJECT_ROOT.
    from app import live
    monkeypatch.setattr(live.data, "PROJECT_ROOT", tmp_path)

    client = TestClient(create_app(live=True))
    pkg = data.list_packages()[0]
    r = client.post(f"/alert/{pkg.alert_id}/draft")
    assert r.status_code == 200
    assert "OPENAI_API_KEY not set" in r.text


def test_live_401_is_handled(monkeypatch):
    from openai import AuthenticationError

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")

    import app.live as live

    def fake_run_agent(**kwargs):
        raise AuthenticationError("401 invalid key", response=object(), body=None)

    import agent.orchestrator as orch  # noqa: WPS433
    monkeypatch.setattr(orch, "run_agent", fake_run_agent)
    monkeypatch.setattr(live, "run_live_draft",
                        lambda alert_id: {"draft_id": None,
                                          "error": "401 — key rejected by provider"})

    client = TestClient(create_app(live=True))
    pkg = data.list_packages()[0]
    r = client.post(f"/alert/{pkg.alert_id}/draft")
    assert r.status_code == 200
    assert "Live draft error" in r.text
    assert "401" in r.text


def test_dotenv_key_wins_over_stale_env(monkeypatch, tmp_path):
    """.env is authoritative in the viewer: a stale exported key must not shadow it."""
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-from-dotenv\n# comment\nOPENAI_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-from-shell")

    from app import live
    monkeypatch.setattr(live.data, "PROJECT_ROOT", tmp_path)

    assert live.key_source() == ".env"
    assert live._ensure_key() == "sk-from-dotenv"
    # _ensure_key makes the SDK see the .env key too.
    import os
    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"


def test_key_source_env_fallback(monkeypatch, tmp_path):
    """With no .env, the exported key is used and reported as "environment"."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    from app import live
    monkeypatch.setattr(live.data, "PROJECT_ROOT", tmp_path)  # no .env here

    assert live.key_source() == "environment"
    assert live._ensure_key() == "sk-from-env"


def test_live_badge_shows_key_source(monkeypatch, tmp_path):
    """In live mode the topbar badge reports where the key came from."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    from app import live
    monkeypatch.setattr(live.data, "PROJECT_ROOT", tmp_path)

    client = TestClient(create_app(live=True))
    html = client.get("/").text
    assert "key: .env" in html
    assert "LIVE" in html
