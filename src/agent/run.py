"""CLI: draft one case (or several) from an evidence package.

    python -m agent.run --alert 12345
    python -m agent.run --package results/evidence/package_00123.json
    python -m agent.run --sample 10 --concurrency 2

Reads the model, endpoint and key from OPENAI_BASE_URL / OPENAI_API_KEY /
OPENAI_MODEL. Writes the draft JSON next to its log and prints a one-line summary per
run. Exit code is 1 if any run failed validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures as _futures
import json
import os
import sys
from pathlib import Path

from .log import DEFAULT_DIR
from .orchestrator import run_agent, replay_check
from .schema import CaseDraft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = PROJECT_ROOT / "results" / "evidence"
DRAFT_DIR = PROJECT_ROOT / "results" / "agent" / "drafts"


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE file if the project has one.

    Existing environment variables win, so an exported key overrides the file.
    Blank lines and '#' comments are ignored; surrounding quotes are stripped.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_package(alert_id: str) -> Path:
    matches = sorted(EVIDENCE_DIR.glob(f"*_{alert_id}.json"))
    if not matches:
        for p in sorted(EVIDENCE_DIR.glob("*.json")):
            pkg = json.loads(p.read_text(encoding="utf-8"))
            if str(pkg.get("transaction", {}).get("row")) == alert_id:
                return p
        raise SystemExit(f"no evidence package for alert {alert_id} in {EVIDENCE_DIR}")
    return matches[0]


def _load_package(path: Path) -> tuple[str, dict]:
    pkg = json.loads(path.read_text(encoding="utf-8"))
    row = pkg.get("transaction", {}).get("row")
    return str(row if row is not None else path.stem), pkg


def _is_alerted(pkg: dict) -> bool:
    """An alert is a package whose rank-in-day falls inside the alert budget."""
    m = pkg.get("model", {})
    if isinstance(m.get("alert"), bool):
        return m["alert"]
    rank, budget = m.get("rank_in_day"), m.get("budget")
    return rank is not None and budget is not None and int(rank) <= int(budget)


def _sample_alerts(n: int) -> list[Path]:
    """Deterministic mixed sample: half alerted, half not, sorted by package name."""
    pkgs = []
    for p in sorted(EVIDENCE_DIR.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            pkgs.append((json.loads(p.read_text(encoding="utf-8")), p))
        except json.JSONDecodeError:
            continue
    alerted = [p for pkg, p in pkgs if _is_alerted(pkg)]
    quiet = [p for pkg, p in pkgs if not _is_alerted(pkg)]
    want_alert = n // 2
    chosen = (alerted[:want_alert] + quiet[: n - want_alert]
              + alerted[want_alert:] + quiet[n - want_alert:])
    return chosen[:n]


def _client():
    _load_dotenv()
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed in this environment")
    key, base, model = (os.environ.get("OPENAI_API_KEY"),
                        os.environ.get("OPENAI_BASE_URL"),
                        os.environ.get("OPENAI_MODEL"))
    if not key or not base or not model:
        raise SystemExit("OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL must all be set")
    return OpenAI(base_url=base, api_key=key), model


def _write_draft(result, alert_id: str) -> Path:
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    out = DRAFT_DIR / f"alert_{alert_id}__run_{result.draft.run_id}.json"
    payload = {
        "draft": result.draft.as_dict(),
        "problems": result.problems,
        "tool_calls": result.tool_calls,
        "log_path": str(result.log_path.relative_to(PROJECT_ROOT)),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return out


def _one(path: Path, client, model: str) -> tuple[str, str, object]:
    alert_id, package = _load_package(path)
    print(f"[run] alert {alert_id} ({path.name})", flush=True)
    res = run_agent(alert_id=alert_id, package=package, client=client, model=model)
    out = _write_draft(res, alert_id)
    status = "OK" if res.ok else "FAIL"
    n_claims = sum(len(getattr(res.draft, s)) for s in
                   ("summary", "observed_facts", "hypothesised_typology",
                    "supporting_evidence", "counter_evidence", "recommended_next_steps"))
    print(f"[done] alert {alert_id}: {status} status={res.draft.status} "
          f"claims={n_claims} tools={res.tool_calls} flags={len(res.flags)} -> {out.name}",
          flush=True)
    if res.problems:
        for p in res.problems:
            print(f"       ! {p}", flush=True)
    return alert_id, status, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--alert", help="alert id (transaction row)")
    g.add_argument("--package", help="path to an evidence package JSON")
    g.add_argument("--sample", type=int, metavar="N", help="run N mixed alerts")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel runs for --sample (each holds its own log)")
    ap.add_argument("--replay-check", action="store_true",
                    help="after running, verify every produced log is replayable")
    args = ap.parse_args(argv)

    client, model = _client()

    if args.alert:
        paths = [_find_package(args.alert)]
    elif args.package:
        paths = [Path(args.package)]
    else:
        paths = _sample_alerts(args.sample)

    failed = 0
    if args.sample and len(paths) > 1 and args.concurrency > 1:
        with _futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(_one, p, client, model) for p in paths]
            for fut in futs:
                _, status, _ = fut.result()
                failed += status != "OK"
    else:
        for p in paths:
            _, status, _ = _one(p, client, model)
            failed += status != "OK"

    if args.replay_check:
        print("[replay-check] scanning logs from this run", flush=True)
        for p in paths:
            alert_id, package = _load_package(p)
            logs = sorted(DEFAULT_DIR.glob(f"alert_{alert_id}__run_*.jsonl"))
            if not logs:
                continue
            probs = replay_check(logs[-1])
            print(f"  {logs[-1].name}: {'clean' if not probs else probs}", flush=True)
            failed += bool(probs)

    print(f"[summary] {len(paths)} run(s), {failed} problem(s)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
