"""Read-only data-access layer for the Alert Triage Viewer.

This is the ONLY module in ``app/`` that knows filesystem paths. Everything
else (routes, templates, tests) goes through the typed loaders here.

Hard rules from the brief (enforced here, not just documented):
  * ``results/evidence/**`` is opened for reading only. No writes.
  * Drafts and run logs under ``results/agent/`` are read-only in this module.
    Live-mode write paths live in ``app/live.py``.
  * All package lookup goes through :func:`list_packages` /
    :func:`find_package`. There is no "build on demand" path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "results" / "evidence"
POSITIVES_DIR = EVIDENCE_DIR / "positives"
DRAFTS_DIR = PROJECT_ROOT / "results" / "agent" / "drafts"
LOGS_DIR = PROJECT_ROOT / "results" / "agent" / "logs"
LADDER_DIR = PROJECT_ROOT / "results" / "ladder"
RETRIEVAL_DIR = PROJECT_ROOT / "results" / "retrieval"
CALIBRATION_DIR = PROJECT_ROOT / "results" / "calibration"

SCORES_PARQUET = LADDER_DIR / "rung2_lgbm_plus_hop_scores.parquet"
RETRIEVAL_SUMMARY = RETRIEVAL_DIR / "summary.json"
RETRIEVAL_BGE_SUMMARY = RETRIEVAL_DIR / "summary_bge.json"

# Citation grammars — mirrors src/agent/schema.py exactly.
TOOL_CITE_RE = re.compile(r"^[a-z_]+:call_[A-Za-z0-9_\-]+$")
PASSAGE_CITE_RE = re.compile(r"^ref:[A-Za-z0-9_\-./:]+$")
CASE_CITE_RE = re.compile(r"^case:row\d+$")

# Each package file: package_<NN>_row<row>.json
_PACKAGE_NAME_RE = re.compile(r"^package_(\d+)_row(\d+)\.json$")


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class PackageInfo:
    """One evidence package as seen by the UI."""

    alert_id: int
    positive: bool
    path: Path
    index: int  # the NN in the filename
    transaction: dict[str, Any]
    model: dict[str, Any]
    features: dict[str, Any]
    context: dict[str, Any]
    provenance: dict[str, Any]
    raw: dict[str, Any]

    @property
    def score(self) -> float:
        return float(self.model.get("raw_margin", 0.0))

    @property
    def rank(self) -> int:
        return int(self.model.get("rank_in_day", 0))

    @property
    def alerted(self) -> bool:
        return bool(self.model.get("alert"))

    @property
    def golden_label(self) -> int:
        return int(self.transaction.get("golden_label", 0))


@dataclass(frozen=True)
class DraftFile:
    """One saved agent run (draft json)."""

    path: Path
    alert_id: int
    run_id: str
    status: str
    draft: dict[str, Any]
    problems: list[str]
    tool_calls: int | list  # SAVED DRAFTS STORE AN INT COUNT, not a list of ids
    log_path: str | None
    flags: list[dict[str, Any]]

    @property
    def clean(self) -> bool:
        """Clean means a real draft with zero validator problems."""
        return self.status == "draft" and not self.problems

    @property
    def has_draft_content(self) -> bool:
        """True iff at least one section contains at least one claim."""
        for section in DRAFT_SECTIONS:
            if self.draft.get(section):
                return True
        return False

    @property
    def draft_id(self) -> str:
        """Stable id used in URLs — the draft filename stem."""
        return self.path.stem


DRAFT_SECTIONS = (
    "summary",
    "observed_facts",
    "hypothesised_typology",
    "supporting_evidence",
    "counter_evidence",
    "recommended_next_steps",
)


@dataclass(frozen=True)
class Resolution:
    """Result of resolving one citation string."""

    citation: str
    kind: str  # "package" | "tool" | "passage" | "case" | "unknown"
    ok: bool
    payload: Any = None
    reason: str | None = None

    @property
    def anchor(self) -> str | None:
        if not self.ok:
            return None
        return _anchor_for(self)


def _anchor_for(r: Resolution) -> str:
    if r.kind == "package":
        return "ev-pkg-" + re.sub(r"[^a-zA-Z0-9]+", "-", r.citation)
    if r.kind == "tool":
        call_id = r.citation.split(":", 1)[1]
        return "ev-tool-" + re.sub(r"[^a-zA-Z0-9]+", "-", call_id)
    if r.kind == "passage":
        return "ev-psg-" + re.sub(r"[^a-zA-Z0-9]+", "-", r.citation.split(":", 1)[1])
    if r.kind == "case":
        return "ev-case-" + r.citation.split(":", 1)[1]
    return ""


# --------------------------------------------------------------------------- packages


def _iter_package_paths() -> Iterable[tuple[Path, bool]]:
    """Yield (path, is_positive) for every saved package. Deterministic order."""
    for d, positive in ((EVIDENCE_DIR, False), (POSITIVES_DIR, True)):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and _PACKAGE_NAME_RE.match(p.name):
                yield p, positive


def _load_package(path: Path, positive: bool) -> PackageInfo:
    raw = json.loads(path.read_text(encoding="utf-8"))
    m = _PACKAGE_NAME_RE.match(path.name)
    assert m is not None
    alert_id = int(m.group(2))
    return PackageInfo(
        alert_id=alert_id,
        positive=positive,
        path=path,
        index=int(m.group(1)),
        transaction=raw["transaction"],
        model=raw["model"],
        features=raw["features"],
        context=raw["context"],
        provenance=raw["provenance"],
        raw=raw,
    )


@lru_cache(maxsize=1)
def _packages_index() -> dict[int, PackageInfo]:
    """All 28 saved packages, keyed by alert id. Cache is invalidated by process."""
    out: dict[int, PackageInfo] = {}
    for path, pos in _iter_package_paths():
        pkg = _load_package(path, pos)
        if pkg.alert_id in out:
            # Never happen with the verified set, but guard against duplicates.
            raise RuntimeError(f"duplicate alert_id {pkg.alert_id}: {path}")
        out[pkg.alert_id] = pkg
    return out


def list_packages() -> list[PackageInfo]:
    """All saved packages in a stable order: negatives first, then positives."""
    pkgs = list(_packages_index().values())
    pkgs.sort(key=lambda p: (p.positive, p.index))
    return pkgs


def find_package(alert_id: int) -> PackageInfo | None:
    return _packages_index().get(int(alert_id))


# --------------------------------------------------------------------------- drafts


def _iter_draft_paths() -> Iterable[Path]:
    if not DRAFTS_DIR.is_dir():
        return
    for p in sorted(DRAFTS_DIR.iterdir()):
        if p.is_file() and p.name.startswith("alert_") and p.suffix == ".json":
            yield p


_DRAFT_NAME_RE = re.compile(r"^alert_(\d+)__run_(.+)\.json$")


def _load_draft(path: Path) -> DraftFile:
    m = _DRAFT_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognised draft filename: {path.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    draft = raw.get("draft") or {}
    return DraftFile(
        path=path,
        alert_id=int(m.group(1)),
        run_id=str(draft.get("run_id") or m.group(2)),
        status=str(draft.get("status", "")),
        draft=draft,
        problems=list(raw.get("problems") or []),
        tool_calls=raw.get("tool_calls") or 0,  # int count in current artifacts
        log_path=raw.get("log_path"),
        flags=list(draft.get("flags") or []),
    )


@lru_cache(maxsize=1)
def _drafts_index() -> dict[int, list[DraftFile]]:
    out: dict[int, list[DraftFile]] = {}
    for path in _iter_draft_paths():
        d = _load_draft(path)
        out.setdefault(d.alert_id, []).append(d)
    # Stable order: oldest run first within each alert.
    for lst in out.values():
        lst.sort(key=lambda d: d.run_id)
    return out


def drafts_for(alert_id: int) -> list[DraftFile]:
    return list(_drafts_index().get(int(alert_id), []))


def all_drafts() -> list[DraftFile]:
    out: list[DraftFile] = []
    for lst in _drafts_index().values():
        out.extend(lst)
    return out


def find_draft(alert_id: int, draft_id: str) -> DraftFile | None:
    for d in drafts_for(alert_id):
        if d.draft_id == draft_id:
            return d
    return None


# --------------------------------------------------------------------------- logs


def load_log(log_path: str | Path | None) -> list[dict[str, Any]]:
    """Read a JSONL run log, or [] if the path is missing/blank."""
    if not log_path:
        return []
    p = Path(log_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return []
    lines: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    lines.sort(key=lambda r: int(r.get("seq", 0)))
    return lines


def _log_path_for(draft: DraftFile) -> Path | None:
    if draft.log_path:
        p = Path(draft.log_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    # Fallback: same stem under results/agent/logs
    p = LOGS_DIR / (draft.path.stem + ".jsonl")
    return p if p.is_file() else None


# --------------------------------------------------------------------------- citation resolution


def _walk_package(package: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Walk a dotted path into the package with [i] indexing support."""
    cur: Any = package
    for tok in re.split(r"\.(?![^\[]*\])", path):
        m = re.match(r"^([^\[\]]+)((?:\[\d+\])+)$", tok)
        if m:
            key, idxs = m.group(1), re.findall(r"\[(\d+)\]", m.group(2))
            if not isinstance(cur, dict) or key not in cur:
                return False, None
            cur = cur[key]
            for i in idxs:
                if not isinstance(cur, list):
                    return False, None
                n = int(i)
                if n >= len(cur):
                    return False, None
                cur = cur[n]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return False, None
            cur = cur[tok]
    return True, cur


def classify_citation(cite: str) -> str:
    if TOOL_CITE_RE.match(cite):
        return "tool"
    if PASSAGE_CITE_RE.match(cite):
        return "passage"
    if CASE_CITE_RE.match(cite):
        return "case"
    # Anything with a ':' that we didn't recognise is "unknown" so it shows
    # visibly as broken rather than silently rerouted to package-looking.
    if ":" in cite:
        return "unknown"
    return "package"


def _tool_result_index(log: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in log:
        if rec.get("event") == "tool_call":
            cid = rec.get("call_id")
            if cid:
                out[cid] = rec
    return out


def _known_ids(log: list[dict[str, Any]]) -> set[str]:
    """Ids the validator itself accepts: ``draft_validated.passage_ids``.

    That list holds both corpus passages (``ref:...``) and packaged cases
    (``case:row...``) and is exactly what the saved ``problems`` were computed
    against — resolving against it keeps the UI consistent with the validator.
    """
    out: set[str] = set()
    for rec in log:
        if rec.get("event") == "draft_validated":
            for pid in rec.get("passage_ids") or []:
                out.add(str(pid))
    return out


def _passage_detail(log: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Best-effort payload for a passage id: the search_corpus hit that produced it."""
    out: dict[str, dict[str, Any]] = {}
    for rec in log:
        if rec.get("event") != "tool_call" or rec.get("name") != "search_corpus":
            continue
        for hit in (rec.get("result") or {}).get("hits") or []:
            pid = hit.get("passage_id")
            if pid:
                out.setdefault("ref:" + str(pid), hit)
    return out


def resolve_citation(
    cite: str,
    package: dict[str, Any],
    log: list[dict[str, Any]],
) -> Resolution:
    kind = classify_citation(cite)
    if kind == "package":
        ok, val = _walk_package(package, cite)
        if ok:
            return Resolution(cite, "package", True, val)
        return Resolution(cite, "package", False, None, "path does not resolve in package")
    if kind == "tool":
        name, _, call_id = cite.partition(":")
        rec = _tool_result_index(log).get(call_id)
        if rec is None:
            return Resolution(cite, "tool", False, None, "no matching tool call in run log")
        if rec.get("name") != name:
            return Resolution(cite, "tool", False, None,
                              f"tool call id matched but name is {rec.get('name')!r}")
        return Resolution(cite, "tool", True, rec)
    if kind == "passage":
        known = _known_ids(log)
        detail = _passage_detail(log).get(cite)
        if cite in known:
            return Resolution(cite, "passage", True, detail or {"passage_id": cite})
        return Resolution(cite, "passage", False, None,
                          "passage not in this run's validator passage_ids")
    if kind == "case":
        # Case cites point at another packaged alert. The validator accepted
        # them iff they appear in draft_validated.passage_ids.
        if cite in _known_ids(log):
            m = re.match(r"^case:row(\d+)$", cite)
            return Resolution(cite, "case", True, {"row": int(m.group(1)) if m else None})
        return Resolution(cite, "case", False, None, "case not in this run's validator ids")
    return Resolution(cite, "unknown", False, None, "unrecognised citation syntax")


# --------------------------------------------------------------------------- problems → per-claim


_PROBLEM_RE = re.compile(r"^([a-z_]+)\[(\d+)\]:\s*(.+)$")


def problem_map(draft: DraftFile) -> dict[tuple[str, int], list[str]]:
    """Map (section, claim_index) -> list of validator reasons."""
    out: dict[tuple[str, int], list[str]] = {}
    for entry in draft.problems:
        m = _PROBLEM_RE.match(entry)
        if not m:
            out.setdefault(("", -1), []).append(entry)
            continue
        out.setdefault((m.group(1), int(m.group(2))), []).append(m.group(3))
    return out


# --------------------------------------------------------------------------- results / summary


def draft_stats() -> dict[str, int]:
    """Real distribution across the saved drafts. Computed live, never stored."""
    all_d = all_drafts()
    clean = sum(1 for d in all_d if d.clean)
    defective = sum(1 for d in all_d if d.status == "draft" and d.problems)
    insufficient = sum(1 for d in all_d if d.status != "draft")
    return {
        "total": len(all_d),
        "clean": clean,
        "defective": defective,
        "insufficient": insufficient,
    }


def retrieval_summary() -> dict[str, Any]:
    if RETRIEVAL_SUMMARY.is_file():
        return json.loads(RETRIEVAL_SUMMARY.read_text(encoding="utf-8"))
    return {}


def retrieval_bge_summary() -> dict[str, Any]:
    if RETRIEVAL_BGE_SUMMARY.is_file():
        return json.loads(RETRIEVAL_BGE_SUMMARY.read_text(encoding="utf-8"))
    return {}


def calibration_summary() -> dict[str, Any]:
    p = CALIBRATION_DIR / "calibration.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def ladder_rung2() -> dict[str, Any]:
    p = LADDER_DIR / "rung2_lgbm_plus_hop_summary.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def ladder_rung1() -> dict[str, Any]:
    p = LADDER_DIR / "rung1_lgbm_summary.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def top100_false_positive_share(base_rate: float = 3565 / 6_924_900) -> float | None:
    """Approximate share of top-100 alerts that are false positives.

    Derivation: if recall@100 is the fraction of all positives captured in the
    top-100 rows per fold, and the positive base rate is ``base_rate`` over
    N rows, then TP-per-100 ≈ recall × base_rate × N / (N/100) /
    (number of folds)... in practice we use the simpler identity:

        precision@100 ≈ recall@100 × base_rate × (N / 100) / N
                      = recall@100 × base_rate × 100 / 1 ... per fold

    but N varies per fold. The saved recall@100 is the *mean* over folds, so
    we compute per-fold from the score tables is heavy. Instead we use:

        FP share ≈ 1 - (recall@100 ��� positives-per-100-if-perfect) / 100

    where ``positives-per-100-if-perfect`` is not well-defined without N.
    We therefore provide the *raw* recall@100 and let the Results page quote
    the documented 80-97% range from the Stage 4 note as a separate line
    that cites its source note (not a hardcoded "fact" of this app).

    Returns None; kept for symmetry with callers that want a numeric knob.
    """
    return None


def package_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
