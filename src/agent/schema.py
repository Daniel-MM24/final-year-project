"""Case-draft schema. Every factual statement is a Claim carrying citations.

Citation sources, exactly three kinds (Stage 7 rule 4):
  * ``package`` -- a dotted path into the alert's evidence package, e.g.
    ``transaction.amount_usd`` or ``context.sender_history[0].row``. A path that
    does not resolve is a schema violation, so citing a field that does not exist
    is caught at validation time rather than by a human reader.
  * ``tool``    -- ``<tool_name>:<call_id>`` where ``call_id`` is the id the
    orchestrator assigned to that tool result in the run log.
  * ``passage`` -- a retrieved document id from the text index.

Statements with no citation are not allowed. There is no free-floating prose
field on the draft body for exactly this reason.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1.0"

SECTIONS = (
    "summary",
    "observed_facts",
    "hypothesised_typology",
    "supporting_evidence",
    "counter_evidence",
    "recommended_next_steps",
)

DRAFT_LABEL = "DRAFT for investigator review"
INSUFFICIENT = "insufficient evidence"

CITE_KINDS = ("package", "tool", "passage")
_TOOL_CITE_RE = re.compile(r"^[a-z_]+:call_[A-Za-z0-9_\-]+$")
_PASSAGE_CITE_RE = re.compile(r"^ref:[A-Za-z0-9_\-./:]+$")
_CASE_CITE_RE = re.compile(r"^case:row\d+$")


class SchemaError(ValueError):
    """A draft (or one of its parts) violates the case-draft contract."""


@dataclass(frozen=True)
class Claim:
    """One factual statement and the citations that support it."""

    text: str
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise SchemaError("claim text is empty")
        if not self.citations:
            raise SchemaError(f"claim has no citation: {self.text!r}")


@dataclass
class CaseDraft:
    """The only shape the agent is allowed to emit.

    ``status`` is ``"draft"`` or ``"insufficient_evidence"``. When insufficient,
    ``missing`` must be non-empty and every other section must be empty -- the
    agent abstains rather than inventing a typology.
    """

    alert_id: str
    run_id: str
    status: str
    summary: list[Claim] = field(default_factory=list)
    observed_facts: list[Claim] = field(default_factory=list)
    hypothesised_typology: list[Claim] = field(default_factory=list)
    supporting_evidence: list[Claim] = field(default_factory=list)
    counter_evidence: list[Claim] = field(default_factory=list)
    recommended_next_steps: list[Claim] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    label: str = DRAFT_LABEL
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def cite_kind(cite: str) -> str:
    """Classify a citation string. Raises SchemaError on anything malformed."""
    if not cite or not isinstance(cite, str):
        raise SchemaError("citation is not a non-empty string")
    if _TOOL_CITE_RE.match(cite):
        return "tool"
    if cite.startswith("transaction.") or cite.startswith("model.") or \
            cite.startswith("features.") or cite.startswith("context.") or \
            cite.startswith("provenance."):
        return "package"
    if _PASSAGE_CITE_RE.match(cite):
        return "passage"
    if _CASE_CITE_RE.match(cite):
        return "case"
    raise SchemaError(f"unrecognised citation: {cite!r}")


def resolve_package_path(package: dict, path: str):
    """Resolve a dotted path against the evidence package. Returns the leaf value.

    Supports ``a.b`` and ``a.b[3]`` index access on lists. Raises SchemaError when
    the path does not exist in the package.
    """
    node = package
    for part in path.split("."):
        m = re.match(r"^([A-Za-z0-9_]+)((?:\[\d+\])*)$", part)
        if not m:
            raise SchemaError(f"bad package path segment: {part!r}")
        key = m.group(1)
        if not isinstance(node, dict) or key not in node:
            raise SchemaError(f"package path does not exist: {path!r}")
        node = node[key]
        for idx in re.findall(r"\[(\d+)\]", m.group(2)):
            i = int(idx)
            if not isinstance(node, list) or i >= len(node):
                raise SchemaError(f"package path index out of range: {path!r}")
            node = node[i]
    return node


def validate_draft(draft: CaseDraft, package: dict, *,
                   passage_ids: set[str], tool_call_ids: set[str],
                   case_ids: set[str] | None = None) -> list[str]:
    """Validate a draft against the package, the passage set and the run's tool calls.

    Returns a list of human-readable problems (empty means valid). Never raises on
    a content problem; raises SchemaError only on malformed citations.
    """
    problems: list[str] = []
    if draft.status not in ("draft", DELETE := "insufficient_evidence"):  # noqa: F841
        problems.append(f"unknown status: {draft.status!r}")
    if draft.label != DRAFT_LABEL:
        problems.append(f"label must be {DRAFT_LABEL!r}, got {draft.label!r}")

    body_sections = [s for s in SECTIONS]
    has_any = any(getattr(draft, s) for s in body_sections)

    if draft.status == "insufficient_evidence":
        if not draft.missing:
            problems.append("insufficient_evidence must list what is missing")
        if has_any:
            problems.append("insufficient_evidence must not contain body sections")
    else:
        if not draft.summary:
            problems.append("draft has an empty summary")
        if draft.missing:
            problems.append("a non-abstaining draft must not populate 'missing'")

    for section in body_sections:
        for i, claim in enumerate(getattr(draft, section)):
            if not claim.citations:
                problems.append(f"{section}[{i}]: no citation")
            for cite in claim.citations:
                try:
                    kind = cite_kind(cite)
                except SchemaError as exc:
                    problems.append(f"{section}[{i}]: {exc}")
                    continue
                if kind == "package":
                    try:
                        resolve_package_path(package, cite)
                    except SchemaError as exc:
                        problems.append(f"{section}[{i}]: {exc}")
                elif kind == "tool":
                    if cite not in tool_call_ids:
                        problems.append(f"{section}[{i}]: unknown tool call {cite!r}")
                elif kind == "passage":
                    if cite not in passage_ids:
                        problems.append(f"{section}[{i}]: unknown passage {cite!r}")
                elif kind == "case":
                    if case_ids is not None and cite not in case_ids:
                        problems.append(f"{section}[{i}]: unknown case {cite!r}")
    return problems
