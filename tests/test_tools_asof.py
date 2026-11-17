"""Stage 6 Part B acceptance test: every tool is as-of clean and honours its contract.

The question is whether a tool asked about time *t* can answer using information from
strictly after *t*, and whether it reports honestly when it truncates. Four properties
are checked per tool:

1. ``as_of`` is required -- omitting it raises rather than defaulting to "now".
2. Temporal invariance -- an answer at ``as_of = D`` must be identical whether or not
   the underlying table contains rows from day ``D`` and later. The test feeds each
   tool a table truncated at ``D`` and asserts the result equals the result computed
   from the full table. If any tool aggregated over the whole file the two would differ.
3. Truncation honesty -- ``truncated`` is true exactly when the caller's cap bit, and
   ``n_matching`` still reports the uncapped count.
4. ``get_alert_evidence`` refuses lookahead and otherwise reproduces the saved Stage 5
   package byte for byte, tying the tool to the already-verified artefact.

Run with the learn_env interpreter from the project root::

    /home/dan/miniconda3/envs/learn_env/bin/python -m pytest tests/test_tools_asof.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # tools.py imports ``evidence`` bare

import evidence as ev  # noqa: E402
import tools  # noqa: E402

CANONICAL_PATH = PROJECT_ROOT / "data" / "_trans_canonical.parquet"
EVIDENCE_DIR = PROJECT_ROOT / "results" / "evidence"

# A day inside the Stage 5 sample window, and the day after it.
DAY = "2022-09-07"
DAY_NS = tools._as_of_ns(DAY)
DAY_NEXT = "2022-09-08"
DAY_NEXT_NS = tools._as_of_ns(DAY_NEXT)


def _truncate_at(ts_ns: int):
    """Canonical table with every row at or after ``ts_ns`` removed."""
    full = pq.read_table(CANONICAL_PATH)
    return full.filter(pc.less(full["ts_ns"], ts_ns))


@pytest.fixture(scope="module")
def full():
    return ev._load_table(None)


@pytest.fixture(scope="module")
def truncated():
    """The canonical table as it stood at 00:00 UTC on DAY (nothing from DAY onward)."""
    return _truncate_at(DAY_NS)


@pytest.fixture(scope="module")
def anchor():
    """A real (row, sender, receiver, bank) tuple from a saved evidence package."""
    path = sorted(EVIDENCE_DIR.glob("package_00_row*.json"))[0]
    pkg = json.loads(path.read_text())
    txn = pkg["transaction"]
    return {
        "path": path,
        "package": pkg,
        "row": int(txn["row"]),
        "sender": int(txn["from_account"]),
        "receiver": int(txn["to_account"]),
        "bank": int(txn["from_bank"]),
    }


# --- 1. as_of is required ----------------------------------------------------------


@pytest.mark.parametrize("tool_name,args", [
    ("get_account_history", (1,)),
    ("get_neighbours", (1,)),
    ("find_cycles", (1,)),
    ("get_alert_evidence", (1,)),
    ("get_bank_stats", (1,)),
])
def test_as_of_is_required(tool_name, args):
    fn = getattr(tools, tool_name)
    with pytest.raises(ValueError, match="as_of is required"):
        fn(*args, as_of=None)


def test_as_of_accepts_epoch_ns_and_iso_day(anchor):
    a = tools.get_account_history(anchor["sender"], DAY, limit=1000)
    b = tools.get_account_history(anchor["sender"], DAY_NS, limit=1000)
    assert a["rows"] == b["rows"]


def test_as_of_rejects_garbage():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        tools.get_account_history(1, "not-a-date")


# --- 2. temporal invariance --------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "truncated"])
def test_account_history_invariance(anchor, full, truncated, mode):
    table = full if mode == "full" else truncated
    r = tools.get_account_history(anchor["sender"], DAY, limit=1000, table=table)
    # An answer at DAY cannot contain a row stamped on or after DAY.
    assert all(row["ts_ns"] < DAY_NS for row in r["rows"])
    if mode == "truncated":
        again = tools.get_account_history(anchor["sender"], DAY, limit=1000, table=full)
        assert r == again


def test_neighbours_invariance(anchor, full, truncated):
    a = tools.get_neighbours(anchor["sender"], DAY, limit=1000, table=full)
    b = tools.get_neighbours(anchor["sender"], DAY, limit=1000, table=truncated)
    assert a == b
    for row in a["inbound"] + a["outbound"]:
        assert row["ts_ns"] < DAY_NS


def test_neighbours_windows(anchor, full):
    r = tools.get_neighbours(anchor["sender"], DAY, limit=10_000, table=full)
    for row in r["outbound"]:
        assert row["ts_ns"] > DAY_NS - ev.HOP1_WINDOW_NS
    for row in r["inbound"]:
        assert row["ts_ns"] > DAY_NS - ev.HOP2_WINDOW_NS
    # All-time degree is at least the windowed edge count; it never contradicts it.
    assert r["degree_all_time"]["out_degree_all_time"] >= r["n_outbound"]
    assert r["degree_all_time"]["in_degree_all_time"] >= r["n_inbound"]


def test_cycles_invariance(anchor, full, truncated):
    a = tools.find_cycles(anchor["sender"], DAY, table=full)
    b = tools.find_cycles(anchor["sender"], DAY, table=truncated)
    assert a == b
    for c in a["cycles"]:
        for hop in c["hops"]:
            assert hop["row"] >= 0


def test_cycles_only_returns_cycles_through_the_account(anchor, full):
    r = tools.find_cycles(anchor["sender"], DAY, table=full)
    for c in r["cycles"]:
        assert c["hops"][0]["from"] == anchor["sender"]
        assert c["hops"][-1]["to"] == anchor["sender"]


def test_bank_stats_invariance(anchor, full, truncated):
    a = tools.get_bank_stats(anchor["bank"], DAY, limit=1000, table=full)
    b = tools.get_bank_stats(anchor["bank"], DAY, limit=1000, table=truncated)
    assert a == b


# --- 3. truncation honesty ---------------------------------------------------------


def test_account_history_truncation_flag(anchor, full):
    narrow = tools.get_account_history(anchor["sender"], DAY, limit=1, table=full)
    assert narrow["truncated"] is (narrow["n_matching"] > 1)
    assert len(narrow["rows"]) == min(1, narrow["n_matching"])

    wide = tools.get_account_history(anchor["sender"], DAY, limit=10_000, table=full)
    assert wide["truncated"] is False
    assert wide["n_matching"] == len(wide["rows"])


def test_neighbours_truncation_flag(anchor, full):
    narrow = tools.get_neighbours(anchor["sender"], DAY, limit=1, table=full)
    expected = narrow["n_inbound"] > 1 or narrow["n_outbound"] > 1
    assert narrow["truncated"] is expected
    assert len(narrow["inbound"]) <= 1 and len(narrow["outbound"]) <= 1


def test_bank_stats_truncation_flag(anchor, full):
    narrow = tools.get_bank_stats(anchor["bank"], DAY, limit=1, table=full)
    assert narrow["truncated"] is (narrow["n_counterparty_banks"] > 1)
    assert len(narrow["top_counterparties"]) == 1

    wide = tools.get_bank_stats(anchor["bank"], DAY, limit=10_000, table=full)
    assert wide["truncated"] is False
    assert len(wide["top_counterparties"]) == wide["n_counterparty_banks"]


def test_bank_stats_counts_are_disjoint_and_complete(anchor, full):
    r = tools.get_bank_stats(anchor["bank"], DAY, limit=10_000, table=full)
    assert r["n_out"] + r["n_in"] + r["n_intra_bank"] == r["n_txn"]
    # Every counterparty bank is a different bank, never the queried one.
    assert all(c["bank"] != anchor["bank"] for c in r["top_counterparties"])
    assert sum(c["txn_count"] for c in r["top_counterparties"]) == r["n_out"] + r["n_in"]


def test_limit_must_be_positive(anchor, full):
    with pytest.raises(ValueError, match="limit must be positive"):
        tools.get_account_history(anchor["sender"], DAY, limit=0, table=full)


# --- 4. alert evidence: lookahead + parity with the saved artefact -----------------


def test_alert_evidence_rejects_lookahead(anchor):
    with pytest.raises(ValueError, match="as_of precedes the transaction"):
        tools.get_alert_evidence(anchor["row"], "2022-09-01")


def test_alert_evidence_matches_saved_package(anchor):
    built = tools.get_alert_evidence(anchor["row"], DAY_NEXT)
    saved = anchor["package"]
    # ``tool`` and ``as_of`` are envelope fields the sampler did not record.
    built_body = {k: v for k, v in built.items() if k not in ("tool", "as_of")}
    assert built_body == saved


def test_alert_evidence_is_as_of_clean(anchor, full, truncated):
    a = tools.get_alert_evidence(anchor["row"], DAY_NEXT, table=full)
    b = tools.get_alert_evidence(anchor["row"], DAY_NEXT, table=truncated)
    a.pop("as_of"), b.pop("as_of")
    assert a == b
