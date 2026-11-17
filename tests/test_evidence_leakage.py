"""Stage 5 acceptance test: the evidence builder is as-of clean.

The question is whether a package for transaction *t* can contain, or change when given,
information from strictly after *t*. The test answers it by building the same package
twice: once from the full canonical table, once from a table truncated at *t*'s
timestamp, and asserting the two are byte-identical after JSON serialisation.

If any field - a score, a neighbour count, a cycle, a fan star, a chain, the sender's
history - were computed over the whole file, the truncated build would see less and the
two would differ. The schema and the builder are placeholders here: this test lives
ahead of ``src/evidence.py`` so the contract is fixed before the code is written. Run
one builder first, then run this test and the two should never disagree.

Run with the learn_env interpreter from the project root::

    /home/dan/miniconda3/envs/learn_env/bin/python -m pytest tests/test_evidence_leakage.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_PATH = PROJECT_ROOT / "data" / "_trans_canonical.parquet"
SCORES_PATH = PROJECT_ROOT / "results" / "ladder" / "rung2_lgbm_plus_hop_scores.parquet"

# Number of sample transactions to test. A package build is expensive (reads the full
# table), so the test picks a fixed seed and a small slice.
N_SAMPLES = 5
SEED = 42


def _select_samples() -> pd.DataFrame:
    """Pick ``N_SAMPLES`` transactions deterministically: a mix of laundering and not."""
    scores = pq.read_table(SCORES_PATH, columns=["row", "is_laundering"]).to_pandas()
    pos = scores[scores["is_laundering"] == 1]
    neg = scores[scores["is_laundering"] == 0]
    n_pos = N_SAMPLES // 2
    n_neg = N_SAMPLES - n_pos
    pos_s = pos.sample(n=n_pos, random_state=SEED)
    neg_s = neg.sample(n=n_neg, random_state=SEED)
    return pd.concat([pos_s, neg_s], ignore_index=True)


def _truncate_table_at(ts_ns: int) -> pa.Table:
    """Canonical table with every row at or after ``ts_ns`` removed.

    Row groups overlap in time (the file is not sorted by timestamp), so we read every
    row group and filter, not just the groups whose min is below ``ts_ns``. The filter
    is strict: ``ts_n < ts_ns`` means a package built at time *t* cannot see *t* itself,
    let alone anything later.
    """
    full = pq.read_table(CANONICAL_PATH)
    return full.filter(pc.less(full["ts_ns"], ts_ns))


def _serialise(package) -> str:
    """Canonical JSON with sorted keys, so byte-equality means field-equality."""
    return json.dumps(package, sort_keys=True, default=str)


def _build(row: int, table: pa.Table):
    evidence = pytest.importorskip(
        "src.evidence", reason="src/evidence.py not written yet; contract fixed ahead of builder"
    )
    return evidence.build_evidence(row=row, table=table)


@pytest.fixture(scope="module")
def samples() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        pytest.skip(f"missing {SCORES_PATH}; run the model ladder first")
    return _select_samples()


@pytest.mark.parametrize("i", range(N_SAMPLES))
def test_package_identical_under_truncation(i, samples):
    row = int(samples.iloc[i]["row"])
    label = int(samples.iloc[i]["is_laundering"])

    full = pq.read_table(CANONICAL_PATH)
    ts_ns = int(full["ts_ns"][row].as_py())
    del full

    pkg_full = _build(row, pq.read_table(CANONICAL_PATH))
    pkg_cut = _build(row, _truncate_table_at(ts_ns))

    assert _serialise(pkg_full) == _serialise(pkg_cut), (
        f"evidence package for row {row} (label={label}, ts={ts_ns}) "
        f"changes when the table is truncated at the transaction's own timestamp - "
        f"the builder reads data from after the transaction"
    )
