"""Stable transaction IDs for the canonical transaction table.

Run from the project root with the learn_env interpreter::

    python src/transactions.py

The dataset has **no transaction id column**. The natural key is
``(ts_ns, from_account, from_bank, to_account, to_bank, payment_format, amount_usd)``.
On the 6,924,049-row canonical table that key is unique for all but 8 rows: eight pairs
of transactions are identical in every stored field. Those pairs are genuinely
indistinguishable from the data, so the id keeps a ``dup_ordinal`` that says which member
of the pair a package refers to; without it two different alerts would hash to one id.

The id is ``txn_<16 hex chars>``, a SHA-256 over the natural key plus the ordinal. It is
stable across rebuilds as long as the canonical table is rebuilt from the same inputs, and
it is not derived from any file position, so re-sorting the table does not change it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data" / "_trans_canonical.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "_trans_ids.parquet"

# The natural key, in a fixed order. amount_usd is included because the eight duplicate
# pairs differ from their neighbours only by being exact copies; without the amount a
# different collision pattern would appear among same-payment same-instant rows.
KEY_COLS = ["ts_ns", "from_account", "from_bank", "to_account", "to_bank",
            "payment_format", "amount_usd"]


def _key_strings(frame: pd.DataFrame) -> list[str]:
    """One canonical string per row; stable across dtypes and numpy versions."""
    parts = [
        frame["ts_ns"].to_numpy(dtype=np.int64).astype(str),
        frame["from_account"].astype(str).to_numpy(),
        frame["from_bank"].to_numpy().astype(str),
        frame["to_account"].astype(str).to_numpy(),
        frame["to_bank"].to_numpy().astype(str),
        frame["payment_format"].to_numpy().astype(str),
        # 6 dp is the canonicalisation precision; repr() would be float-noise sensitive.
        np.char.mod("%.6f", frame["amount_usd"].to_numpy(dtype=np.float64)),
    ]
    return ["|".join(row) for row in zip(*parts)]


def assign_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with ``dup_ordinal`` and ``transaction_id`` columns added.

    ``dup_ordinal`` is 0 for every unique row and 0/1/... within a duplicate group, in
    canonical-table order. It is what makes the id unique on this table.
    """
    keys = _key_strings(frame)
    ordinal = np.zeros(len(keys), dtype=np.int16)
    seen: dict[str, int] = {}
    for i, k in enumerate(keys):
        c = seen.get(k, 0)
        ordinal[i] = c
        seen[k] = c + 1

    ids = [
        "txn_" + hashlib.sha256(f"{k}#{o}".encode()).hexdigest()[:16]
        for k, o in zip(keys, ordinal)
    ]
    out = frame.copy()
    out["dup_ordinal"] = ordinal
    out["transaction_id"] = ids
    return out


def load_ids(path: Path = CANONICAL_PATH, force: bool = False) -> pd.DataFrame:
    """Load (or build and cache) the canonical table with ids attached.

    Only the key columns are read, so the cache is small; callers that need the rest join
    the cache back on ``transaction_id`` after reading the full table.
    """
    if OUT_PATH.exists() and not force and OUT_PATH.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(OUT_PATH)
    import pyarrow.parquet as pq

    frame = pq.read_table(path, columns=KEY_COLS + ["date"]).to_pandas()
    out = assign_ids(frame)
    out.to_parquet(OUT_PATH, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    ids = load_ids(force=True)
    n = len(ids)
    n_distinct = ids["transaction_id"].nunique()
    n_dup_groups = int((ids["dup_ordinal"] > 0).sum())
    print(f"{n:,} rows -> {n_distinct:,} distinct transaction ids")
    if n_distinct != n:
        raise AssertionError(f"transaction_id is not unique: {n - n_distinct} collisions")
    print(f"duplicate rows needing an ordinal: {n_dup_groups} "
          f"(in {int((ids['dup_ordinal'] == 1).sum())} pairs)")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
