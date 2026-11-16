"""Stage 3: leakage-safe, versioned edge-level feature pipeline.

As-of rule (stricter than a per-fold cutoff)
--------------------------------------------
For a transaction on day ``D``, every feature about its sender or receiver uses only
transactions dated **strictly before D**. Not "before the fold cutoff" - before the
row's own day. This guarantees an edge's own amount cannot feed its own features, which
is what the self-contribution test checks.

Why the day boundary is safe
----------------------------
All transactions on day D share the same features (built from D-1 and earlier), so the
pipeline works in two steps per day:

1. read day D's edges, attach the *current* cumulative snapshot (built from D-1 and
   earlier), and write those features out;
2. fold day D's edges into the snapshot for the next day.

Step 2 keeps velocity *counts* built from day D's edges, but never reads
``is_laundering``, so no label leaks. Cumulative sums are maintained incrementally; the
full history is never regrouped.

Feature families (all sender and receiver separately unless noted)
-------------------------------------------------------------------
- amount: count, mean, std, max of ``amount_usd``
- counterparty: fan_out, fan_in (distinct counterparties)
- cross-bank fraction of prior activity
- account age: days since first observed transaction, measured from the window start
- timing: smallest gap in seconds between the account's consecutive prior transactions,
  floored at 60 s (sub-minute gaps are a simulator artefact)
- 1-hop: mean and max of counterparties' own as-of fan counts
- transaction: log amount_usd, payment format one-hot, cross_bank flag
- velocity: distinct (from, to) relationship count (contract s6.2; the note's
  (from,to,amount) key is not used - see the manifest for the recorded deviation)

Amounts
-------
USD amounts are built from the SAVED fx tables exactly as ``docs/data_contract.md`` s3
specifies: fiat via ``fx_usd_rate.csv``, Bitcoin via ``fx_btc_daily_rate.csv``. The script
asserts **zero** unconverted rows before doing anything else.

Memory
------
The 650 MB CSV is scanned twice: pass 1 learns the account / bank / payment-format
vocabularies, pass 2 writes a coded parquet cache (``data/_trans_canonical.parquet``)
with ``uint32`` account codes, ``uint16`` bank codes, ``uint8`` format codes and
``float64`` amounts.
That avoids the multi-GB object frame pandas builds from string columns. The cache is
reused on subsequent runs; delete it or pass ``--rebuild`` to regenerate.

The snapshot is a dense per-account numpy table indexed by code, and each day's features
are appended straight to the parquet writer, so no cross-day frames accumulate. Account
IDs are restored as strings on output, in row order.

Run::

    /home/dan/miniconda3/envs/learn_env/bin/python src/features/build_edge_features.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.measure_daily_table import TRANS_PATH  # noqa: E402

FX_USD = PROJECT_ROOT / "eda_output" / "fx_usd_rate.csv"
FX_BTC = PROJECT_ROOT / "eda_output" / "fx_btc_daily_rate.csv"
OUT_DIR = PROJECT_ROOT / "eda_output" / "features"
VOCAB_CACHE = PROJECT_ROOT / "data" / "_trans_vocab.json"
CANON_CACHE = PROJECT_ROOT / "data" / "_trans_canonical.parquet"
VERSION = "v1"
CACHE_VERSION = "2"
CHUNK_ROWS = 1_000_000

READ_COLS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
]
RAW_TO_CANON = {
    "From Bank": "from_bank", "Account": "from_account",
    "To Bank": "to_bank", "Account.1": "to_account",
    "Amount Paid": "amount_paid", "Payment Currency": "currency_paid",
    "Payment Format": "payment_format", "Is Laundering": "is_laundering",
}

PAYMENT_FORMATS = [
    "ACH", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire",
]
FMT_COLS = [f"fmt_{f.lower().replace(' ', '_')}" for f in PAYMENT_FORMATS]

STATE_COLS = [
    "txn_count", "amount_mean", "amount_std", "amount_max",
    "fan", "fan_out", "fan_in", "cross_bank_frac",
]

FEATURE_COLS = (
    [f"from_{c}" for c in STATE_COLS] + ["from_age_days", "from_has_history",
     "from_hop_mean_fan", "from_hop_max_fan"]
    + [f"to_{c}" for c in STATE_COLS] + ["to_age_days", "to_has_history",
       "to_hop_mean_fan", "to_hop_max_fan"]
    + ["from_min_gap_seconds", "to_min_gap_seconds", "log_amount_usd", "is_cross_bank"]
)
OUT_COLS = ["from_account", "to_account", "date", "is_laundering"] + FEATURE_COLS + FMT_COLS


# --- Load: pass 1 vocab, pass 2 coded cache ---------------------------------------


def _read_chunks(path: Path):
    return pd.read_csv(
        path, usecols=READ_COLS,
        dtype={"Account": "str", "Account.1": "str"},
        parse_dates=["Timestamp"], chunksize=CHUNK_ROWS,
    )


def _fx_tables() -> tuple[pd.Series, pd.Series]:
    usd_rate = pd.read_csv(FX_USD, index_col=0)["usd_rate"]
    btc_rate = pd.read_csv(FX_BTC, index_col=0)["btc_usd_rate"]
    btc_rate.index = pd.to_datetime(btc_rate.index)
    return usd_rate, btc_rate


def build_vocab(path: Path) -> dict:
    """Pass 1: every distinct account, bank and payment format, sorted for stability."""
    accts: set[str] = set()
    banks: set[str] = set()
    fmts: set[str] = set()
    for raw in _read_chunks(path):
        df = raw.rename(columns=RAW_TO_CANON)
        accts.update(df["from_account"].unique().tolist())
        accts.update(df["to_account"].unique().tolist())
        banks.update(df["from_bank"].unique().tolist())
        banks.update(df["to_bank"].unique().tolist())
        fmts.update(df["payment_format"].unique().tolist())

    vocab = {
        "version": CACHE_VERSION,
        "accounts": sorted(accts),
        "banks": sorted(banks),
        "formats": sorted(fmts),
    }
    VOCAB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_CACHE.write_text(json.dumps(vocab))
    return vocab


def encode_to_cache(path: Path, vocab: dict) -> None:
    """Pass 2: strings -> codes, amounts -> USD; write the canonical parquet cache."""
    usd_rate, btc_rate = _fx_tables()
    acct_index = pd.Index(vocab["accounts"])
    bank_index = pd.Index(vocab["banks"])
    fmt_index = pd.Index(vocab["formats"])
    if len(acct_index) >= 2**32:
        raise AssertionError(f"account vocabulary {len(acct_index):,} does not fit uint32")
    if len(bank_index) >= 2**16:
        raise AssertionError(f"bank vocabulary {len(bank_index):,} does not fit uint16")
    if len(fmt_index) > 255:
        raise AssertionError("format vocabulary must fit uint8")

    writer = None
    try:
        for raw in _read_chunks(path):
            df = raw.rename(columns=RAW_TO_CANON)
            date = df["Timestamp"].dt.normalize()

            is_btc = df["currency_paid"].to_numpy() == "Bitcoin"
            fiat = df["amount_paid"].to_numpy() * df["currency_paid"].map(usd_rate).to_numpy()
            btc = df["amount_paid"].to_numpy() * date.map(btc_rate).to_numpy()
            amount_usd = np.where(is_btc, btc, fiat)
            bad = np.isnan(amount_usd)
            if bad.any():
                pairs = df.loc[bad, ["currency_paid"]].assign(date=date[bad])
                raise AssertionError(
                    f"{int(bad.sum()):,} rows have no USD rate. Offending pairs:\n"
                    f"{pairs.drop_duplicates().head(20)}"
                )

            out = pd.DataFrame({
                "from_account": acct_index.get_indexer(df["from_account"]).astype(np.uint32),
                "to_account": acct_index.get_indexer(df["to_account"]).astype(np.uint32),
                "from_bank": bank_index.get_indexer(df["from_bank"]).astype(np.uint16),
                "to_bank": bank_index.get_indexer(df["to_bank"]).astype(np.uint16),
                "payment_format": fmt_index.get_indexer(df["payment_format"]).astype(np.uint8),
                "date": date.to_numpy(),
                "ts_ns": df["Timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64),
                "amount_usd": amount_usd,
                "is_laundering": df["is_laundering"].to_numpy().astype(np.int8),
            })
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(CANON_CACHE, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def load_edges(path: Path, rebuild: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load coded transactions, building the canonical cache on first use."""
    fresh = rebuild or not (VOCAB_CACHE.exists() and CANON_CACHE.exists())
    if fresh:
        for p in (VOCAB_CACHE, CANON_CACHE):
            if p.exists():
                p.unlink()
        t = time.perf_counter()
        vocab = build_vocab(path)
        print(f"vocab: {len(vocab['accounts']):,} accounts, {len(vocab['banks']):,} banks, "
              f"{len(vocab['formats'])} formats  ({time.perf_counter() - t:,.0f}s)")
        t = time.perf_counter()
        encode_to_cache(path, vocab)
        print(f"canonical cache written; USD conversion verified, 0 unconverted  "
              f"({time.perf_counter() - t:,.0f}s)")

    vocab = json.loads(VOCAB_CACHE.read_text())
    if vocab.get("version") != CACHE_VERSION:
        return load_edges(path, rebuild=True)
    return pd.read_parquet(CANON_CACHE), vocab


# --- Snapshot ---------------------------------------------------------------------


class Snapshot:
    """Cumulative per-account state as dense arrays indexed by account code."""

    def __init__(self, n_accounts: int) -> None:
        self.n = n_accounts
        self.txn_count = np.zeros(n_accounts, dtype=np.int64)
        self.amount_sum = np.zeros(n_accounts, dtype=np.float64)
        self.amount_sumsq = np.zeros(n_accounts, dtype=np.float64)
        self.amount_max = np.zeros(n_accounts, dtype=np.float64)
        self.cross_bank_edges = np.zeros(n_accounts, dtype=np.int64)
        self.fan_out = np.zeros(n_accounts, dtype=np.int32)
        self.fan_in = np.zeros(n_accounts, dtype=np.int32)
        self.first_day_ns = np.full(n_accounts, np.iinfo(np.int64).min, dtype=np.int64)
        # Timestamp of the account's most recent prior edge, and the smallest gap between
        # consecutive prior edges, both in nanoseconds.
        self.last_txn_ns = np.full(n_accounts, np.iinfo(np.int64).min, dtype=np.int64)
        self.min_gap_ns = np.full(n_accounts, np.iinfo(np.int64).max, dtype=np.int64)
        # Distinct directed relationships seen so far, keyed as acct * n + partner.
        self._keys_out = np.empty(0, dtype=np.int64)
        self._keys_in = np.empty(0, dtype=np.int64)

    @staticmethod
    def _pairs_key(acct: np.ndarray, partner: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Unique account/partner pairs and their linear keys."""
        pairs = np.unique(np.stack([acct, partner], axis=1), axis=0)
        return pairs, pairs[:, 0].astype(np.int64) * n + pairs[:, 1]

    def update_day(self, day: pd.DataFrame, day_ts: pd.Timestamp) -> None:
        """Fold one day's edges into the cumulative state. Never reads labels."""
        day_ns = np.datetime64(day_ts, "ns").astype(np.int64)
        amounts = day["amount_usd"].to_numpy()
        cross = (day["from_bank"].to_numpy() != day["to_bank"].to_numpy()).astype(np.int64)

        for side, acct_col, partner_col in (
            ("out", "from_account", "to_account"),
            ("in", "to_account", "from_account"),
        ):
            acct = day[acct_col].to_numpy()
            np.add.at(self.txn_count, acct, 1)
            np.add.at(self.amount_sum, acct, amounts)
            np.add.at(self.amount_sumsq, acct, amounts * amounts)
            np.maximum.at(self.amount_max, acct, amounts)
            np.add.at(self.cross_bank_edges, acct, cross)

            unseen = self.first_day_ns[acct] == np.iinfo(np.int64).min
            self.first_day_ns[acct[unseen]] = day_ns

            # Gap between consecutive edges of this account. The day's timestamps come
            # from the file in arbitrary order, so sort by (account, timestamp) and take
            # the diffs; the first edge of an account also diffs against its prior day's
            # last edge, which is a real gap across the midnight boundary.
            ts = day["ts_ns"].to_numpy().astype(np.int64)
            order = np.lexsort((ts, acct))
            a_s, t_s = acct[order], ts[order]
            first_of_acct = np.r_[True, a_s[1:] != a_s[:-1]]
            starts = np.flatnonzero(first_of_acct)
            owners = a_s[starts].astype(np.int64)
            # Within-day gaps: diff(t_s) everywhere except across an account boundary.
            if t_s.size > 1:
                step = np.diff(t_s)
                same = ~first_of_acct[1:]
                if same.any():
                    np.minimum.at(self.min_gap_ns, a_s[1:][same], step[same])
            # Cross-day gap: first edge today minus the prior day's last edge.
            last = self.last_txn_ns[owners]
            seen = last != np.iinfo(np.int64).min
            if seen.any():
                np.minimum.at(self.min_gap_ns, owners[seen], t_s[starts][seen] - last[seen])
            np.maximum.at(self.last_txn_ns, acct, ts)

            pairs, new_keys = self._pairs_key(acct, day[partner_col].to_numpy(), self.n)
            keys = self._keys_out if side == "out" else self._keys_in
            if keys.size:
                fresh = new_keys[~np.isin(new_keys, keys, assume_unique=True)]
            else:
                fresh = new_keys
            if fresh.size:
                keys = np.union1d(keys, fresh)
                if side == "out":
                    self._keys_out = keys
                    np.add.at(self.fan_out, (fresh // self.n).astype(np.int64), 1)
                else:
                    self._keys_in = keys
                    np.add.at(self.fan_in, (fresh // self.n).astype(np.int64), 1)

    def frame(self) -> dict[str, np.ndarray]:
        """Materialise the snapshot columns as float arrays indexed by account code."""
        n = self.txn_count.astype(np.float64)
        safe_n = np.maximum(n, 1.0)
        mean = self.amount_sum / safe_n
        std = np.sqrt(np.maximum(self.amount_sumsq / safe_n - mean**2, 0.0))
        return {
            "txn_count": n,
            "amount_mean": mean,
            "amount_std": std,
            "amount_max": self.amount_max,
            "fan": (self.fan_out + self.fan_in).astype(np.float64),
            "fan_out": self.fan_out.astype(np.float64),
            "fan_in": self.fan_in.astype(np.float64),
            "cross_bank_frac": self.cross_bank_edges / safe_n,
        }


# --- Build ------------------------------------------------------------------------


def build(edges: pd.DataFrame, vocab: dict, out_path: Path, verbose: bool = True) -> dict:
    """Build edge features one calendar day at a time, appending to parquet."""
    acct_labels = np.array(vocab["accounts"], dtype=object)
    bank_labels = np.array(vocab["banks"], dtype=object)
    fmt_labels = np.array(vocab["formats"], dtype=object)
    n_accounts = len(acct_labels)

    days = np.unique(edges["date"].to_numpy())
    day_codes = edges["date"].to_numpy()
    snap = Snapshot(n_accounts)

    writer = None
    rows_written = 0
    try:
        for day_ts in days:
            idx = np.flatnonzero(day_codes == day_ts)
            day = edges.iloc[idx]
            day_pd = pd.Timestamp(day_ts)
            state = snap.frame()
            day_ns = np.datetime64(day_ts, "ns").astype(np.int64)

            from_code = day["from_account"].to_numpy()
            to_code = day["to_account"].to_numpy()
            f: dict[str, np.ndarray] = {
                "from_account": acct_labels[from_code],
                "to_account": acct_labels[to_code],
                "date": np.full(idx.shape[0], day_pd, dtype="datetime64[ns]"),
                "is_laundering": day["is_laundering"].to_numpy(),
            }

            for side, codes in (("from", from_code), ("to", to_code)):
                hist = snap.txn_count[codes] > 0
                f[f"{side}_has_history"] = hist.astype(np.int8)
                for c in STATE_COLS:
                    f[f"{side}_{c}"] = state[c][codes]
                first = snap.first_day_ns[codes]
                age = np.where(hist, (day_ns - first) // 86_400_000_000_000, -1)
                f[f"{side}_age_days"] = age.astype(np.int32)
                # Floor the gap at one minute: sub-minute gaps are a timestamp artefact
                # of the simulator, not a behaviour worth representing. An account can
                # have history yet no gap (one prior edge only), so guard on the sentinel.
                raw_gap = snap.min_gap_ns[codes]
                has_gap = raw_gap != np.iinfo(np.int64).max
                gap_s = np.maximum(raw_gap // 1_000_000_000, 60)
                f[f"{side}_min_gap_seconds"] = np.where(has_gap, gap_s, -1).astype(np.int64)

            # 1-hop: aggregate counterparties' own as-of fan over the day's *distinct*
            # pairs, per account, then scatter back to rows. Sorting is by account code,
            # so the reduced array is indexed by account and gather is length len(day).
            fan = state["fan"]
            pairs = np.unique(np.stack([from_code, to_code], axis=1), axis=0)

            def _hop_by_acct(owner: np.ndarray, partner: np.ndarray):
                """Mean and max of partner fan over distinct partners, indexed by acct code."""
                nb = fan[partner]
                order = np.argsort(owner, kind="stable")
                o_s, nb_s = owner[order], nb[order]
                starts = np.flatnonzero(np.r_[True, o_s[1:] != o_s[:-1]])
                step = np.diff(np.r_[starts, o_s.shape[0]])
                mean_by_acct = np.zeros(n_accounts, dtype=np.float64)
                max_by_acct = np.zeros(n_accounts, dtype=np.float64)
                mean_by_acct[o_s[starts].astype(np.int64)] = np.add.reduceat(nb_s, starts) / step
                max_by_acct[o_s[starts].astype(np.int64)] = np.maximum.reduceat(nb_s, starts)
                return mean_by_acct, max_by_acct

            for side, codes, a_col, b_col in (
                ("from", from_code, pairs[:, 0], pairs[:, 1]),
                ("to", to_code, pairs[:, 1], pairs[:, 0]),
            ):
                mean_by_acct, max_by_acct = _hop_by_acct(a_col, b_col)
                codes_i = codes.astype(np.int64)
                f[f"{side}_hop_mean_fan"] = mean_by_acct[codes_i]
                f[f"{side}_hop_max_fan"] = max_by_acct[codes_i]

            amounts = day["amount_usd"].to_numpy()
            f["log_amount_usd"] = np.log1p(amounts)
            f["is_cross_bank"] = (day["from_bank"].to_numpy() != day["to_bank"].to_numpy()).astype(np.int8)
            fmt = day["payment_format"].to_numpy()
            for fmt_name, col in zip(PAYMENT_FORMATS, FMT_COLS):
                f[col] = (fmt_labels[fmt] == fmt_name).astype(np.int8)

            frame = pd.DataFrame({c: f[c] for c in OUT_COLS})
            nan_cols = {c: int(frame[c].isna().sum()) for c in FEATURE_COLS
                        if frame[c].isna().any()}
            if nan_cols:
                raise AssertionError(f"NaN in features on {day_pd.date()}: {nan_cols}")

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            rows_written += table.num_rows

            snap.update_day(day, day_pd)
            if verbose:
                print(f"  {day_pd.date()}  rows {table.num_rows:>7,}  "
                      f"accounts in snapshot {int((snap.txn_count > 0).sum()):>7,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows_written, "days": int(days.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=TRANS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the coded cache")
    args = parser.parse_args()

    t0 = time.perf_counter()
    edges, vocab = load_edges(args.path, rebuild=args.rebuild)
    print(f"loaded {len(edges):,} edges coded (accounts={len(vocab['accounts']):,})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"edge_features_{VERSION}.parquet"
    info = build(edges, vocab, out)
    elapsed = time.perf_counter() - t0
    print(f"wrote {out}  rows {info['rows']:,}  runtime {elapsed:,.0f}s")

    summary = {
        "version": VERSION,
        "rows": info["rows"],
        "days": info["days"],
        "columns": len(OUT_COLS),
        "runtime_seconds": round(elapsed, 1),
        "disk_mb": round(out.stat().st_size / 1e6, 1),
        "format_columns": FMT_COLS,
    }
    (args.out_dir / f"edge_features_{VERSION}_build.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
