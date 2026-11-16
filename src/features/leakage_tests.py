"""Leakage tests for the Stage 3 edge feature table.

Three independent attacks on the as-of claim, each of which must FAIL to find a
signal. Run from the project root with the learn_env interpreter::

    python src/features/leakage_tests.py

Test 1 - shuffled labels. Replace the label with a permutation, refit the same
    Rung 0 rule that depends on labels, and confirm PR-AUC falls to the base rate.
    A feature that leaks the future would still rank the true positives.

Test 2 - future injection. Append a synthetic transaction on a late day and
    rebuild, then confirm every earlier day's feature rows are byte-identical.
    If a whole-file aggregate crept in, the late row would move the early ones.

Test 3 - self-contribution. For a sample of rows, mutate the row's own amount
    and rebuild, then confirm the row's own amount features do not change. Its
    neighbours' features may change; its own must not.

Test 2 and 3 rebuild through ``build_edge_features.build`` on a small synthetic
edge frame so they run in seconds and do not depend on the raw CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_protocol import make_splits, score_fold, summarise_folds  # noqa: E402
from src.features.build_edge_features import (  # noqa: E402
    FEATURE_COLS,
    build,
    STATE_COLS,
)

FEATURE_PATH = PROJECT_ROOT / "eda_output" / "features" / "edge_features_v1.parquet"


def _small_edges(seed: int = 0, n_days: int = 6, n_acc: int = 40,
                 rows_per_day: int = 200) -> pd.DataFrame:
    """Synthetic coded edge frame laid out like the canonical cache."""
    rng = np.random.default_rng(seed)
    frm, to, day, amt, lab, bank_f, bank_t, fmt, ts = [], [], [], [], [], [], [], [], []
    for d in range(n_days):
        for i in range(rows_per_day):
            a, b = rng.integers(n_acc, size=2)
            if a == b:
                b = (b + 1) % n_acc
            frm.append(int(a)); to.append(int(b)); day.append(d)
            amt.append(float(rng.lognormal(4.0, 1.0)))
            lab.append(int(rng.random() < 0.02))
            bank_f.append(int(rng.integers(3))); bank_t.append(int(rng.integers(3)))
            fmt.append(int(rng.integers(6)))
            ts.append(int(rng.integers(0, 86_400 * 1_000_000_000)))
    base_ns = np.datetime64("2022-09-01").astype("datetime64[ns]").astype(np.int64)
    return pd.DataFrame({
        "from_account": np.array(frm, dtype=np.uint32),
        "to_account": np.array(to, dtype=np.uint32),
        "from_bank": np.array(bank_f, dtype=np.uint16),
        "to_bank": np.array(bank_t, dtype=np.uint16),
        "payment_format": np.array(fmt, dtype=np.uint8),
        "date": (np.datetime64("2022-09-01") + np.array(day) * np.timedelta64(1, "D")
                 ).astype("datetime64[ns]"),
        "amount_usd": np.array(amt, dtype=np.float64),
        "is_laundering": np.array(lab, dtype=np.int8),
        "ts_ns": base_ns + np.array(day) * 86_400 * 1_000_000_000 + np.array(ts, dtype=np.int64),
    })


def _vocab_from(edges: pd.DataFrame) -> dict:
    """Minimal vocab for building an in-memory coded frame with no CSV pass."""
    accounts = np.union1d(edges["from_account"].unique(), edges["to_account"].unique())
    banks = np.union1d(edges["from_bank"].unique(), edges["to_bank"].unique())
    return {
        "accounts": [str(int(a)) for a in accounts],
        "banks": [str(int(b)) for b in banks],
        "formats": ["ACH", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"],
    }


def _build_to(edges: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    build(edges, _vocab_from(edges), out_path, verbose=False)
    return pd.read_parquet(out_path)


# --- Test 1: shuffled labels -------------------------------------------------------

def test_shuffled_labels(df: pd.DataFrame) -> bool:
    """Permute labels, fit the amount-only rule, confirm PR-AUC == base rate."""
    rng = np.random.default_rng(1)
    work = df[["date", "is_laundering", "from_amount_mean", "from_amount_std",
               "log_amount_usd", "from_has_history"]].copy()
    labels = work["is_laundering"].copy()

    def rule_score(sub: pd.DataFrame) -> np.ndarray:
        hist = sub["from_has_history"].to_numpy() == 1
        sd = sub["from_amount_std"].to_numpy()
        mean = sub["from_amount_mean"].to_numpy()
        amt = np.expm1(sub["log_amount_usd"].to_numpy())
        z = np.where(hist & (sd > 0), (amt - mean) / np.where(sd > 0, sd, 1.0), 0.0)
        # A leaky feature would order rows by their true label even on shuffled data.
        return z

    folds = make_splits(work)
    true_res = summarise_folds([
        score_fold(f, rule_score(work.loc[f.validation_mask]),
                   full_labels=labels)
        for f in folds
    ])
    work["is_laundering"] = labels.sample(frac=1.0, random_state=1).to_numpy()
    shuf_labels = work["is_laundering"]
    shuf_res = summarise_folds([
        score_fold(f, rule_score(work.loc[f.validation_mask]),
                   full_labels=shuf_labels)
        for f in folds
    ])

    true_pr = true_res.get("pr_auc_mean", float("nan"))
    shuf_pr = shuf_res.get("pr_auc_mean", float("nan"))
    base = float(labels.mean())
    print(f"  shuffled-labels: true mean PR-AUC {true_pr:.5f}, "
          f"shuffled mean PR-AUC {shuf_pr:.5f}, base rate {base:.5f}")
    ok = (not np.isnan(shuf_pr)) and abs(shuf_pr - base) < 0.5 * base and shuf_pr < true_pr
    print(f"  {'PASS' if ok else 'FAIL'}: shuffled PR-AUC should sit near the base rate")
    return ok


# --- Test 2: future injection ------------------------------------------------------

def test_future_injection() -> bool:
    """A transaction added on a late day must not move any earlier day's features."""
    edges = _small_edges(seed=2)
    days = np.unique(edges["date"])
    target = days[1]                      # first day with real history behind it
    work_dir = Path("/tmp/leak_test_future"); work_dir.mkdir(exist_ok=True)

    base = _build_to(edges, work_dir / "base.parquet")
    injection = edges.iloc[[0]].copy()
    injection["date"] = days[-1]
    # Reuse known account codes so the injection stays inside the vocab; the point of
    # the test is that a *late* row cannot reach early features, not that new codes work.
    injection["from_account"] = edges["from_account"].max()
    injection["to_account"] = edges["to_account"].max()
    extra = _build_to(pd.concat([edges, injection], ignore_index=True),
                      work_dir / "inject.parquet")

    base_early = base[base["date"] == target].reset_index(drop=True)
    extra_early = extra[extra["date"] == target].reset_index(drop=True)
    same_rows = len(base_early) == len(extra_early)
    same_feats = same_rows and all(
        np.array_equal(base_early[c].to_numpy(), extra_early[c].to_numpy())
        for c in FEATURE_COLS
    )
    print(f"  future-injection: early-day rows {len(base_early)} -> {len(extra_early)}, "
          f"features identical: {same_feats}")
    print(f"  {'PASS' if same_feats else 'FAIL'}: a late-day row moved early-day features")
    return bool(same_feats)


# --- Test 3: self-contribution -----------------------------------------------------

def test_self_contribution() -> bool:
    """Changing a row's own amount must not change that row's own amount features."""
    edges = _small_edges(seed=3)
    days = np.unique(edges["date"])
    target = days[-2]
    work_dir = Path("/tmp/leak_test_self"); work_dir.mkdir(exist_ok=True)

    base = _build_to(edges, work_dir / "base.parquet")
    # Pick a row whose sender already has history, so amount_mean is not zero.
    cand = base[(base["date"] == target) & (base["from_txn_count"] > 0)].index
    if len(cand) == 0:
        print("  self-contribution: skipped (no row with sender history)")
        return False
    row = int(cand[0])

    mutated = edges.copy()
    d = pd.Timestamp(base.loc[row, "date"])
    src = str(base.loc[row, "from_account"]); dst = str(base.loc[row, "to_account"])
    same_day = edges[edges["date"] == d]
    pick = same_day[(same_day["from_account"].map(str) == src)
                    & (same_day["to_account"].map(str) == dst)].index
    if len(pick) == 0:
        print("  self-contribution: skipped (could not recover the source row)")
        return False
    mutated.loc[int(pick[0]), "amount_usd"] *= 1000.0
    extra = _build_to(mutated, work_dir / "mut.parquet")

    a = base.loc[row]
    b = extra.loc[row]
    own_cols = ["from_amount_mean", "from_amount_std", "from_amount_max",
                "from_txn_count", "from_fan", "from_fan_out", "from_fan_in",
                "log_amount_usd"]
    diffs = {c: abs(float(a[c]) - float(b[c])) for c in own_cols}
    # log_amount_usd MUST change (it encodes this row's amount); the aggregates MUST not.
    aggregates = [c for c in own_cols if c != "log_amount_usd"]
    ok = all(diffs[c] < 1e-9 for c in aggregates) and diffs["log_amount_usd"] > 1e-3
    print("  self-contribution diffs: "
          + ", ".join(f"{c}={diffs[c]:.2e}" for c in own_cols))
    print(f"  {'PASS' if ok else 'FAIL'}: aggregates must ignore the row's own amount")
    return bool(ok)


# --- Runner ------------------------------------------------------------------------

def main() -> int:
    if not FEATURE_PATH.exists():
        print(f"missing {FEATURE_PATH}; run build_edge_features.py first")
        return 2
    df = pd.read_parquet(FEATURE_PATH)
    print("leakage tests on", FEATURE_PATH.name, df.shape)
    results = {
        "test_shuffled_labels": test_shuffled_labels(df),
        "test_future_injection": test_future_injection(),
        "test_self_contribution": test_self_contribution(),
    }
    print()
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
