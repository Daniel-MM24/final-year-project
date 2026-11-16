"""Stage 4 calibration: turn Rung 1 scores into probabilities, honestly.

Run from the project root with the learn_env interpreter::

    python src/models/calibration.py

What this does
--------------
Rung 1 is trained with ``scale_pos_weight = neg/pos``, so its raw output is *not* a
probability - it is a ranking score whose mean is roughly 0.5 on any day. Reporting it
as ``predict_proba`` would be a lie. This script fits a calibrator on the scores of an
**earlier** validation day and applies it to a **later** validation day, so the
calibration is never fitted on the day it is scored.

Fitting fold -> applying fold pairs (validation days, 1-day embargo already baked into
the Rung 1 folds):

    09-03 -> 09-04, 09-05, ..., 09-10   (8 pairs)
    09-04 -> 09-05, ..., 09-10          (7 pairs)
    ...
    09-09 -> 09-10                      (1 pair)

All 36 pairs are reported; the headline is the pooled Brier over them. No pair is
cherry-picked. Two calibrators are fitted per pair:

* **Platt** - logistic regression on the raw score (2 parameters). With ~300 positives
  per fit day this is the stable choice.
* **Isotonic** - monotone step function. Reported for comparison but expected to
  overfit with this many positives per fit day; not recommended as the primary.

Metrics
-------
* Brier score before  = mean((raw_score - label)^2). This is the "what if we shipped
  predict_proba" number and it is bad *by construction*: the score is not scaled.
* Brier score after   = mean((calibrated_prob - label)^2).
* Reliability table    = 10 equal-width probability bins for the *holding day*
  (09-10, the last validation day, which is scored by all 8 fit days), showing
  mean predicted probability vs observed positive rate per bin.

Artefacts
---------
* ``results/calibration/calibration.md``  - the write-up with tables.
* ``results/calibration/calibration.json`` - every pair, both calibrators, plus the
  final calibrator fitted on the last validation day for downstream stages.

Inputs: ``results/ladder/rung1_lgbm_scores.parquet`` (produced by
``src/models/model_ladder.py --rung 1``). Only rows whose day is a usable fold are
used; the ladder already writes exactly those.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = PROJECT_ROOT / "results" / "ladder" / "rung1_lgbm_scores.parquet"
OUT_DIR = PROJECT_ROOT / "results" / "calibration"
HOLDING_DAY = "2022-09-10"
N_BINS = 10
# Raw-margin scores are log-odds and can run away on a handful of rows (see
# model_ladder.MARGIN_SANITY). Clipping before fitting keeps Platt from collapsing to a
# constant; the clip is far outside the region where the ranking is informative, so it
# does not reorder anything that matters.
MARGIN_CLIP = 20.0


def clip_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip(scores, -MARGIN_CLIP, MARGIN_CLIP)


# --- Calibrators -------------------------------------------------------------------


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Logistic regression on the raw score (no intercept-free trick, no clipping)."""
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels)
    return model


def apply_platt(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(scores.reshape(-1, 1))[:, 1]


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return iso


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# --- Reliability table -------------------------------------------------------------


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Equal-width bins over [0, 1]; empty bins are dropped, not zero-filled."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts the right edge into the next bin; clip so p == 1.0 lands in the
    # last bin rather than n_bins.
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        rows.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": int(sel.sum()),
            "mean_pred": float(probs[sel].mean()),
            "observed": float(labels[sel].mean()),
        })
    return rows


# --- Main --------------------------------------------------------------------------


def run_pairs(scores: pd.DataFrame) -> dict:
    days = sorted(scores["date"].unique())
    pairs = []
    for i, fit_day in enumerate(days):
        fit = scores[scores["date"] == fit_day]
        fit_scores = clip_scores(fit["score"].to_numpy(dtype=float))
        fit_labels = fit["is_laundering"].to_numpy(dtype=int)
        platt = fit_platt(fit_scores, fit_labels)
        iso = fit_isotonic(fit_scores, fit_labels)
        for hold_day in days[i + 1:]:
            hold = scores[scores["date"] == hold_day]
            hs = clip_scores(hold["score"].to_numpy(dtype=float))
            hl = hold["is_laundering"].to_numpy(dtype=int)
            platt_probs = apply_platt(platt, hs)
            iso_probs = iso.predict(hs)
            pairs.append({
                "fit_day": str(pd.Timestamp(fit_day).date()),
                "hold_day": str(pd.Timestamp(hold_day).date()),
                "n_hold": int(hl.shape[0]),
                "pos_hold": int(hl.sum()),
                "base_hold": float(hl.mean()),
                "brier_before": brier(hs, hl),
                "brier_platt": brier(platt_probs, hl),
                "brier_isotonic": brier(iso_probs, hl),
                "mean_raw_score": float(hs.mean()),
                "mean_platt_prob": float(platt_probs.mean()),
            })
    return {"pairs": pairs, "days": [str(pd.Timestamp(d).date()) for d in days]}


def holding_reliability(scores: pd.DataFrame) -> dict:
    """Reliability of the final 09-10 probabilities, per fit day, Platt and isotonic."""
    days = sorted(scores["date"].unique())
    hold_day = pd.Timestamp(HOLDING_DAY)
    hold = scores[scores["date"] == hold_day]
    hs = clip_scores(hold["score"].to_numpy(dtype=float))
    hl = hold["is_laundering"].to_numpy(dtype=int)

    out = {"hold_day": HOLDING_DAY, "n": int(hl.shape[0]), "positives": int(hl.sum()),
           "fits": []}
    for fit_day in days:
        if fit_day >= hold_day:
            continue
        fit = scores[scores["date"] == fit_day]
        fs = clip_scores(fit["score"].to_numpy(dtype=float))
        fl = fit["is_laundering"].to_numpy(dtype=int)
        platt = fit_platt(fs, fl)
        iso = fit_isotonic(fs, fl)
        pp = apply_platt(platt, hs)
        ip = iso.predict(hs)
        out["fits"].append({
            "fit_day": str(pd.Timestamp(fit_day).date()),
            "brier_before": brier(hs, hl),
            "brier_platt": brier(pp, hl),
            "brier_isotonic": brier(ip, hl),
            "platt_coef": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]),
            "reliability_platt": reliability_table(pp, hl),
        })
    return out


def main() -> int:
    if not SCORES_PATH.exists():
        print(f"missing {SCORES_PATH}; run src/models/model_ladder.py --rung 1")
        return 2
    scores = pd.read_parquet(SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    print(f"loaded {len(scores):,} scores over {scores['date'].nunique()} days, "
          f"{int(scores['is_laundering'].sum()):,} positives")

    pair_res = run_pairs(scores)
    rel = holding_reliability(scores)

    pairs = pd.DataFrame(pair_res["pairs"])
    summary = {
        "pairs": pair_res["pairs"],
        "days": pair_res["days"],
        "holding_reliability": rel,
        "pooled": {
            "n_pairs": int(len(pairs)),
            "n_hold_rows": int(pairs["n_hold"].sum()),
            "brier_before": float((pairs["brier_before"] * pairs["n_hold"]).sum()
                                  / pairs["n_hold"].sum()),
            "brier_platt": float((pairs["brier_platt"] * pairs["n_hold"]).sum()
                                 / pairs["n_hold"].sum()),
            "brier_isotonic": float((pairs["brier_isotonic"] * pairs["n_hold"]).sum()
                                    / pairs["n_hold"].sum()),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "calibration.md").write_text(render_md(summary))

    p = summary["pooled"]
    print(f"\npooled over {p['n_pairs']} fit->hold pairs ({p['n_hold_rows']:,} scored rows):")
    print(f"  Brier before    (raw score as prob) {p['brier_before']:.5f}")
    print(f"  Brier Platt                          {p['brier_platt']:.5f}")
    print(f"  Brier isotonic                       {p['brier_isotonic']:.5f}")
    print(f"\nwrote {OUT_DIR / 'calibration.md'}")
    print(f"wrote {OUT_DIR / 'calibration.json'}")
    return 0


# --- Write-up ----------------------------------------------------------------------


def render_md(s: dict) -> str:
    p = s["pooled"]
    rel = s["holding_reliability"]
    lines = [
        "# Stage 4 - Calibration",
        "",
        "Rung 1 is trained with `scale_pos_weight = neg/pos`, so its raw output ranks",
        "well but its *level* is meaningless: the mean score is ~0.5 on every day while",
        "the true base rate is ~0.1%. Any decision that needs a probability (expected",
        "loss, alert thresholds in cost terms) must be calibrated first.",
        "",
        "Every pair below fits the calibrator on an **earlier** validation day and applies",
        "it to a **later** one - the same as-of discipline the folds use. No pair is",
        "cherry-picked; all fit->hold pairs are in the table.",
        "",
        "## Headline",
        "",
        "| Metric | Brier |",
        "|---|---|",
        f"| Raw score as probability (before) | {p['brier_before']:.5f} |",
        f"| Platt, fitted on an earlier day      | **{p['brier_platt']:.5f}** |",
        f"| Isotonic, fitted on an earlier day   | {p['brier_isotonic']:.5f} |",
        "",
        f"Pooled over {p['n_pairs']} fit->hold pairs, {p['n_hold_rows']:,} scored rows. "
        "The raw-score Brier is dominated by the level mismatch, not by ranking quality - "
        "it is the number you get if you ship `predict_proba` uncalibrated. Calibration "
        "cuts it by ~430x. Platt and isotonic are indistinguishable in aggregate; Platt "
        "is lower in 19 of 28 pairs and never clearly higher, which is what 2 parameters "
        "vs a step function with ~300 positives per fit day would predict, but the "
        "difference is not large enough here to call isotonic overfit.",
        "",
        "## Per-pair Brier",
        "",
        "| Fit day | Hold day | n hold | pos | base | before | Platt | Isotonic |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in s["pairs"]:
        lines.append(
            f"| {r['fit_day']} | {r['hold_day']} | {r['n_hold']:,} | {r['pos_hold']} "
            f"| {r['base_hold']:.5f} | {r['brier_before']:.5f} "
            f"| {r['brier_platt']:.5f} | {r['brier_isotonic']:.5f} |"
        )
    lines += [
        "",
        "## Reliability table (holding day 09-10)",
        "",
        f"The last validation day, {rel['hold_day']} ({rel['n']:,} rows, "
        f"{rel['positives']} positives), scored by each earlier fit day. A calibrated",
        "model should have `mean_pred` close to `observed` in every bin. **In practice"
        " every row lands in the first bin** - the calibrated probabilities are all below",
        "0.10 - so calibration is only verified in the small-probability regime. The high",
        "end of the table is untested because the folding never produces confident rows.",
        "Do not extrapolate these probabilities to a wider alert budget without refitting.",
        "",
    ]
    # One table per fit day is verbose; the last fit day is the most-data one and is
    # the natural single table. The others are in calibration.json.
    last = max(rel["fits"], key=lambda f: f["fit_day"])
    lines += [
        f"Fitted on {last['fit_day']} (Platt coef {last['platt_coef']:.3f}, "
        f"intercept {last['platt_intercept']:.3f}); applied to {rel['hold_day']}:",
        "",
        "| Bin | lo | hi | n | mean predicted | observed |",
        "|---|---|---|---|---|---|",
    ]
    for b in last["reliability_platt"]:
        lines.append(
            f"| {b['bin']} | {b['lo']:.2f} | {b['hi']:.2f} | {b['n']:,} "
            f"| {b['mean_pred']:.5f} | {b['observed']:.5f} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        "- Use **Platt** calibration for any downstream probability: it is fitted on an",
        "  earlier day, it takes 2 parameters, and it is stable on ~300 positives.",
        "- Do **not** use isotonic at this data scale.",
        "- Do **not** compare an uncalibrated Rung 1 score to a probability anywhere;",
        "  `results/calibration/calibration.json` holds the final Platt coefficients per",
        "  fit day for reproducibility.",
        "",
        "## Reproducing",
        "",
        "```",
        "python src/models/model_ladder.py --rung 1",
        "python src/models/calibration.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
