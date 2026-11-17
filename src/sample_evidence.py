"""Stage 5 deliverable: build, validate and measure a sample of evidence packages.

Produces 20 packages for the Stage 5 note's acceptance line - "packages are
deterministic, schema-valid and under a size limit" - and writes them to
``results/evidence/`` alongside a summary.

Sample design
-------------
* Days 2022-09-04..2022-09-10 are the only days with calibrated thresholds, so 2-3
  rows are drawn from each (20 rows over 7 days) rather than letting one day dominate.
* Within a day the label is split 10 alerting / 10 not by ``score >= threshold@200``,
  so the model block is exercised on both sides of the cut rather than trivially null.
* Selection is deterministic (fixed seed on the sorted candidate rows), so re-running
  produces the same sample.

Validation
----------
Each package is checked three ways: JSON Schema (``docs/evidence_schema.json``),
serialised size under ``MAX_PACKAGE_BYTES``, and determinism (the same row is built
twice and the JSON compared byte-for-byte). A non-zero exit code means at least one
check failed.

Run from the project root with the learn_env interpreter::

    /home/dan/miniconda3/envs/learn_env/bin/python src/sample_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.evidence as ev  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "evidence"
SCHEMA_PATH = PROJECT_ROOT / "docs" / "evidence_schema.json"
SCORES_PATH = ev.SCORES_PATH
THRESHOLDS_PATH = ev.THRESHOLDS_PATH

N_SAMPLES = 20
BUDGET = ev.DEFAULT_BUDGET
# A package is a handful of neighbour rows and short paths; 64 KiB is generous for
# that shape and tight enough that accidentally embedding the feature vector's full
# edge list (or a raw CSV row) would fail loudly.
MAX_PACKAGE_BYTES = 64 * 1024
SEED = 42
DETERMINISM_SAMPLES = 3  # re-build this many packages a second time to prove determinism


def _threshold_days() -> list[str]:
    t = pq.read_table(THRESHOLDS_PATH, columns=["day", "budget"])
    t = t.filter(pc.equal(t["budget"], BUDGET))
    return sorted({str(z)[:10] for z in t["day"].to_pylist()})


def _select_rows() -> list[int]:
    """Deterministic 20 rows: spread over threshold days, balanced on the alert cut."""
    days = _threshold_days()
    if not days:
        raise SystemExit(f"no thresholds for budget {BUDGET} in {THRESHOLDS_PATH}")

    scores = pq.read_table(SCORES_PATH, columns=["row", "score", "date"]).to_pandas()
    scores["day"] = scores["date"].map(lambda d: str(d)[:10])
    scores = scores[scores["day"].isin(days)].copy()

    thr = pq.read_table(THRESHOLDS_PATH)
    thr = thr.filter(pc.equal(thr["budget"], BUDGET)).to_pandas()
    thr_map = {str(d)[:10]: float(x) for d, x in zip(thr["day"], thr["threshold"])}
    scores["alert"] = scores.apply(
        lambda r: r["score"] >= thr_map[r["day"]], axis=1)

    rng = np.random.default_rng(SEED)
    # Round-robin over days so each day contributes 2 or 3 rows, then split the quota
    # evenly between alerting and non-alerting rows.
    per_day = [N_SAMPLES // len(days) + (1 if i < N_SAMPLES % len(days) else 0)
               for i in range(len(days))]
    half = N_SAMPLES // 2
    chosen: list[int] = []
    n_alert = 0
    for day, k in zip(days, per_day):
        pool = scores[scores["day"] == day]
        for is_alert in (True, False):
            want = k // 2 + (k % 2 if is_alert and n_alert < half else 0)
            want = min(want, max(0, half - n_alert)) if is_alert else want
            cand = np.sort(pool[pool["alert"] == is_alert]["row"].to_numpy())
            if cand.size == 0 or want <= 0:
                continue
            pick = rng.choice(cand, size=min(want, cand.size), replace=False)
            chosen.extend(int(x) for x in pick)
            n_alert += len(pick) if is_alert else 0
    # Backfill to N_SAMPLES from the remaining pool if a day was short on one side.
    if len(chosen) < N_SAMPLES:
        rest = np.sort(np.setdiff1d(scores["row"].to_numpy(), chosen))
        chosen.extend(int(x) for x in rng.choice(
            rest, size=N_SAMPLES - len(chosen), replace=False))
    return sorted(chosen)


def _serialise(pkg: dict) -> str:
    return json.dumps(pkg, sort_keys=True, default=str)


def main() -> int:
    import jsonschema

    schema = json.load(open(SCHEMA_PATH))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    rows = _select_rows()
    print(f"sampled {len(rows)} rows from {len(_threshold_days())} days at budget {BUDGET}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    sizes: list[int] = []
    n_alert = n_scored = 0
    for i, row in enumerate(rows):
        pkg = ev.build_evidence(row=row, budget=BUDGET)
        body = _serialise(pkg)
        size = len(body.encode("utf-8"))
        sizes.append(size)
        if size > MAX_PACKAGE_BYTES:
            print(f"  ! row {row}: {size} B exceeds {MAX_PACKAGE_BYTES} B")
            failures += 1
        errs = sorted(validator.iter_errors(pkg), key=lambda e: list(e.absolute_path))
        if errs:
            failures += 1
            print(f"  ! row {row}: {len(errs)} schema errors")
            for e in errs[:3]:
                print("     ", "/".join(map(str, e.absolute_path)), "-", e.message[:140])
        if i < DETERMINISM_SAMPLES:
            again = _serialise(ev.build_evidence(row=row, budget=BUDGET))
            if again != body:
                print(f"  ! row {row}: package is not deterministic")
                failures += 1
        m = pkg["model"]
        n_scored += m["raw_margin"] is not None
        n_alert += m["alert"] is True
        (OUT_DIR / f"package_{i:02d}_row{row}.json").write_text(body)
        print(f"  [{i:02d}] row {row:>8}  {pkg['transaction']['date'][:10]}"
              f"  {size:>6} B  alert={m['alert']}  rank={m['rank_in_day']}"
              f"  thr={m['threshold']}")

    summary = {
        "budget": BUDGET,
        "n_samples": len(rows),
        "days": _threshold_days(),
        "max_package_bytes": int(max(sizes)),
        "limit_bytes": MAX_PACKAGE_BYTES,
        "mean_package_bytes": int(np.mean(sizes)),
        "n_scored": n_scored,
        "n_alert": n_alert,
        "failures": failures,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmax {max(sizes)} B (limit {MAX_PACKAGE_BYTES}), mean {int(np.mean(sizes))} B; "
          f"{n_scored}/{len(rows)} scored, {n_alert} alerting; failures={failures}")
    print(f"wrote {len(rows)} packages + summary.json to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
