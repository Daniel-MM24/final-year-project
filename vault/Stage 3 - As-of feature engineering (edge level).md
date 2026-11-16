TASK: Build a leakage-safe, versioned edge-level feature pipeline.

DESIGN RULE
For a transaction on calendar day D, every feature about its sender or receiver may use
only transactions dated strictly BEFORE D. This is stricter than a per-fold cutoff and is
required because an edge's own amount must never contribute to its own features.

FEATURES TO BUILD (all as-of, sender and receiver sides separately)
- Amount behaviour (USD): std, mean, max of amount_usd; count of prior transactions.
- Counterparty structure: fan_out, fan_in (distinct counterparties), fan ratio.
- Cross-bank fraction of prior activity.
- Account age: days since the account's first observed transaction, relative to the window
  start. Late-arriving accounts looked disproportionately laundering-linked; test whether
  this is real signal or only a late-window artifact before keeping it.
- 1-hop neighbour aggregates: mean and max of the counterparties' own as-of features.
- Transaction-level: log(amount_usd), payment_format, cross_bank flag.
- Velocity: build it from DISTINCT (from, to, amount) relationships, or raw counts, per the
  Stage 2 near-duplicate decision. Apply a 1-minute floor to any inter-transaction gap
  (timestamps are minute-granular; same-minute pairs give a zero denominator).
Do NOT build: structuring features, currency-mismatch or crossing features, round-number
features, bank_id or account_id as raw categoricals.

IMPLEMENTATION
- Reload transactions from the raw CSV and reapply the SAVED fx rates (fx_usd_rate.csv,
  fx_btc_daily_rate.csv). Assert 0 unconverted rows.
- Build per-day snapshots incrementally (cumulative aggregates), not by re-grouping the full
  history 17 times, so it scales to 6.9M rows. Report runtime and memory.
- Write features to eda_output/features/edge_features_vN.parquet with a version tag and a
  manifest (feature name, definition, as-of rule, null policy). Missing history is 0 plus an
  explicit has_history flag. Never leave NaN to be silently imputed.

LEAKAGE TESTS (required, must be automated)
- Shuffle test: permute the labels within each day; model PR-AUC must fall to about the base rate.
- Future-injection test: add a synthetic transaction dated after a validation day and assert
  that no validation feature changes.
- Self-contribution test: assert a row's own amount does not change its own features.

ACCEPTANCE
- All three tests pass; the manifest is complete; a README explains how to rebuild.