# Data contract (Stage 2)

Every stage reads data through this contract. If code disagrees with this file, the code
is wrong. Measurements cite the script that produced them; rerun the script rather than
trusting the number here.

**All data is synthetic.** Nothing here is evidence about the real world.

---

## 1. Source files

| File | Rows | Grain |
|---|---|---|
| `data/LI-Small_Trans.csv` | 6,924,049 | one transaction (edge) |
| `data/LI-Small_accounts.csv` | 712,688 | one account |

### 1.1 Duplicate header names

`LI-Small_Trans.csv` has **two columns named `Account`** (sender and receiver). pandas
renames the second to `Account.1`. Never rely on position; rename on load.

Canonical names, used everywhere from here on:

| Raw | Canonical |
|---|---|
| `Timestamp` | `timestamp` |
| `From Bank` / `Account` (1st) | `from_bank` / `from_account` |
| `To Bank` / `Account.1` | `to_bank` / `to_account` |
| `Amount Paid` / `Payment Currency` | `amount_paid` / `currency_paid` |
| `Amount Received` / `Receiving Currency` | `amount_received` / `currency_received` |
| `Payment Format` | `payment_format` |
| `Is Laundering` | `is_laundering` |

`src/measure_daily_table.py::load_transactions` implements this.

### 1.2 Account IDs are hex strings

IDs look like `8000ECA90` — **9 characters, hex, no leading zeros stripped**. Read them
as `str`, never `int`. Treating them as integers is a lossless-looking bug that breaks
joins.

---

## 2. Time

- Coverage is **2022-09-01 .. 2022-09-17** (17 days), timestamps are wall-clock, no timezone.
- Derive `date = timestamp.normalize()`; never bucket by string.
- Only **09-01 .. 09-10** are usable for evaluation (section 5).

---

## 3. Amounts and USD

Rule 2 of the always-on rules: **never compare raw amounts across currencies.**

### 3.1 The only permitted conversion path

```
amount_usd(currency, amount) =
    amount * usd_rate[currency]                      if currency != "Bitcoin"
    amount * btc_usd_rate[date(timestamp)]           if currency == "Bitcoin"
```

- `usd_rate` from `eda_output/fx_usd_rate.csv` — 14 currencies, `US Dollar` = 1.0.
- `btc_usd_rate` from `eda_output/fx_btc_daily_rate.csv` — one rate per day 09-01..09-17.

Both legs of a transaction are converted independently:
`amount_paid_usd = amount_usd(currency_paid, amount_paid)` and likewise for received.

### 3.2 Do not use `fx_pair_rates.csv` for conversion

It is the **raw, unfiltered** pair table (`src` note: MIN_N=30 is a reliability cutoff,
but the saved file still contains 106 of 195 rows with n < 30, including rows with n=1
and std=NaN). Its direct Bitcoin rates are placeholders, not observations:

| Pair | median | n | Problem |
|---|---|---|---|
| `Euro -> Bitcoin` | 1.000000e-04 | 228 | round placeholder |
| `UK Pound -> Bitcoin` | 1.000000e-04 | 143 | round placeholder |
| `US Dollar -> Bitcoin` | 8.42e-05 | 3156 | real (1/11877) |

Converting EUR-denominated Bitcoin through the pair table yields a rate wrong by ~7 orders
of magnitude. Fiat->fiat pairs are consistent with `fx_usd_rate` (all 13 `<currency> ->
US Dollar` rows agree to 1.0000), but they are not needed and add risk.

**Verification** (`sanity` of the saved tables, run in Stage 2): every currency pair in the
file resolves to a known rate; 0 pairs unresolved. Fiat USD rates match the pair graph
exactly.

### 3.3 Concrete effects

- **98.6% of rows are same-currency** (paid currency == received currency), so USD
  conversion rarely changes relative magnitudes within the file.
- **Bitcoin is 4.52% of rows** (313,276). Its laundering rate is 0.0351% vs 0.0515%
  overall — mildly depleted, no special treatment needed beyond the rate lookup.
- `amount_received` is *not* guaranteed equal to `amount_paid * rate`. Where they differ,
  the difference is a real (synthetic) fee/spread. Do **not** "correct" it.

### 3.4 Zero amounts

There are **no** rows with `amount_paid == 0` or `amount_received == 0`. The earlier EDA
open item about `$0.00` amounts does not apply; treat it as closed.

---

## 4. Identifiers: what may not be a model input

### 4.1 Raw account IDs

Rule 6: never use `account_id` as a model input. It is an arbitrary hex identifier with
no generalisable meaning, and using it lets the model memorise.

### 4.2 The fabricated `80`-prefix family

A width-4 prefix scan ranked prefixes by **behavioural anomaly** (fan-out, laundering
lift) rather than frequency (`src/eda_item2_prefixes.py`). Flagged prefixes — all starting
`80`:

```
8000 8001 8002 8003 8004 8006 8007 8008 800A 800C 8016
```

Of these, `8000` is anomalous for fan-out only (x3.3 median partners, laundering lift
x1.0); the rest are laundering-driven (x2.1 .. x5.8). The full list is in
`eda_output/prefix_flags.json`.

**Contract rule:** any account whose ID starts with `80` may not enter the model as an
identity-derived feature, and must not be used to construct counters that are then fed in
raw. Prefix features are excluded wholesale.

### 4.3 Width-8 prefixes are degenerate

Every account ID is exactly 9 characters. An 8-character prefix leaves 1 character
(16 values), so any prefix with >= 2 accounts is a structural chunk of the ID space, not a
bank or entity code. **Do not use 8-character prefixes.** The note asked for a scan at 4
and 8; 8 was examined and rejected on this ground, not skipped for convenience.

---

## 5. Unusable days

Measured in `src/measure_daily_table.py`. Validation days come from Stage 1's
`src/eval_protocol.py`, which already excludes these.

| Days | Rows/day | Positive rate | Verdict |
|---|---|---|---|
| 09-01, 09-02 | 1.52M, 1.03M | 0.020%, 0.034% | usable, but ramp-up |
| 09-03 .. 09-10 | 282K .. 891K | 0.041% .. 0.106% | **steady state** |
| 09-11 .. 09-17 | 3 .. 77 | 56% .. 71% | **excluded** |

09-11 onward is simulator tail-off: a handful of rows per day, almost all labelled. Never
validate or train on it as if it were representative. It may be *dropped* entirely.

09-01/02 hold **84% of all new-account first appearances** (599K of 703K on day 1). Any
feature depending on "history so far" is meaningless for them; Stage 1 flags the 09-03
fold as low-history for this reason.

---

## 6. Labels

### 6.1 Two label definitions, two base rates

| Unit | Positives | Base rate |
|---|---|---|
| **Edge** (`is_laundering` on a transaction) | 3,565 | **0.0515%** |
| **Node** (account touches >= 1 laundering edge) | 5,304 of 705,903 | **0.7514%** (measured: `src/reproduce_baseline.py`) |

A **14x** difference. Node labels smear onto clean counterparties: any account that
receives a laundered dollar is "laundering". The node base rate is therefore an artefact
of the label definition, not a fact about accounts.

**Contract rule:** the prediction unit is the **edge**. Node-level scores are an internal
diagnostic only, and the 0.0408 baseline is a node-level number (Stage 1 must reproduce
it only to prove the protocol, never as the project's headline).

### 6.2 Repeated transactions are padding, not recurrence

`(from, to, amount, currency)` repeats on 80.5% of rows — 2.21M distinct keys for 6.92M
rows. Measurements (`src/eda_item1_duplicates.py`):

| Signal | Value |
|---|---|
| Timing regularity (CV of gaps within a group) | median 0.56, p10 0.35 |
| Groups with regular spacing (CV < 0.01) | **0.0%** of 20k sampled |
| Laundering rate, `n=1` groups | **0.2037%** |
| Laundering rate, `n>=2` groups | **0.0146%** |
| Raw vs distinct count per account | **2.4x** (p99 7.2x) |

Real recurring payments (payroll, subscriptions) have regular spacing. These do not, and
repeated groups are **14x depleted** in laundering. They are simulator padding of the
benign majority.

**Contract rule:** velocity features count **distinct relationships**
(`(from_account, to_account)`), not raw row counts. Note that the same rule applies to
fan-out in the prefix scan. Amount is *not* part of the relationship key: the same two
parties transacting at different amounts are one relationship.

---

## 7. Banks

From `src/eda_item3_banks.py`. Bank attribution is on the **sending** account's bank only
(double-counting both sides inflates banks that appear on both ends of edges). 4 account
numbers appear in two banks each; those are dropped from bank attribution rather than
arbitrarily resolved.

### 7.1 `#N` suffixes are not a laundering signal

27,184 of 27,652 bank names carry a `#N` suffix (61.7% of accounts). After fuzzy grouping
(strip `#N`, lowercase, fold known typos) and a one-sided binomial test against the global
base rate with Bonferroni correction across 1,903 tested banks:

- **3 banks** are significantly above base rate (alpha_adj = 2.63e-05).
- **0%** of them have a `#N` name.

The "Crytpo Bank #57" style names in the earlier EDA are **noise, not actors**. Do not
build a `#N`-suffix feature.

### 7.2 Two real bank effects

| Bank | Edges | Lift | Spread (top 1% of accts) | Reading |
|---|---|---|---|---|
| **Willows Thrift** | 610,166 | x2.71 | 37% | genuine, broad |
| Bank of Philadelphia | 38,373 | x2.43 | 94% | a few accounts |
| National Bank of Columbus | 64,793 | x1.92 | 88% | a few accounts |

Willows Thrift is 8.8% of all edges and the only bank whose laundering signal is spread
across its accounts. If any bank identity is usable as a feature it is that one — but
bank identity must still be derived **as-of** (from transactions before the prediction
time), never from the whole file.

---

## 8. Entities

From `src/eda_item4_entities.py`. 224,931 entities, 712,684 accounts, mean 3.17 accounts
each; 510 entities hold 50+ accounts (150,240 accounts, 21% of the file).

### 8.1 Larger entities are *less* laundering-exposed

| Accounts per entity | Accounts | % accounts touching laundering |
|---|---|---|
| 1 | 145,348 | 1.349% |
| 2 | 38,960 | 0.765% |
| 3-5 | 118,852 | 0.644% |
| 6-10 | 181,467 | 0.584% |
| 11-50 | 77,817 | 0.574% |
| 50+ | 150,240 | 0.515% |

Monotonic, Spearman 0.088. Entity size is usable as a feature; do not assume the sign.

### 8.2 Intra-entity transactions are **negative** signal

889,289 edges (12.84% of matched edges) have sender and receiver in the same entity.
Their laundering rate is **0.0049%** vs 0.0583% for inter-entity edges — **0.08x**, i.e.
12.8x *less* likely.

This contradicts the classic detection intuition (moving money between your own accounts
is structuring). In this synthetic data it is the opposite. Build the feature if you
like, but **never hand-assert its direction**; let the model find it.

---

## 9. As-of discipline (applies to every derived feature)

Rule 4. Every feature for a prediction at time `t` may use only rows with
`timestamp < t` (minus the embargo, see Stage 1). In particular:

- entity size, bank fan-out, partner counts, velocity all change over time;
- computing any of them over the whole file leaks the future;
- `eda_output/account_features.parquet` is a **whole-file** table and is therefore
  unusable for temporal CV. It exists for EDA and for reproducing the old node baseline
  only. `baseline_lightgbm.py` warns about exactly this.

---

## 10. Known discrepancies to carry forward

| Discrepancy | Numbers | Status |
|---|---|---|
| Exact duplicate rows | 6,924,049 rows, 6,924,041 distinct | **8** exact duplicate rows; drop with `drop_duplicates()` over the full row. Do not dedupe on a column subset - that merges real distinct transactions and produced a bogus 6,787,653 figure in an early Stage 1 print |
| Account slots vs accounts file | 1,257,457 distinct account slots in transactions vs 712,684 in accounts file | many transacting accounts have no metadata; features must handle missing |
| Earlier "$0.00 amounts" concern | no zero amounts exist | closed (section 3.4) |
| Earlier "8 dup rows" | 8 | **confirmed correct** |

---

## 11. Quick reference

```python
from src.measure_daily_table import load_transactions   # canonical names, str IDs
# amounts:
#   usd  = pd.read_csv("eda_output/fx_usd_rate.csv", index_col=0)["usd_rate"]
#   btc  = pd.read_csv("eda_output/fx_btc_daily_rate.csv", index_col=0)["btc_usd_rate"]
# Never: fx_pair_rates.csv, raw account IDs as features, 80-prefixes, 8-char prefixes,
#        whole-file aggregates, node-level headline metrics, days 09-11..17.
```
