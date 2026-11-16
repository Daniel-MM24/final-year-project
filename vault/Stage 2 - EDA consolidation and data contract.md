TASK: Close the open items from the EDA and produce a data contract that every later
stage can rely on. The EDA notebooks exist (IBM_AML_LI_Small_EDA_1_.ipynb); do not rewrite
them, extend them.

OPEN ITEMS TO RESOLVE (measure each, do not assume the answer)
1. Near-duplicates: about 80% of rows share (from, to, amount) with another row. Determine
   whether these are recurring payments (fixed amount, regular spacing) or simulator
   padding. Report: distribution of repeat counts, spacing of timestamps within a repeating
   group, the laundering rate among repeated vs non-repeated rows, and whether $0.00 amounts
   are involved. Recommend whether velocity features should count raw transactions or
   distinct (from, to, amount) relationships.
2. ID-prefix scan: the earlier scan used 6-character prefixes only. Repeat at 4 and 8
   characters, ranking prefixes by behavioural anomaly (fan-out lift, laundering rate), not
   by frequency alone. List every prefix family to be excluded from ID-derived features.
3. Bank check: re-run the bank analysis with a fuzzy match so that the typo "Crytpo Bank
   #57" is caught. For banks with elevated laundering rates, test with a binomial test
   against the base rate (not a flat count cutoff), and check whether the effect is spread
   across a bank's accounts or driven by a handful of injected ones.
4. Entity ID: "Entity ID" (beneficial owner) was never analysed. Report how many accounts
   map to each entity, whether laundering concentrates in multi-account entities, and
   whether entities transact among their own accounts.
5. Late-window behaviour: build the per-day table (rows, positives, new accounts,
   positive rate). Identify which days are unusable for evaluation and justify the cut.

DELIVERABLES
- docs/data_contract.md: schema after normalisation, canonical column names, USD rule,
  excluded ID prefixes, unusable days, label definitions, and every known data artifact.
- notebooks/eda_addendum.ipynb answering the five items, with outputs saved.

ACCEPTANCE
- Each of the five items ends with a one-line decision (adopt / drop / investigate) and the
  evidence for it. Anything you could not determine is listed as unknown, not guessed.