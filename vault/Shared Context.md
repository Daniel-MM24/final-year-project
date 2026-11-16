
You are working on an anti-money-laundering (AML) project built on the IBM AMLSim "LI-Small" synthetic dataset. Treat the following as established facts and hard rules. 

**DATA**

- data/LI-Small_Trans.csv: 6,924,049 transactions over 17 days (2022-09-01 to 2022-09-17). The raw header has TWO columns both named "Account" (sender, receiver). Pandas renames the second to "Account.1". Never assume the names; normalise them. 
- data/LI-Small_accounts.csv: 712,684 unique accounts (4 duplicate rows). Columns include Bank Name, Bank ID, Account Number, Entity ID, Entity Name. There is NO account-type or country column. 
- Edge label: "Is Laundering". 3,565 positives, about 0.0515% (roughly 1 in 1,941). - Referential integrity is clean: 0 orphan accounts; 6,781 dormant accounts (in the accounts file, never transact). 705,903 accounts are active. 
- Artifacts already saved in eda_output/: fx_usd_rate.csv, fx_btc_daily_rate.csv, fx_pair_rates.csv, account_features.parquet, baseline_lightgbm_summary.json. 
- 
**HARD RULES** (each one was learned from a real bug in this project) 
1. Count distinct things with .unique()/.nunique(), never len(). pd.Index.union on non-unique inputs silently returns a multiset (it reported 10.7M "accounts" vs the true 705,903). 
2. Amounts span 15 currencies. Never sum, average or threshold raw amount_paid. Use amount_usd / amount_received_usd, rebuilt from the SAVED rate tables, not a re-fit. 
3. Category-dtype columns: .map() keeps categorical dtype, so call .astype(float) before arithmetic. groupby on categoricals needs observed=True. Aggregating a categorical column into sets needs .astype(str) first. 
4. LEAKAGE: any feature computed over the whole file leaks future behaviour into earlier folds. Every feature must be computed as-of a cutoff using only earlier data. (Measured: one fold dropped from PR-AUC 1.0 to 0.78 once this was fixed.) 
5. Validation folds with fewer than 50 rows, or only one class, give a meaningless PR-AUC (a single-class fold returns a trivial 1.0). Skip them and say so. Never average them in. (An honest mean of 0.041 had been inflated to 0.616 by six such folds.) 
6. Never use raw account_id, or any ID-derived feature, as model input. The accounts with prefix 10042B are simulator-injected hubs; a model would memorise them. 
7. Evaluate with PR-AUC and recall at a fixed alert budget. Never accuracy or ROC-AUC. 
8. Evaluate with rolling-origin splits and a 1-day embargo, never random splits. 
9. The data is synthetic. Do not present any result as evidence about real-world performance, and do not invent real-world facts (regulations, case histories). 

**KNOWN FINDINGS** 
- Structuring near $10k/$5k/$3k: ruled out (below/above ratio shrinks as the band narrows). 
- Per-leg cross-currency flag: 0 laundering across about 98.9k FX legs. Account-level currency crossing exists on only 11 of 6,693 laundering-linked accounts and had zero model importance. Do not build features on it. 
- Bitcoin is effectively pegged (about 0.2% to 3% variation depending on sample); not a real volatility signal. - "Reinvestment" payments (about 650k rows, 0% laundering) stop after roughly day 6.
- About 80% of rows are near-duplicates on (from, to, amount): recurring-payment behaviour. Unexplained; do not build raw-count velocity features until it is understood. 
- Days 11-17 carry very few transactions and almost no new accounts; folds there are unusable. - Strongest node-level signals so far: out_amount_std, in_amount, fan_in, cross-bank fraction. The node-level baseline (target is_laundering_node, a weak derived label) scored mean PR-AUC 0.0408 over 4 usable folds (about 2-4x lift over base rate). 
- Outlier bank "Willows Thrift" and a typo'd bank "Crytpo Bank #57" have elevated rates. WORKING STYLE - Verify before claiming. Run the code. If a number is not measured, say it is not measured. - Report negative results plainly. Do not tune until a number looks good. 
- Put outputs in a named folder and write a short README of what was produced. - If a rule above conflicts with the task, stop and ask rather than quietly breaking it.