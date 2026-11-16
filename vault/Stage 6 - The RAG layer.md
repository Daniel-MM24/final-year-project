TASK: Build the retrieval layer in two parts. Keep them separate.

PART A: Text retrieval (typologies, policy, past cases)
- Corpus. Do NOT fabricate regulatory text or case histories. Build the pipeline around a
  corpus folder (corpus/) that the user fills with real public documents (for example FATF
  typology reports, FinCEN advisories, guidance on structuring, layering, mule accounts).
  Until that exists, create a small clearly-labelled SYNTHETIC set (each file header says
  "SYNTHETIC: for pipeline testing only") that describes the typologies actually present in
  this dataset: fan-out from hub accounts, cyclic transfers, high amount variability.
- Ingest: chunk by section (not fixed length), keep metadata (source, title, section, date,
  document type), and store the original text for exact quotation.
- Retrieval: hybrid (BM25 keyword plus dense embeddings) with a reranker. Return chunk text,
  source and a relevance score. Include a "no relevant passage found" outcome with a score floor.
- Build a retrieval test set of at least 30 queries paired with the passages that should come
  back (queries derived from alert evidence packages, for example "account fans out to many
  recipients in two days"). Report recall@5 and MRR.

PART B: Structured retrieval (graph and account history). DO NOT embed this.
- Expose exact, typed tools over the data, each returning JSON and each enforcing the as-of
  rule via a required "as_of" timestamp parameter:
  get_account_history(account, as_of), get_neighbours(account, as_of, hops, limit),
  find_cycles(account, as_of, max_len), get_alert_evidence(transaction_id),
  get_bank_stats(bank, as_of).
- Cap result sizes, and return an explicit "truncated" flag.
- Write tests that call each tool with an as_of earlier than the data and assert that nothing
  from the future is returned.

DELIVERABLES
- src/retrieval/text_index.py, src/retrieval/tools.py, tests, and the retrieval test set with
  its measured recall@5 and MRR. Document the embedding model and reranker used.

ACCEPTANCE
- Text retrieval metrics are reported on the test set; every tool passes the as-of test.
- Nothing in corpus/ that is not the user's own sourced material is presented as real.