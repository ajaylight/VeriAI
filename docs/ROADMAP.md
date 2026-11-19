# VeriAI roadmap

## Milestone 1 — Claim extraction
Sentence-level candidate claims, order, and original character offsets.

## Milestone 2 — FEVER ingestion
Read official labelled JSONL files; preserve claim IDs, labels, and evidence annotations.

## Milestone 3 — BM25 evidence retrieval
- Stream the FEVER wiki-pages JSONL members from the official ZIP archive.
- Build a local SQLite FTS5 index with one row per numbered Wikipedia sentence.
- Retrieve top-k sentence candidates with page and sentence identifiers.
- Evaluate evidence Recall@k using annotated examples from the labelled development split.

## Milestone 4 — Claim verification
Pair retrieved passages with claims and run an NLI model for SUPPORTS / REFUTES / NOT ENOUGH INFO. Tune decision thresholds on development data only.

## Milestone 5 — User-facing application
Add source-linked explanations, transparent abstention, and a React interface.

## Milestone 6 — Evaluation
Compare lexical BM25, dense retrieval + NLI, and hybrid retrieval + NLI. Report macro-F1, per-class precision/recall, evidence Recall@k, and latency.
