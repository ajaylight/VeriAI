# VeriAI roadmap

## Milestone 1 — Claim extraction
- Sentence-level candidate claims, order, and original offsets.
- API: POST /claims/extract.

## Milestone 2 — FEVER ingestion
- Read official JSONL safely; preserve IDs, labels, and evidence references.
- Inspect label counts and samples.
- APIs: GET /fever/status, /fever/stats, /fever/sample.

## Milestone 3 — Evidence retrieval
- Obtain/index the FEVER Wikipedia corpus.
- Implement BM25 and evaluate Recall@k.
- Add dense semantic retrieval and compare.

## Milestone 4 — Claim verification
- Pair retrieved passages with claims.
- NLI classification: SUPPORTS / REFUTES / NOT ENOUGH INFO.
- Tune thresholds on development data only.

## Milestone 5 — Application
- Source-linked explanations, transparent abstention, React interface.

## Milestone 6 — Evaluation
- Compare lexical baseline, dense retrieval + NLI, hybrid retrieval + NLI.
- Report macro-F1, per-class precision/recall, evidence Recall@k, and latency.
