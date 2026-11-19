# VeriAI

**Evidence-grounded AI answer verification using claim-level retrieval and semantic entailment.**

VeriAI breaks an answer into factual claims, searches a defined evidence collection, and checks whether retrieved evidence supports, contradicts, or is insufficient to verify each claim.

## Team
| Member | USN |
|---|---|
| Aryan Lokesh | 1BG24AI009 |
| Aditya Dhumale | 1BG24AI004 |
| Ajay Prakash | 1BG24AI005 |
| Sakshi | 1BG24CS149 |

## Milestones
1. Sentence-level claim extraction.
2. FEVER dataset ingestion and inspection.
3. BM25 and dense semantic evidence retrieval.
4. Natural Language Inference (NLI) verification.
5. Claim-level explanations and source references.
6. React frontend and evaluation dashboard.
7. Baseline comparisons and held-out evaluation.

## Windows setup (Anaconda Prompt)
From repository root:
```cmd
conda create -n veriai python=3.11 -y
conda activate veriai
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs.

## FEVER data
Download labelled files to `data/fever/`:
- https://fever.ai/download/fever/train.jsonl
- https://fever.ai/download/fever/paper_dev.jsonl

See [data/fever/README.md](data/fever/README.md). Dataset files are excluded from Git. The larger Wikipedia corpus is deferred until retrieval development.

## API
- `GET /health`
- `POST /claims/extract`
- `GET /fever/status`
- `GET /fever/stats?split=train`
- `GET /fever/sample?split=train&limit=3`

## Tests
From `backend/`: `pytest -q`

## Evaluation discipline
Preserve FEVER claim IDs, labels, and evidence annotations. Use labelled splits for measured evaluation; do not treat the unlabelled official test split as ground truth. `NOT ENOUGH INFO` means available evidence is insufficient, not that the claim is false.

## Current status
- Claim extraction: manually verified through Swagger.
- FEVER loader and inspection endpoints: implemented; download JSONL files locally to activate them.
- Evidence retrieval and NLI: planned, not yet implemented.
