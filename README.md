# VeriAI

**Evidence-grounded claim verification using Wikipedia retrieval and natural language inference.**

VeriAI retrieves evidence for a submitted claim and classifies the relationship between the claim and the retrieved evidence as **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**.

## Team

| Member | USN |
|---|---|
| Aryan Lokesh | 1BG24AI009 |
| Aditya Dhumale | 1BG24AI004 |
| Ajay Prakash | 1BG24AI005 |
| Sakshi | 1BG24CS149 |

## System overview

1. A user submits a claim through the React interface or verification API.
2. The FastAPI backend retrieves relevant sentences from the FEVER Wikipedia corpus using SQLite FTS5/BM25 retrieval, including subject-page prioritization.
3. The DeBERTa NLI model evaluates each retrieved evidence sentence as the premise and the submitted claim as the hypothesis.
4. Entailment maps to **SUPPORTS**, contradiction maps to **REFUTES**, and neutral maps to **NOT ENOUGH INFO**.
5. The response includes the retrieved evidence, per-sentence NLI relations, the deciding evidence pair, and a representative evidence citation.

The verifier selects the strongest eligible non-neutral evidence pair to determine a verdict. If all eligible pairs are neutral, it returns **NOT ENOUGH INFO**. The API distinguishes:

- **decision_evidence**: the strongest NLI pair used to determine the verdict.
- **selected_evidence**: the earliest retrieved pair that agrees with the verdict, used as the representative citation.

The returned confidence value is an **uncalibrated softmax score**, not a calibrated probability that the claim is factually correct.

## Architecture

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React 19, Vite 6 | Claim input, API status, verdict and evidence display |
| API | FastAPI, Pydantic | Request validation and verification endpoints |
| Retrieval | SQLite FTS5, BM25 | Search the indexed FEVER Wikipedia sentences |
| Inference | PyTorch, Transformers, DeBERTa | Evidence-premise/claim-hypothesis NLI |
| Evaluation | FEVER paper_dev | Retrieval and end-to-end classification evaluation |

NLI checkpoint: [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

The first live verification request may download model weights into the local Hugging Face cache. Model weights, the FEVER source archives, and the generated SQLite index are not stored in this Git repository. The backend uses CUDA when available and otherwise runs inference on CPU.

## Data and index setup

Place the labelled FEVER files, including `train.jsonl` and `paper_dev.jsonl`, in `data/fever/`. Download the Wikipedia archive `wiki-pages.zip` to the same directory. See [data/fever/README.md](data/fever/README.md) for dataset details.

Build the SQLite FTS5 index from the `backend/` directory.

A small development index can be built with:

~~~cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip --max-pages 20000
~~~

For the full corpus, omit the page limit:

~~~cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip
~~~

The default index path is `data/index/fever_wiki_fts5.sqlite3`. The generated index is excluded from Git and must be built or supplied locally.

The validated full index used during development contained:

| Index statistic | Value |
|---|---:|
| Pages | 5,396,106 |
| Sentences | 25,247,890 |
| SQLite database size | 8,369,197,056 bytes (about 8.37 GB) |
| Schema version | 2 |

## Backend setup

The project was tested with Python 3.11.18 in a Conda environment named `veriai`.

From the repository root in Anaconda Prompt or PowerShell:

~~~cmd
conda activate veriai
cd backend
python -m pip install -r requirements-nli.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

The NLI requirements file includes the base backend requirements and the PyTorch/Transformers dependencies required for verification.

API documentation: http://127.0.0.1:8000/docs

Useful status endpoints:

- `GET /health`: API health.
- `GET /evidence/status`: evidence-index readiness and schema compatibility.
- `GET /verification/status`: NLI dependencies, model-loaded state and inference device.

## Frontend setup

Keep the backend running. Open a second terminal from the repository root:

~~~cmd
cd frontend
npm ci
npm run dev
~~~

Open http://127.0.0.1:5173.

To create a production build:

~~~cmd
npm run build
~~~

The frontend uses http://127.0.0.1:8000 as its default API base URL. See [frontend/README.md](frontend/README.md) for configuration details.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health |
| POST | `/claims/extract` | Extract candidate claims from answer text |
| GET | `/fever/status` | Check FEVER labelled-data files |
| GET | `/fever/stats` | Summarize a FEVER split |
| GET | `/fever/sample` | Return sample labelled claims |
| GET | `/evidence/status` | Check the Wikipedia evidence index |
| POST | `/evidence/search` | Retrieve evidence sentences |
| GET | `/verification/status` | Check NLI dependency/model status |
| POST | `/verification/verify` | Retrieve evidence and classify a claim |

Example verification request:

~~~json
{
  "claim": "Magic Johnson played for the Lakers.",
  "top_k": 5
}
~~~

The verification API accepts claims with 1–2,000 characters and a `top_k` value from 1 to 10.

## Automated tests

From the `backend/` directory:

~~~cmd
python -m pytest -q -W always
~~~

The latest local validation completed with **55 passed, 0 failed, and 0 warnings**. Unit tests use controlled test predictors where appropriate and do not need to download the pretrained model. Live verification requires the NLI dependencies, model weights, and a working local evidence index.

## End-to-end evaluation

The latest recorded end-to-end run is `reports/verification_paper_dev_holdout_100_v3.json`.

### Evaluation protocol

- Dataset split: FEVER `paper_dev`.
- Eligible labelled claims in the split: 9,999.
- Earlier baseline report claim IDs excluded: 30.
- Claims evaluated: 100, sampled with seed 2026 and stratified by label.
- Sample composition: 33 SUPPORTS, 33 REFUTES, 34 NOT ENOUGH INFO.
- Retrieval depth: top 5.
- Model: DeBERTa-v3-base-mnli-fever-anli.
- Device: CPU.
- Runtime: 728.17 seconds (about 12.1 minutes).

This is a **claim-ID-disjoint sample from the same FEVER paper_dev split**, not an independent external dataset or an entity-disjoint evaluation.

### Classification results

| Metric | Result |
|---|---:|
| Accuracy | **72.00%** |
| Macro precision | 74.17% |
| Macro recall | 72.01% |
| Macro-F1 | **72.27%** |

| Verdict | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUPPORTS | 87.50% | 63.64% | 73.68% | 33 |
| REFUTES | 75.00% | 81.82% | 78.26% | 33 |
| NOT ENOUGH INFO | 60.00% | 70.59% | 64.86% | 34 |

### Evidence retrieval results

Retrieval metrics were computed on the 66 evaluated claims with sentence-level gold evidence references.

| Metric | Result |
|---|---:|
| Claims with gold evidence | 66 |
| Gold sentence references | 102 |
| Gold references retrieved | 65 |
| Precision@5 | 19.70% |
| Sentence Recall@5 | 63.73% |
| Claim Hit Rate@5 | 80.30% |
| Complete evidence-set recall@5 | 72.73% |

Precision@5 is relatively low: 65 of the 330 returned evidence slots matched a gold sentence reference. Claim Hit Rate@5 is higher because it measures whether at least one gold sentence was retrieved for a claim. These metrics describe different aspects of retrieval and should not be conflated.

The evaluation reports are stored under `reports/`. The V1/V2/V3 holdout reports preserve the development history; V3 is the strongest recorded 100-claim result.

## Limitations and next steps

- The 100-claim holdout result is a limited sample from the same FEVER paper_dev split. It does not establish generalization to unrelated datasets, domains, or unseen entities.
- Retrieval precision@5 (19.70%) indicates that many returned sentence slots are not annotated gold evidence. Better ranking and evidence filtering remain potential improvements.
- NOT ENOUGH INFO has the lowest per-class F1 in this evaluation (64.86%).
- The NLI softmax score is uncalibrated and must not be presented as a factual probability.
- The current verdict aggregation is a baseline based on the strongest eligible non-neutral evidence pair. Conflicting, multi-hop, temporal, and multi-sentence claims need further evaluation.
- The recorded 100-claim run took about 12.1 minutes on CPU. This is a research prototype, not a measured production-throughput deployment.
- Dense retrieval and systematic comparisons against alternate retrieval architectures remain future work.

## Repository data policy

The following are generated or large local assets and should not be committed:

- FEVER source data under `data/fever/`, except its README.
- Generated SQLite index under `data/index/`.
- Frontend dependencies under `frontend/node_modules/`.
- Frontend production output under `frontend/dist/`.
- Local environment files and Python caches.

Keep the benchmark JSON reports in `reports/` so the reported metrics can be inspected and reproduced from their recorded claim IDs and settings.
