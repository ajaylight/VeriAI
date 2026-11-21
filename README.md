# VeriAI

**Evidence-grounded AI answer verification using claim-level retrieval and semantic entailment.**

VeriAI splits an answer into candidate claims, retrieves relevant Wikipedia evidence, and classifies each claim as supported, contradicted, or insufficiently evidenced.

## Team
| Member | USN |
|---|---|
| Aryan Lokesh | 1BG24AI009 |
| Aditya Dhumale | 1BG24AI004 |
| Ajay Prakash | 1BG24AI005 |
| Sakshi | 1BG24CS149 |

## Setup (Windows / Anaconda Prompt)
From the repository root:

\`\`\`cmd
conda activate veriai
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`

API docs: http://127.0.0.1:8000/docs

## FEVER claim data
Place the labelled \`train.jsonl\` and \`paper_dev.jsonl\` files in \`data/fever/\`. See [data/fever/README.md](data/fever/README.md).

## Evidence retrieval index
Download the official \`wiki-pages.zip\` to \`data/fever/wiki-pages.zip\`. The indexer reads the JSONL members directly from the ZIP; it does not require extracting the whole corpus first. Build the FTS5 BM25 index from the \`backend\` directory:

\`\`\`cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip --max-pages 20000
\`\`\`

This creates a small development index for a smoke test. For the full corpus, omit \`--max-pages\`:

\`\`\`cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip
\`\`\`

The default index is \`data/index/fever_wiki_fts5.sqlite3\`. Dataset and generated index files are excluded from Git.

## NLI verification setup
The retrieval API works without the NLI packages. To enable claim verification, install the optional dependencies from \`backend/\`:

\`\`\`cmd
python -m pip install -r requirements-nli.txt
\`\`\`

VeriAI uses the pretrained [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) checkpoint. It was trained on MultiNLI, FEVER-NLI and ANLI. The first verification request downloads the model weights into the local Hugging Face cache; the weights are not committed to this repository. The service automatically uses CUDA when available and otherwise runs on CPU.

Optional model override:

\`\`\`cmd
set VERIAI_NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
\`\`\`

Check model/dependency state without downloading weights using \`GET /verification/status\`.

## Current API
- \`GET /health\`
- \`POST /claims/extract\`
- \`GET /fever/status\`
- \`GET /fever/stats?split=train\`
- \`GET /fever/sample?split=train&limit=3\`
- \`GET /evidence/status\`
- \`POST /evidence/search\`
- \`GET /verification/status\`
- \`POST /verification/verify\`

Example verification body:

\`\`\`json
{
  "claim": "Oliver Reed was a film actor.",
  "top_k": 5
}
\`\`\`

The verification endpoint retrieves evidence first, then evaluates each evidence sentence as the premise and the claim as the hypothesis. NLI entailment maps to \`SUPPORTS\`, contradiction maps to \`REFUTES\`, and neutral maps to \`NOT ENOUGH INFO\`. The current baseline chooses the strongest non-neutral evidence pair; if all retrieved pairs are neutral, it returns \`NOT ENOUGH INFO\`. This aggregation strategy must be evaluated and may be improved for multi-sentence evidence. Returned confidence is an uncalibrated model softmax score, not a calibrated probability of factual correctness.

## Tests
From \`backend/\`:

\`\`\`cmd
python -m pytest -q
\`\`\`

Unit tests do not download the pretrained NLI checkpoint. The first live verification request requires the optional NLI dependencies and internet access for the initial model download.

## Important evaluation notes
\`NOT ENOUGH INFO\` means the available evidence is insufficient; it does not mean the claim is false. FEVER's Wikipedia evidence annotations are sentence references (page ID/title and sentence number). Retrieval evaluation should distinguish records with non-empty annotation sets from records that contain an actual Wikipedia sentence reference.

Keep retrieval metrics, NLI metrics and end-to-end verification metrics separate. A test using gold FEVER evidence measures NLI with oracle evidence; it is not an end-to-end retrieval-and-verification score.

## Current implementation status
- Claim extraction: implemented.
- FEVER JSONL inspection: implemented.
- Full-corpus BM25 retrieval with title-aware subject-page prioritization: implemented.
- Three-way NLI verifier and API integration: implemented; live model evaluation pending.
- Dense retrieval, end-to-end evaluation and frontend: future milestones.
