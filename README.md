# VeriAI

**Evidence-grounded AI answer verification using claim-level retrieval and semantic entailment.**

VeriAI splits an answer into candidate claims, searches an evidence collection, and will classify each claim as supported, contradicted, or insufficiently evidenced.

## Team
| Member | USN |
|---|---|
| Aryan Lokesh | 1BG24AI009 |
| Aditya Dhumale | 1BG24AI004 |
| Ajay Prakash | 1BG24AI005 |
| Sakshi | 1BG24CS149 |

## Setup (Windows / Anaconda Prompt)
From the repository root:

```cmd
conda activate veriai
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## FEVER claim data
Place the labelled `train.jsonl` and `paper_dev.jsonl` files in `data/fever/`. See [data/fever/README.md](data/fever/README.md).

## Evidence retrieval index
Download the official `wiki-pages.zip` to `data/fever/wiki-pages.zip`. The indexer reads the JSONL members directly from the ZIP; it does not require extracting the whole corpus first. Build the FTS5 BM25 index from the `backend` directory:

```cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip --max-pages 20000
```

This creates a small development index for a smoke test. For the full corpus, omit `--max-pages`:

```cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip
```

The default index is `data/index/fever_wiki_fts5.sqlite3`. Check free disk space first; the corpus is large and the generated index may require several GB. Dataset and index files are excluded from Git.

## Current API
- `GET /health`
- `POST /claims/extract`
- `GET /fever/status`
- `GET /fever/stats?split=train`
- `GET /fever/sample?split=train&limit=3`
- `GET /evidence/status`
- `POST /evidence/search`

Example search body:

```json
{
  "claim": "Oliver Reed was a film actor.",
  "top_k": 5
}
```

## Tests
From `backend/`:

```cmd
python -m pytest -q
```

## Important evaluation notes
`NOT ENOUGH INFO` means the available evidence is insufficient; it does not mean the claim is false. FEVER's Wikipedia evidence annotations are sentence references (page ID/title and sentence number). For this reason, evaluation should distinguish records with non-empty annotation sets from records that contain an actual Wikipedia sentence reference.

## Current implementation status
- Claim extraction: implemented and previously checked through Swagger.
- FEVER JSONL inspection: implemented.
- BM25 sentence retrieval: implemented using SQLite FTS5; build the local index before searching.
- NLI claim verification, dense retrieval, and the frontend: next milestones.
