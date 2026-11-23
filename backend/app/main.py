import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ClaimExtractionRequest,
    ClaimExtractionResponse,
    ClaimItem,
    EvidenceSearchRequest,
    VerificationRequest,
)
from app.services.claim_splitter import split_into_claims
from app.services.fever_loader import FeverDataError, get_samples, summarize_split
from app.services.nli_verifier import NliModelError, NliVerifier
from app.services.wiki_index import WikiIndexError, index_status, search_index

app = FastAPI(
    title="VeriAI API",
    description="Evidence-grounded, claim-level answer verification.",
    version="0.4.0",
)

# Permit the local Vite development and production-preview servers to call
# this API from the browser. Additional trusted origins can be supplied as a
# comma-separated VERIAI_CORS_ORIGINS environment variable.
configured_origins = [
    origin.strip()
    for origin in os.getenv("VERIAI_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
allowed_origins = sorted(set(configured_origins + [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

_nli_verifier = NliVerifier()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "veriai-api",
        "milestone": "nli-verification",
    }


@app.post("/claims/extract", response_model=ClaimExtractionResponse)
def extract_claims(payload: ClaimExtractionRequest) -> ClaimExtractionResponse:
    extracted = split_into_claims(payload.answer)
    claims = [ClaimItem(**item.__dict__) for item in extracted]
    return ClaimExtractionResponse(total_claims=len(claims), claims=claims)


@app.get("/fever/status")
def fever_status() -> dict[str, object]:
    from app.services.fever_loader import DEFAULT_FEVER_DIR, SPLIT_FILES

    files = {
        name: {
            "filename": filename,
            "present": (DEFAULT_FEVER_DIR / filename).is_file(),
        }
        for name, filename in SPLIT_FILES.items()
    }
    return {
        "data_directory": str(DEFAULT_FEVER_DIR),
        "files": files,
        "ready": all(item["present"] for item in files.values()),
    }


@app.get("/fever/stats")
def fever_stats(split: str = Query(..., pattern="^(train|paper_dev)$")) -> dict[str, object]:
    try:
        return summarize_split(split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FeverDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/fever/sample")
def fever_sample(
    split: str = Query(..., pattern="^(train|paper_dev)$"),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, object]:
    try:
        samples = get_samples(split, limit=limit)
        return {"split": split, "returned": len(samples), "samples": samples}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FeverDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/evidence/status")
def evidence_status() -> dict[str, object]:
    return index_status()


@app.post("/evidence/search")
def evidence_search(payload: EvidenceSearchRequest) -> dict[str, object]:
    try:
        results = search_index(payload.claim, top_k=payload.top_k)
        return {"claim": payload.claim, "total_results": len(results), "results": results}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WikiIndexError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/verification/status")
def verification_status() -> dict[str, object]:
    """Check NLI dependencies and lazy model-load state without downloading weights."""
    return _nli_verifier.status()


@app.post("/verification/verify")
def verify_claim(payload: VerificationRequest) -> dict[str, object]:
    """Retrieve evidence and classify the claim as SUPPORTS, REFUTES or NOT ENOUGH INFO."""
    try:
        evidence = search_index(
            payload.claim,
            top_k=payload.top_k,
            operator="OR_FILTERED",
        )
        result = _nli_verifier.verify(payload.claim, evidence)
        return {
            "claim": payload.claim,
            "retrieval_mode": "subject_page_first_bm25_or_filtered",
            "retrieved_evidence_count": len(evidence),
            **result,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WikiIndexError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except NliModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
