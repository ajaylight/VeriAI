from fastapi import FastAPI, HTTPException, Query
from app.schemas import ClaimExtractionRequest, ClaimExtractionResponse, ClaimItem
from app.services.claim_splitter import split_into_claims
from app.services.fever_loader import FeverDataError, get_samples, summarize_split

app = FastAPI(title="VeriAI API", description="Evidence-grounded, claim-level answer verification.", version="0.2.0")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "veriai-api", "milestone": "fever-ingestion"}

@app.post("/claims/extract", response_model=ClaimExtractionResponse)
def extract_claims(payload: ClaimExtractionRequest) -> ClaimExtractionResponse:
    try:
        extracted = split_into_claims(payload.answer)
        return ClaimExtractionResponse(total_claims=len(extracted), claims=[ClaimItem(**item.__dict__) for item in extracted])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Claim extraction failed: {exc}") from exc

@app.get("/fever/status")
def fever_status() -> dict[str, object]:
    from app.services.fever_loader import DEFAULT_FEVER_DIR, SPLIT_FILES
    files = {name: {"filename": filename, "present": (DEFAULT_FEVER_DIR / filename).is_file()} for name, filename in SPLIT_FILES.items()}
    return {"data_directory": str(DEFAULT_FEVER_DIR), "files": files, "ready": all(item["present"] for item in files.values())}

@app.get("/fever/stats")
def fever_stats(split: str = Query(..., pattern="^(train|paper_dev)$")) -> dict[str, object]:
    try:
        return summarize_split(split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FeverDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/fever/sample")
def fever_sample(split: str = Query(..., pattern="^(train|paper_dev)$"), limit: int = Query(5, ge=1, le=20)) -> dict[str, object]:
    try:
        samples = get_samples(split, limit=limit)
        return {"split": split, "returned": len(samples), "samples": samples}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FeverDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
