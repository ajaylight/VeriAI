from pydantic import BaseModel, Field

class ClaimExtractionRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=12000, description="Answer to split into candidate claims")

class ClaimItem(BaseModel):
    claim_id: int
    text: str
    start_char: int
    end_char: int

class ClaimExtractionResponse(BaseModel):
    total_claims: int
    claims: list[ClaimItem]

class EvidenceSearchRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2000, description="Claim to search evidence for")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of evidence sentences")
