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
