"""Initial sentence-level claim extraction; preserves original character offsets."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class ExtractedClaim:
    claim_id: int
    text: str
    start_char: int
    end_char: int

_ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "e.g.", "i.e.", " vs.")

def split_into_claims(answer: str) -> list[ExtractedClaim]:
    if not answer or not answer.strip():
        return []
    protected = answer
    replacements: dict[str, str] = {}
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        token = f"<ABBR{index}>"
        if abbreviation in protected:
            protected = protected.replace(abbreviation, token)
            replacements[token] = abbreviation
    claims: list[ExtractedClaim] = []
    pattern = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.DOTALL)
    cursor = 0
    for match in pattern.finditer(protected):
        fragment = match.group(0)
        if not fragment:
            continue
        restored = fragment
        for token, abbreviation in replacements.items():
            restored = restored.replace(token, abbreviation)
        left_trimmed = restored.lstrip()
        if not left_trimmed.strip():
            cursor = match.end()
            continue
        start = cursor + len(restored) - len(left_trimmed)
        text = left_trimmed.rstrip()
        claims.append(ExtractedClaim(len(claims) + 1, text, start, start + len(text)))
        cursor = match.end()
    if not claims and answer.strip():
        start = len(answer) - len(answer.lstrip())
        text = answer.strip()
        claims.append(ExtractedClaim(1, text, start, start + len(text)))
    return claims
