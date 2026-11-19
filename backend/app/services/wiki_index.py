"""Streaming FEVER Wikipedia indexing and BM25 search using SQLite FTS5.

FTS5 stores one row per FEVER sentence. The Wikipedia archive can be read
straight from ZIP without extracting the ~7 GB JSONL corpus to disk.
"""
from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "fever_wiki_fts5.sqlite3"
TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)

class WikiIndexError(RuntimeError):
    """Raised for an unavailable or malformed search index."""

def tokenize_query(text: str, max_terms: int = 64) -> list[str]:
    """Return unique lexical terms accepted by the FTS5 query builder."""
    terms: list[str] = []
    seen: set[str] = set()
    for match in TOKEN_RE.finditer(text.casefold()):
        term = match.group(0)
        if len(term) < 2 or term in seen:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms

def build_fts_query(terms: list[str]) -> str:
    """Build a safe OR query; each term is quoted to avoid FTS operators."""
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

def parse_wiki_page(record: dict[str, Any]) -> Iterator[tuple[str, int, str]]:
    """Yield (page_id, sentence_id, sentence_text) from one FEVER page row."""
    page_id = record.get("id")
    lines = record.get("lines")
    if not isinstance(page_id, str) or not isinstance(lines, str):
        return
    for raw_line in lines.splitlines():
        sentence_id_raw, separator, sentence = raw_line.partition("\t")
        if not separator:
            continue
        try:
            sentence_id = int(sentence_id_raw)
        except ValueError:
            continue
        sentence = sentence.strip()
        if sentence:
            yield page_id, sentence_id, sentence

def create_index_schema(connection: sqlite3.Connection) -> None:
    """Create the FTS5 table and build metadata table."""
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE sentences USING fts5("
            "page_id UNINDEXED, sentence_id UNINDEXED, text, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as exc:
        raise WikiIndexError(
            "This Python SQLite build does not support FTS5. Install a current "
            "official Python 3.11 build or use a Python distribution with FTS5 enabled."
        ) from exc
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.commit()

def insert_sentence_batch(connection: sqlite3.Connection, rows: Iterable[tuple[str, int, str]]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    connection.executemany(
        "INSERT INTO sentences(page_id, sentence_id, text) VALUES (?, ?, ?)",
        payload,
    )
    return len(payload)

def search_index(query: str, top_k: int = 5, index_path: Path | None = None) -> list[dict[str, Any]]:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    path = index_path or DEFAULT_INDEX_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Wikipedia search index not found: {path}. Build it with the FEVER index command first.")
    terms = tokenize_query(query)
    if not terms:
        raise ValueError("The claim must contain at least one searchable word (2+ characters).")
    fts_query = build_fts_query(terms)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT page_id, CAST(sentence_id AS INTEGER) AS sentence_id, text, "
            "bm25(sentences) AS bm25_score "
            "FROM sentences WHERE sentences MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        return [
            {"page_id": row["page_id"], "sentence_id": row["sentence_id"], "text": row["text"], "bm25_score": float(row["bm25_score"])}
            for row in rows
        ]
    except sqlite3.OperationalError as exc:
        raise WikiIndexError(f"FTS5 search failed: {exc}") from exc
    finally:
        connection.close()

def index_status(index_path: Path | None = None) -> dict[str, Any]:
    path = index_path or DEFAULT_INDEX_PATH
    if not path.is_file():
        return {"ready": False, "index_path": str(path), "size_bytes": 0}
    connection = sqlite3.connect(str(path), timeout=10)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
    except sqlite3.Error:
        metadata = {}
    finally:
        connection.close()
    return {"ready": bool(metadata.get("complete") == "true"), "index_path": str(path), "size_bytes": path.stat().st_size, "metadata": metadata}
