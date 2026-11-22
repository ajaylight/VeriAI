"""Streaming FEVER Wikipedia indexing and BM25 search using SQLite FTS5.

Both Wikipedia page titles and sentence text are searchable. Retrieval first
looks for an exact subject-page title, ranks sentences within that page, and
then supplements them with broad BM25 sentence matches.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "fever_wiki_fts5.sqlite3"
INDEX_SCHEMA_VERSION = "2"
TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
TITLE_TOKEN_RE = re.compile(r"[^\W_]+(?:[.'’&-][^\W_]+)*", flags=re.UNICODE)
DATE_PREFIX_RE = re.compile(
    r"^\\s*(?:on\\s+)?(?:january|february|march|april|may|june|july|august|
    r"september|october|november|december)\\s+\\d{1,2}(?:st|nd|rd|th)?
    r"(?:,\\s*|\\s+)\\d{4}\\s*[,;:]\\s*",
    flags=re.IGNORECASE,
)

SEARCH_STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "am", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "their", "his", "her", "do", "does", "did", "has", "have", "had",
}


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


def filter_query_terms(terms: list[str]) -> list[str]:
    """Remove common grammatical terms while preserving negation and conjunction."""
    filtered = [term for term in terms if term not in SEARCH_STOPWORDS]
    return filtered or terms


def normalize_page_title(value: str) -> str:
    """Normalize a FEVER page ID or phrase for exact title comparison."""
    normalized = value
    for marker, replacement in (
        ("-LRB-", " "), ("-RRB-", " "), ("-LSB-", " "), ("-RSB-", " "),
        ("-LCB-", " "), ("-RCB-", " "),
    ):
        normalized = re.sub(marker, replacement, normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("_", " ")
    return " ".join(re.findall(r"[\w]+", normalized.casefold(), flags=re.UNICODE))


SUBJECT_BOUNDARY_WORDS = {
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
    "died", "wins", "won", "became", "becomes", "born", "lived", "serves",
    "served", "founded", "helped", "plays", "played", "wrote", "released",
    "started", "includes", "included", "contains", "consists", "affects",
    "affect", "owns", "owned",
}


def extract_subject_title_candidates(query: str, max_terms: int = 5) -> list[str]:
    """Generate longest-first title candidates from the leading subject phrase."""
    # Claims can begin with a date adjunct ("On June 13th, 1986, ...").
    # Remove that prefix before guessing the leading subject page title.
    subject_query = DATE_PREFIX_RE.sub("", query, count=1)
    words = TITLE_TOKEN_RE.findall(subject_query)
    content_words: list[str] = []
    for word in words:
        # Treat possessive forms such as "Reed's" as the underlying title token
        # ("Reed") so exact Wikipedia page matching still works.
        title_word = re.sub(r"['’]s$", "", word, flags=re.IGNORECASE)
        normalized = normalize_page_title(title_word)
        if normalized in SUBJECT_BOUNDARY_WORDS and content_words:
            break
        if normalized and normalized not in SEARCH_STOPWORDS:
            content_words.append(title_word)
        if len(content_words) >= max_terms:
            break
    return [
        " ".join(content_words[:length])
        for length in range(len(content_words), 0, -1)
    ]


def build_fts_query(terms: list[str], operator: str = "OR") -> str:
    """Build a safe FTS query using OR (broad recall) or AND (strict matching)."""
    normalized_operator = operator.upper()
    if normalized_operator not in {"OR", "AND"}:
        raise ValueError("operator must be OR or AND")
    if not terms:
        raise ValueError("At least one search term is required.")
    quoted_terms = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return f" {normalized_operator} ".join(quoted_terms)


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
    """Create an FTS5 table with searchable page IDs and sentence text."""
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE sentences USING fts5("
            "page_id, sentence_id UNINDEXED, text, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as exc:
        raise WikiIndexError(
            "This Python SQLite build does not support FTS5. Install a current "
            "official Python 3.11 build or use a Python distribution with FTS5 enabled."
        ) from exc
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO index_metadata(key, value) VALUES ('schema_version', ?)",
        (INDEX_SCHEMA_VERSION,),
    )
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


def open_index_connection(
    index_path: Path | None = None,
    timeout: int = 60,
) -> sqlite3.Connection:
    """Open a reusable read connection and reject obsolete index schemas."""
    path = index_path or DEFAULT_INDEX_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Wikipedia search index not found: {path}. Build it with the FEVER index command first."
        )
    connection = sqlite3.connect(str(path), timeout=timeout)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error:
        connection.close()
        raise WikiIndexError(
            "The Wikipedia index has no schema version. Rebuild it with "
            "python -m app.scripts.build_fever_index --archive ..\\data\\fever\\wiki-pages.zip --overwrite."
        )
    if version is None or version["value"] != INDEX_SCHEMA_VERSION:
        found = version["value"] if version is not None else "missing"
        connection.close()
        raise WikiIndexError(
            f"Wikipedia index schema {found} is incompatible with required schema "
            f"{INDEX_SCHEMA_VERSION}. Rebuild the index with --overwrite."
        )
    connection.execute("PRAGMA cache_size=-100000")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def _find_exact_subject_pages(
    connection: sqlite3.Connection,
    query: str,
) -> list[tuple[str, list[str]]]:
    """Find page IDs whose normalized title exactly matches a leading claim phrase."""
    for candidate in extract_subject_title_candidates(query):
        candidate_normalized = normalize_page_title(candidate)
        terms = tokenize_query(candidate)
        if not terms:
            continue
        title_query = " AND ".join(
            f'page_id : "{term.replace(chr(34), chr(34) * 2)}"'
            for term in terms
        )
        rows = connection.execute(
            "SELECT DISTINCT page_id FROM sentences WHERE sentences MATCH ? LIMIT 20000",
            (title_query,),
        ).fetchall()
        exact_matches = [
            row["page_id"]
            for row in rows
            if normalize_page_title(row["page_id"]) == candidate_normalized
        ]
        if exact_matches:
            return [(page_id, terms) for page_id in exact_matches]
    return []


def _search_subject_page(
    connection: sqlite3.Connection,
    page_id: str,
    title_terms: list[str],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank matching sentences within an exact subject page before global BM25 results."""
    title_filter = " AND ".join(
        f'page_id : "{term.replace(chr(34), chr(34) * 2)}"'
        for term in title_terms
    )
    body_terms = filter_query_terms(tokenize_query(query))
    body_filter = " OR ".join(
        f'text : "{term.replace(chr(34), chr(34) * 2)}"'
        for term in body_terms
    )

    if body_filter:
        match_query = f"({title_filter}) AND ({body_filter})"
    else:
        match_query = title_filter

    sql = (
        "SELECT page_id, CAST(sentence_id AS INTEGER) AS sentence_id, text, "
        "bm25(sentences, 5.0, 0.0, 1.0) AS bm25_score "
        "FROM sentences WHERE sentences MATCH ? AND page_id = ? "
        "ORDER BY bm25(sentences, 5.0, 0.0, 1.0) LIMIT ?"
    )
    try:
        rows = connection.execute(sql, (match_query, page_id, top_k)).fetchall()
    except sqlite3.OperationalError:
        rows = []

    # Always include the page's opening sentence as high-value context, then
    # fill remaining slots with the best claim-matching sentences from that page.
    title_only_query = f"({title_filter})"
    opening_rows = connection.execute(
        "SELECT page_id, CAST(sentence_id AS INTEGER) AS sentence_id, text, "
        "bm25(sentences, 5.0, 0.0, 1.0) AS bm25_score "
        "FROM sentences WHERE sentences MATCH ? AND page_id = ? "
        "ORDER BY CAST(sentence_id AS INTEGER) LIMIT 1",
        (title_only_query, page_id),
    ).fetchall()

    ordered_rows = list(opening_rows)
    seen_sentence_ids = {row["sentence_id"] for row in opening_rows}
    for row in rows:
        if row["sentence_id"] not in seen_sentence_ids:
            ordered_rows.append(row)
            seen_sentence_ids.add(row["sentence_id"])
        if len(ordered_rows) >= top_k:
            break

    if not ordered_rows:
        ordered_rows = connection.execute(
            "SELECT page_id, CAST(sentence_id AS INTEGER) AS sentence_id, text, "
            "bm25(sentences, 5.0, 0.0, 1.0) AS bm25_score "
            "FROM sentences WHERE sentences MATCH ? AND page_id = ? "
            "ORDER BY CAST(sentence_id AS INTEGER) LIMIT ?",
            (title_only_query, page_id, top_k),
        ).fetchall()

    return [
        {
            "page_id": row["page_id"],
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "bm25_score": float(row["bm25_score"]),
        }
        for row in ordered_rows[:top_k]
    ]


def search_index_with_connection(
    connection: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    operator: str = "OR",
) -> list[dict[str, Any]]:
    """Search exact subject pages first, then supplement with global BM25 matches."""
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    terms = tokenize_query(query)
    if not terms:
        raise ValueError("The claim must contain at least one searchable word (2+ characters).")

    normalized_operator = operator.upper()
    if normalized_operator == "OR_FILTERED":
        terms = filter_query_terms(terms)
        normalized_operator = "OR"
    fts_query = build_fts_query(terms, operator=normalized_operator)

    try:
        global_rows = connection.execute(
            "SELECT page_id, CAST(sentence_id AS INTEGER) AS sentence_id, text, "
            "bm25(sentences, 5.0, 0.0, 1.0) AS bm25_score "
            "FROM sentences WHERE sentences MATCH ? "
            "ORDER BY bm25(sentences, 5.0, 0.0, 1.0) LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        global_results = [
            {
                "page_id": row["page_id"],
                "sentence_id": row["sentence_id"],
                "text": row["text"],
                "bm25_score": float(row["bm25_score"]),
            }
            for row in global_rows
        ]
    except sqlite3.OperationalError as exc:
        raise WikiIndexError(f"FTS5 search failed: {exc}") from exc

    subject_pages = _find_exact_subject_pages(connection, query)
    subject_results: list[dict[str, Any]] = []
    # Search deeper within the subject page so specific annotated evidence
    # is not excluded by an overly small per-page candidate cap.
    per_page_limit = max(1, min(10, top_k))
    for page_id, title_terms in subject_pages:
        subject_results.extend(
            _search_subject_page(connection, page_id, title_terms, query, per_page_limit)
        )

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in subject_results + global_results:
        key = (item["page_id"], item["sentence_id"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= top_k:
            break

    return merged


def search_index(
    query: str,
    top_k: int = 5,
    index_path: Path | None = None,
    operator: str = "OR",
) -> list[dict[str, Any]]:
    """Open the index, perform one search, and close the connection."""
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    connection = open_index_connection(index_path)
    try:
        return search_index_with_connection(
            connection,
            query,
            top_k=top_k,
            operator=operator,
        )
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
    schema_compatible = metadata.get("schema_version") == INDEX_SCHEMA_VERSION
    return {
        "ready": bool(metadata.get("complete") == "true" and schema_compatible),
        "schema_compatible": schema_compatible,
        "required_schema_version": INDEX_SCHEMA_VERSION,
        "index_path": str(path),
        "size_bytes": path.stat().st_size,
        "metadata": metadata,
    }
