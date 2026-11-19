import sqlite3
from pathlib import Path
import pytest
from app.services.wiki_index import build_fts_query, create_index_schema, insert_sentence_batch, parse_wiki_page, search_index, tokenize_query

def test_parse_wiki_page_extracts_numbered_sentences():
    record = {"id": "Test_Page", "lines": "0\tFirst sentence.\n1\tSecond sentence.\n2\t\nnot-a-line"}
    assert list(parse_wiki_page(record)) == [("Test_Page", 0, "First sentence."), ("Test_Page", 1, "Second sentence.")]

def test_tokenize_query_deduplicates_and_builds_safe_fts_query():
    terms = tokenize_query("The river, river banks and water!")
    assert terms == ["the", "river", "banks", "and", "water"]
    assert build_fts_query(terms) == '"the" OR "river" OR "banks" OR "and" OR "water"'

def test_sqlite_fts_search_returns_relevant_sentence(tmp_path: Path):
    db = tmp_path / "test.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("River", 0, "The Amazon river flows through South America."),
        ("Football", 0, "The football club won a national championship."),
        ("Water", 2, "Water freezes at zero degrees Celsius."),
    ])
    con.commit()
    con.close()
    result = search_index("Amazon river South America", top_k=2, index_path=db)
    assert result
    assert result[0]["page_id"] == "River"
    assert result[0]["sentence_id"] == 0

def test_search_rejects_invalid_top_k(tmp_path: Path):
    with pytest.raises(ValueError, match="top_k"):
        search_index("some query", 0, tmp_path / "absent.sqlite3")

def test_build_index_streams_jsonl_from_zip(tmp_path: Path):
    import json
    import zipfile
    from app.scripts.build_fever_index import build_index
    archive = tmp_path / "tiny-wiki.zip"
    page = {
        "id": "Example_Page",
        "text": "Mercury is a planet.",
        "lines": "0\tMercury is the smallest planet in the Solar System.\n1\tIt has no moons.",
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wiki-pages/wiki-001.jsonl", json.dumps(page) + "\n")
    output = tmp_path / "tiny-index.sqlite3"
    stats = build_index(archive, output)
    assert stats["pages"] == 1
    assert stats["sentences"] == 2
    results = search_index("smallest planet Solar System", 3, output)
    assert results[0]["page_id"] == "Example_Page"
    assert results[0]["sentence_id"] == 0


def test_build_index_recovers_isolated_invalid_utf8_byte(tmp_path: Path):
    archive = tmp_path / "legacy-byte.zip"
    raw_line = (
        b'{"id":"Degree_Page","text":"Caf\\xc3\\xa9 and 30 \\xb0C",'
        b'"lines":"0\\tCaf\\xc3\\xa9 reached 30 \\xb0C."}\\n'
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wiki-pages/page.jsonl", raw_line)

    output = tmp_path / "legacy-index.sqlite3"
    stats = build_index(archive, output)
    assert stats["pages"] == 1
    assert stats["sentences"] == 1
    assert stats["utf8_fallback_lines"] == 1
    results = search_index("Café reached 30", 3, output)
    assert results[0]["text"] == "Café reached 30 °C."
