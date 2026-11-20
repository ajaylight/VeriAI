import sqlite3
from pathlib import Path

import pytest

from app.services.wiki_index import (
    build_fts_query,
    create_index_schema,
    insert_sentence_batch,
    open_index_connection,
    parse_wiki_page,
    search_index,
    search_index_with_connection,
    tokenize_query,
)


def test_parse_wiki_page_extracts_numbered_sentences():
    record = {"id": "Test_Page", "lines": "0\tFirst sentence.\n1\tSecond sentence.\n2\t\nnot-a-line"}
    assert list(parse_wiki_page(record)) == [
        ("Test_Page", 0, "First sentence."),
        ("Test_Page", 1, "Second sentence."),
    ]


def test_tokenize_query_deduplicates_and_builds_safe_fts_query():
    terms = tokenize_query("The river, river banks and water!")
    assert terms == ["the", "river", "banks", "and", "water"]
    assert build_fts_query(terms) == '"the" OR "river" OR "banks" OR "and" OR "water"'


def test_build_fts_query_supports_and_mode():
    assert build_fts_query(["oliver", "reed"], operator="AND") == '"oliver" AND "reed"'


def test_build_fts_query_rejects_unknown_operator():
    with pytest.raises(ValueError, match="operator"):
        build_fts_query(["oliver", "reed"], operator="NEAR")


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


def test_reusable_connection_supports_strict_and_search(tmp_path: Path):
    db = tmp_path / "and-search.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Both_Terms", 0, "Oliver Reed was an actor."),
        ("Oliver_Only", 0, "Oliver was mentioned in this sentence."),
        ("Reed_Only", 0, "Reed was mentioned in this sentence."),
    ])
    con.commit()
    con.close()

    connection = open_index_connection(db)
    try:
        results = search_index_with_connection(
            connection, "Oliver Reed", top_k=5, operator="AND"
        )
    finally:
        connection.close()

    assert results
    assert all("Oliver" in item["text"] and "Reed" in item["text"] for item in results)
    assert results[0]["page_id"] == "Both_Terms"


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
    import json
    import zipfile

    from app.scripts.build_fever_index import build_index

    archive = tmp_path / "legacy-byte.zip"
    page = {
        "id": "Degree_Page",
        "text": "Café and 30 °C",
        "lines": "0" + chr(9) + "Café reached 30 °C.",
    }
    raw_line = json.dumps(page, ensure_ascii=False).encode("utf-8")
    raw_line = raw_line.replace("°".encode("utf-8"), bytes([0xB0])) + bytes([10])

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wiki-pages/page.jsonl", raw_line)

    output = tmp_path / "legacy-index.sqlite3"
    stats = build_index(archive, output)
    assert stats["pages"] == 1
    assert stats["sentences"] == 1
    assert stats["utf8_fallback_lines"] == 1
    results = search_index("Café reached 30", 3, output)
    assert results[0]["text"] == "Café reached 30 °C."


def test_build_index_ignores_macos_appledouble_sidecars(tmp_path: Path):
    import json
    import zipfile

    from app.scripts.build_fever_index import build_index

    archive = tmp_path / "macos-style.zip"
    page = {
        "id": "Real_Page",
        "text": "A valid page",
        "lines": "0" + chr(9) + "This is a real Wikipedia sentence.",
    }
    real_jsonl = (json.dumps(page) + "\n").encode("utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__MACOSX/wiki-pages/._wiki-001.jsonl", b"AppleDouble metadata, not JSONL")
        zf.writestr("wiki-pages/._wiki-001.jsonl", b"AppleDouble metadata, not JSONL")
        zf.writestr("wiki-pages/wiki-001.jsonl", real_jsonl)

    output = tmp_path / "macos-style-index.sqlite3"
    stats = build_index(archive, output)
    assert stats["pages"] == 1
    assert stats["sentences"] == 1
    results = search_index("real Wikipedia sentence", 3, output)
    assert results[0]["page_id"] == "Real_Page"
