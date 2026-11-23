import sqlite3
from pathlib import Path
import pytest

from app.services.wiki_index import (
    build_fts_query,
    create_index_schema,
    extract_subject_title_candidates,
    filter_query_terms,
    insert_sentence_batch,
    normalize_page_title,
    open_index_connection,
    parse_wiki_page,
    search_index,
    search_index_with_connection,
    tokenize_query,
    _is_disambiguation_page,
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


def test_filter_query_terms_removes_grammatical_words_and_keeps_negation():
    assert filter_query_terms(["the", "oliver", "was", "not", "an", "actor"]) == [
        "oliver", "not", "actor"
    ]


def test_filter_query_terms_falls_back_when_everything_is_a_stopword():
    assert filter_query_terms(["the", "was", "an"]) == ["the", "was", "an"]


def test_title_candidate_extraction_keeps_subject_phrase():
    assert extract_subject_title_candidates("Palo Alto, California is a city.") == [
        "Palo Alto California", "Palo Alto", "Palo"
    ]

def test_title_candidate_extraction_normalizes_possessive_subject():
    assert extract_subject_title_candidates("Oliver Reed's favorite color was purple.") == [
        "Oliver Reed favorite color",
        "Oliver Reed favorite",
        "Oliver Reed",
        "Oliver",
    ]


def test_page_title_normalization_handles_fever_parentheses_and_punctuation():
    assert normalize_page_title("Palo_Alto,_California") == "palo alto california"
    assert normalize_page_title("Film_-LRB-2005_film-RRB-") == "film 2005 film"


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


def test_reusable_connection_supports_filtered_or(tmp_path: Path):
    db = tmp_path / "filtered-or.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Target", 0, "Oliver Reed was an actor."),
        ("Distractor", 0, "The actor was very popular."),
    ])
    con.commit()
    con.close()

    connection = open_index_connection(db)
    try:
        results = search_index_with_connection(
            connection, "Oliver Reed was an actor", top_k=5, operator="OR_FILTERED"
        )
    finally:
        connection.close()

    assert results
    assert results[0]["page_id"] == "Target"




def test_title_candidate_extraction_skips_leading_date_prefix():
    candidates = extract_subject_title_candidates(
        "On June 13th, 1986, Mary-Kate and Ashley Olsen were born."
    )
    assert candidates == [
        "Mary-Kate and Ashley Olsen",
        "Mary-Kate and Ashley",
        "Mary-Kate and",
        "Mary-Kate",
    ]

def test_page_title_is_searchable_and_boosted_over_unrelated_body_match(tmp_path: Path):
    db = tmp_path / "title-search.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Vietnam", 0, "Vietnam is a country in Southeast Asia."),
        ("Brevard_County_Manatees", 0, "Their mascot was Manny Manatee."),
        ("Forest_Heights_Collegiate_Institute", 0, "The mascot is a Trojan."),
        ("Kentucky_Black_Stone_Cherry_album", 0, "It was released through Mascot Records."),
    ])
    con.commit()
    con.close()

    results = search_index(
        "Vietnam is a mascot", top_k=4, index_path=db, operator="OR_FILTERED"
    )

    assert results
    assert results[0]["page_id"] == "Vietnam"
    assert results[0]["sentence_id"] == 0


def test_search_prioritizes_exact_subject_page_sentence(tmp_path: Path):
    db = tmp_path / "subject-page.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Vietnam", 0, "Vietnam is a country in Southeast Asia."),
        ("Vietnam", 1, "Vietnam has a long coastline."),
        ("Mascot_Records", 0, "Mascot Records is a record label."),
        ("Tusk_mascot", 1, "The current live mascot is Tusk IV."),
    ])
    con.commit()
    con.close()

    results = search_index(
        "Vietnam is a mascot", top_k=5, index_path=db, operator="OR_FILTERED"
    )

    assert results
    assert results[0]["page_id"] == "Vietnam"
    assert results[0]["sentence_id"] == 0


def test_search_resolves_possessive_full_name_to_exact_subject_page(tmp_path: Path):
    db = tmp_path / "possessive-subject-page.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Oliver_Reed", 0, "Robert Oliver Reed was an English actor."),
        ("Oliver", 0, "Oliver is a given name."),
        ("Oliver!", 0, "Oliver is a musical."),
    ])
    con.commit()
    con.close()

    results = search_index(
        "Oliver Reed's favorite color was purple.",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert results
    assert results[0]["page_id"] == "Oliver_Reed"
    assert results[0]["sentence_id"] == 0


def test_search_skips_leading_date_and_prioritizes_olsen_subject_page(tmp_path: Path):
    db = tmp_path / "date-prefix-subject.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("June", 0, "June is the sixth month of the year."),
        (
            "Mary-Kate_and_Ashley_Olsen",
            0,
            "Mary-Kate and Ashley Olsen were born on June 13, 1986.",
        ),
    ])
    con.commit()
    con.close()

    results = search_index(
        "On June 13th, 1986, Mary-Kate and Ashley Olsen were born.",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert results
    assert results[0]["page_id"] == "Mary-Kate_and_Ashley_Olsen"


def test_search_resolves_disambiguation_page_to_specific_subject_title(tmp_path: Path):
    db = tmp_path / "disambiguated-subject.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Jewell", 0, "Jewell may refer to :"),
        ("Jewell_-LRB-singer-RRB-", 0, "Jewell is an American singer-songwriter."),
        ("Jewell_-LRB-singer-RRB-", 1, "She sings country and pop music."),
        ("Buddy_Jewell", 0, "Buddy Jewell is an American country music singer."),
        ("Jewell_High_School", 0, "Jewell School is located in Oregon."),
    ])
    con.commit()
    con.close()

    results = search_index(
        "Jewell sings.",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert results
    assert results[0]["page_id"] == "Jewell_-LRB-singer-RRB-"
    assert results[0]["sentence_id"] == 0


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

def test_subject_page_returns_more_than_three_evidence_candidates(tmp_path: Path):
    db = tmp_path / "deeper-subject-page.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Sample_Town", 0, "Sample Town is a settlement in the county."),
        ("Sample_Town", 1, "Sample Town has a school and several homes."),
        ("Sample_Town", 2, "Sample Town has several parks and shops."),
        ("Sample_Town", 3, "Sample Town has many local community facilities."),
        ("Sample_Town", 4, "Sample Town is a place served by regional trains."),
    ])
    con.commit()
    con.close()

    results = search_index(
        "Sample Town is a place",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert len(results) == 5
    assert all(item["page_id"] == "Sample_Town" for item in results)



def test_disambiguation_page_check_uses_indexed_page_title_terms(tmp_path: Path):
    db = tmp_path / "disambiguation-check.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Jewell", 0, "Jewell may refer to :"),
        ("Jewell_-LRB-singer-RRB-", 0, "Jewell is an American R&B singer."),
    ])
    con.commit()
    con.close()

    connection = open_index_connection(db)
    try:
        assert _is_disambiguation_page(connection, "Jewell", ["jewell"])
        assert not _is_disambiguation_page(
            connection, "Jewell_-LRB-singer-RRB-", ["jewell"]
        )
    finally:
        connection.close()



def test_title_candidates_preserve_stopwords_and_prioritize_meaningful_phrases():
    cincinnati = extract_subject_title_candidates(
        "The Cincinnati Kid stars Steve McQueen."
    )
    assert cincinnati[0] == "The Cincinnati Kid"
    assert cincinnati.index("Cincinnati Kid") < cincinnati.index("The Cincinnati")

    greek = extract_subject_title_candidates(
        "The Greek language is spoken in the Mediterranean."
    )
    assert greek[0] == "The Greek language"
    assert greek.index("Greek language") < greek.index("The Greek")

    assert extract_subject_title_candidates(
        "Highway to Heaven is a drama series."
    )[0] == "Highway to Heaven"
    assert extract_subject_title_candidates(
        "The King and I is based on a novel."
    )[0] == "The King and I"
    assert extract_subject_title_candidates(
        "How to Train Your Dragon 2 used scalable multicore processing."
    )[0] == "How to Train Your Dragon 2"


def test_search_prioritizes_exact_title_with_stopwords(tmp_path: Path):
    db = tmp_path / "stopword-title-search.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        (
            "The_Cincinnati_Kid",
            0,
            "The Cincinnati Kid is a film starring Steve McQueen.",
        ),
        (
            "Cincinnati",
            0,
            "Cincinnati is a city in Ohio.",
        ),
    ])
    con.commit()
    con.close()

    results = search_index(
        "The Cincinnati Kid stars Steve McQueen.",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert results
    assert results[0]["page_id"] == "The_Cincinnati_Kid"


def test_search_prefers_greek_language_over_the_greek_page(tmp_path: Path):
    db = tmp_path / "greek-language-title-search.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("The_Greek", 0, "The Greek is a fictional character in a television drama."),
        ("Greek_language", 0, "Greek is an Indo-European language."),
        ("Greek_language", 11, "Greek is spoken in Greece and Cyprus."),
    ])
    con.commit()
    con.close()

    results = search_index(
        "The Greek language is spoken in the Mediterranean.",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert results
    assert results[0]["page_id"] == "Greek_language"


def test_unresolved_disambiguation_does_not_select_arbitrary_variant(tmp_path: Path):
    db = tmp_path / "unresolved-title-disambiguation.sqlite3"
    con = sqlite3.connect(db)
    create_index_schema(con)
    insert_sentence_batch(con, [
        ("Home_for_the_Holidays", 0, "Home for the Holidays may refer to :"),
        (
            "Home_for_the_Holidays_-LRB-1995_film-RRB-",
            0,
            "Home for the Holidays is a 1995 American comedy-drama film directed by Jodie Foster.",
        ),
        (
            "Home_for_the_Holidays_-LRB-Anthony_Hamilton_album-RRB-",
            0,
            "Home for the Holidays is an album by American recording artist Anthony Hamilton.",
        ),
    ])
    con.commit()
    con.close()

    results = search_index(
        "Home for the Holidays stars an American director.",
        top_k=5,
        index_path=db,
        operator="OR_FILTERED",
    )

    assert results
    assert results[0]["page_id"] == "Home_for_the_Holidays"
    assert all(
        "_-LRB-" not in item["page_id"]
        for item in results
    )
