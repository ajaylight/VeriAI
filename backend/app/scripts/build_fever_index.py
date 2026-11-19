"""Build a streaming SQLite FTS5 BM25 index from FEVER wiki-pages.zip."""
from __future__ import annotations
import argparse
import io
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Iterator

from app.services.wiki_index import DEFAULT_INDEX_PATH, create_index_schema, insert_sentence_batch, parse_wiki_page

def iter_archive_sentences(archive_path: Path, max_pages: int | None = None) -> Iterator[tuple[str, int, str]]:
    """Read JSONL members from the archive in streaming mode."""
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".jsonl") and not name.endswith("/"))
        if not members:
            raise ValueError("No .jsonl files were found inside the supplied ZIP archive.")
        pages_seen = 0
        for member_name in members:
            with archive.open(member_name, "r") as raw_member:
                with io.TextIOWrapper(raw_member, encoding="utf-8") as text_member:
                    for line_number, line in enumerate(text_member, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON in {member_name}, line {line_number}: {exc.msg}") from exc
                        pages_seen += 1
                        yield from parse_wiki_page(record)
                        if max_pages is not None and pages_seen >= max_pages:
                            return

def build_index(archive_path: Path, output_path: Path = DEFAULT_INDEX_PATH, max_pages: int | None = None, overwrite: bool = False, batch_size: int = 10_000) -> dict[str, int | str]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"FEVER Wikipedia ZIP not found: {archive_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Index already exists: {output_path}. Use --overwrite to rebuild it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_path = output_path.with_name(output_path.name + ".building")
    if building_path.exists():
        if overwrite:
            building_path.unlink()
        else:
            raise FileExistsError(f"Incomplete build file exists: {building_path}. Use --overwrite to remove it.")

    started = time.time()
    pages_count = 0
    sentences_count = 0
    last_page_id: str | None = None
    batch: list[tuple[str, int, str]] = []
    connection = sqlite3.connect(str(building_path))
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-100000")
        create_index_schema(connection)
        for page_id, sentence_id, sentence_text in iter_archive_sentences(archive_path, max_pages=max_pages):
            if page_id != last_page_id:
                pages_count += 1
                last_page_id = page_id
                if pages_count % 100_000 == 0:
                    print(f"Indexed {pages_count:,} pages / {sentences_count + len(batch):,} sentences...", flush=True)
            batch.append((page_id, sentence_id, sentence_text))
            if len(batch) >= batch_size:
                sentences_count += insert_sentence_batch(connection, batch)
                connection.commit()
                batch.clear()
        if batch:
            sentences_count += insert_sentence_batch(connection, batch)
            connection.commit()
            batch.clear()
        connection.execute("INSERT INTO index_metadata(key, value) VALUES ('complete', 'true')")
        connection.execute("INSERT INTO index_metadata(key, value) VALUES ('pages', ?)", (str(pages_count),))
        connection.execute("INSERT INTO index_metadata(key, value) VALUES ('sentences', ?)", (str(sentences_count),))
        connection.execute("INSERT INTO index_metadata(key, value) VALUES ('source_archive', ?)", (archive_path.name,))
        connection.execute("INSERT INTO index_metadata(key, value) VALUES ('max_pages_limit', ?)", (str(max_pages or "all"),))
        connection.commit()
        connection.execute("INSERT INTO sentences(sentences) VALUES ('optimize')")
        connection.commit()
    except Exception:
        connection.close()
        raise
    else:
        connection.close()

    os.replace(building_path, output_path)
    return {
        "pages": pages_count,
        "sentences": sentences_count,
        "elapsed_seconds": int(time.time() - started),
        "index_path": str(output_path),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Build VeriAI's FEVER Wikipedia BM25 index.")
    parser.add_argument("--archive", required=True, type=Path, help="Path to official FEVER wiki-pages.zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH, help="SQLite index output path")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional small subset for a quick smoke test")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing index after a successful build")
    args = parser.parse_args()
    result = build_index(args.archive, args.output, args.max_pages, args.overwrite)
    print("FEVER BM25 index completed")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
