"""Defensive reader for official FEVER JSONL files; preserves evidence references."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

SUPPORTED_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEVER_DIR = PROJECT_ROOT / "data" / "fever"
SPLIT_FILES = {"train": "train.jsonl", "paper_dev": "paper_dev.jsonl"}

class FeverDataError(ValueError):
    """Malformed JSONL or unexpected FEVER record."""

def get_split_path(split: str, data_dir: Path | None = None) -> Path:
    if split not in SPLIT_FILES:
        raise FeverDataError(f"Unknown split '{split}'. Allowed values: {', '.join(SPLIT_FILES)}")
    return (data_dir or DEFAULT_FEVER_DIR) / SPLIT_FILES[split]

def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise FeverDataError(f"Dataset path is not a file: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeverDataError(f"Invalid JSON on line {line_number} of {path.name}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise FeverDataError(f"Expected a JSON object on line {line_number} of {path.name}")
            if "id" not in record or "claim" not in record:
                raise FeverDataError(f"Missing required 'id' or 'claim' on line {line_number} of {path.name}")
            if not isinstance(record["claim"], str) or not record["claim"].strip():
                raise FeverDataError(f"Empty or invalid claim on line {line_number} of {path.name}")
            yield record

def summarize_split(split: str, data_dir: Path | None = None) -> dict[str, Any]:
    path = get_split_path(split, data_dir)
    label_counts: Counter[str] = Counter()
    total = evidence_count = missing_labels = 0
    unknown_labels: Counter[str] = Counter()
    for record in iter_jsonl(path):
        total += 1
        label = record.get("label")
        if isinstance(label, str):
            label_counts[label] += 1
            if label not in SUPPORTED_LABELS:
                unknown_labels[label] += 1
        else:
            missing_labels += 1
        if record.get("evidence"):
            evidence_count += 1
    return {
        "split": split, "file": path.name, "total_records": total,
        "label_counts": dict(sorted(label_counts.items())),
        "records_without_label": missing_labels,
        "records_with_evidence": evidence_count,
        "unknown_labels": dict(sorted(unknown_labels.items())),
    }

def get_samples(split: str, limit: int = 5, data_dir: Path | None = None) -> list[dict[str, Any]]:
    if not 1 <= limit <= 20:
        raise FeverDataError("limit must be between 1 and 20")
    samples = []
    for record in iter_jsonl(get_split_path(split, data_dir)):
        samples.append({
            "id": record["id"], "claim": record["claim"],
            "label": record.get("label"), "evidence": record.get("evidence", []),
        })
        if len(samples) >= limit:
            break
    return samples
