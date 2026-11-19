import json
from pathlib import Path
import pytest
from app.services.fever_loader import FeverDataError, get_samples, iter_jsonl, summarize_split

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

def test_summarize_split_counts_sentence_evidence_not_nei_placeholder(tmp_path: Path) -> None:
    records = [
        {"id": 1, "claim": "A claim.", "label": "SUPPORTS", "evidence": [[[1, 2, "Page", 0]]]},
        {"id": 2, "claim": "Another claim.", "label": "REFUTES", "evidence": [[[1, 3, "Page", 1]]]},
        {"id": 3, "claim": "Unknown claim.", "label": "NOT ENOUGH INFO", "evidence": [[[1, 4, None, None]]]},
    ]
    write_jsonl(tmp_path / "train.jsonl", records)
    result = summarize_split("train", tmp_path)
    assert result["total_records"] == 3
    assert result["records_with_evidence_annotations"] == 3
    assert result["records_with_sentence_evidence"] == 2
    assert result["records_without_sentence_evidence"] == 1

def test_get_samples_preserves_claim_id_and_evidence(tmp_path: Path) -> None:
    records = [{"id": 99, "claim": "A sample claim.", "label": "SUPPORTS", "evidence": [[[1, 2, "Page", 0]]]}]
    write_jsonl(tmp_path / "paper_dev.jsonl", records)
    samples = get_samples("paper_dev", limit=1, data_dir=tmp_path)
    assert samples[0]["id"] == 99
    assert samples[0]["claim"] == "A sample claim."
    assert samples[0]["evidence"] == [[[1, 2, "Page", 0]]]

def test_malformed_json_has_line_number(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.jsonl"
    file_path.write_text('{"id": 1, "claim": "ok"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(FeverDataError, match="line 2"):
        list(iter_jsonl(file_path))
