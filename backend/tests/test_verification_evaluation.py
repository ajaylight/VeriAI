import json
import sqlite3
from pathlib import Path

from app.scripts.evaluate_verification import (
    classification_metrics,
    load_evaluation_records,
    load_excluded_claim_ids,
    run_evaluation,
    stratified_sample,
)
from app.services.wiki_index import create_index_schema, insert_sentence_batch


def test_stratified_sample_is_reproducible_and_balanced():
    records = [
        {"id": f"{label}-{index}", "label": label}
        for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
        for index in range(20)
    ]

    first = stratified_sample(records, limit=15, seed=42)
    second = stratified_sample(records, limit=15, seed=42)

    assert first == second
    assert len(first) == 15
    counts = {label: sum(item["label"] == label for item in first)
              for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")}
    assert counts == {"SUPPORTS": 5, "REFUTES": 5, "NOT ENOUGH INFO": 5}


def test_load_evaluation_records_reads_paper_dev_style_jsonl(tmp_path: Path):
    path = tmp_path / "paper_dev.jsonl"
    records = [
        {"id": 1, "claim": "A claim.", "label": "SUPPORTS", "evidence": []},
        {"id": 2, "claim": "Another claim.", "label": "REFUTES", "evidence": []},
        {"id": 3, "claim": "Unknown claim.", "label": "NOT ENOUGH INFO", "evidence": []},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    total, selected = load_evaluation_records(
        "paper_dev", limit=0, seed=42, data_dir=tmp_path
    )

    assert total == 3
    assert [item["id"] for item in selected] == [1, 2, 3]


def test_classification_metrics_calculates_three_class_scores():
    gold = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]
    predicted = ["SUPPORTS", "SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]

    result = classification_metrics(gold, predicted)

    assert result["claims"] == 4
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"]["REFUTES"]["SUPPORTS"] == 1
    assert result["per_class"]["NOT ENOUGH INFO"]["f1"] == 1.0


class FakeVerifier:
    model_name = "test-model"
    device = "test"

    def verify(self, claim, evidence):
        if claim.startswith("Alice"):
            verdict = "SUPPORTS"
        elif claim.startswith("Bob"):
            verdict = "REFUTES"
        else:
            verdict = "NOT ENOUGH INFO"
        return {
            "verdict": verdict,
            "confidence": 0.9,
            "decision_basis": "test",
            "evidence_selection_basis": "test",
            "decision_evidence": None,
            "selected_evidence": None,
            "evidence_scores": [],
        }

    def status(self):
        return {"model_name": self.model_name, "device": self.device, "loaded": True}


def test_end_to_end_evaluation_separates_classification_and_retrieval_metrics(tmp_path: Path):
    dataset = tmp_path / "paper_dev.jsonl"
    records = [
        {
            "id": 1,
            "claim": "Alice is a singer.",
            "label": "SUPPORTS",
            "evidence": [[[0, 0, "Alice", 0]]],
        },
        {
            "id": 2,
            "claim": "Bob is a singer.",
            "label": "REFUTES",
            "evidence": [[[0, 0, "Bob", 0]]],
        },
        {
            "id": 3,
            "claim": "Carol is a pilot.",
            "label": "NOT ENOUGH INFO",
            "evidence": [],
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    index_path = tmp_path / "evaluation.sqlite3"
    connection = sqlite3.connect(index_path)
    create_index_schema(connection)
    insert_sentence_batch(connection, [
        ("Alice", 0, "Alice is a singer."),
        ("Bob", 0, "Bob is a doctor."),
    ])
    connection.commit()
    connection.close()

    report = run_evaluation(
        split="paper_dev",
        limit=0,
        seed=42,
        top_k=3,
        index_path=index_path,
        data_dir=tmp_path,
        verifier=FakeVerifier(),
    )

    assert report["evaluated_claims"] == 3
    assert report["classification_metrics"]["accuracy"] == 1.0
    assert report["retrieval_evaluated_claims"] == 2
    assert report["retrieval_metrics_on_evidence_bearing_claims"]["claim_hit_rate_at_k"] == 1.0
    assert report["label_counts_in_sample"] == {
        "NOT ENOUGH INFO": 1,
        "REFUTES": 1,
        "SUPPORTS": 1,
    }



def test_load_excluded_claim_ids_reads_ids_and_validates_split(tmp_path: Path):
    report = tmp_path / "prior-report.json"
    report.write_text(
        json.dumps({
            "split": "paper_dev",
            "claim_details": [{"id": 10}, {"id": 20}, {"id": None}],
        }),
        encoding="utf-8",
    )

    assert load_excluded_claim_ids(report, "paper_dev") == {"10", "20"}

    import pytest
    with pytest.raises(ValueError, match="expected 'train'"):
        load_excluded_claim_ids(report, "train")


def test_load_evaluation_records_excludes_prior_claim_ids(tmp_path: Path):
    path = tmp_path / "paper_dev.jsonl"
    records = [
        {"id": 1, "claim": "Support one.", "label": "SUPPORTS", "evidence": []},
        {"id": 2, "claim": "Support two.", "label": "SUPPORTS", "evidence": []},
        {"id": 3, "claim": "Refute one.", "label": "REFUTES", "evidence": []},
        {"id": 4, "claim": "Refute two.", "label": "REFUTES", "evidence": []},
        {"id": 5, "claim": "Unknown one.", "label": "NOT ENOUGH INFO", "evidence": []},
        {"id": 6, "claim": "Unknown two.", "label": "NOT ENOUGH INFO", "evidence": []},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    total, selected = load_evaluation_records(
        "paper_dev",
        limit=0,
        seed=2026,
        data_dir=tmp_path,
        excluded_ids={"1", "4", "6"},
    )

    assert total == 6
    assert {item["id"] for item in selected} == {2, 3, 5}
