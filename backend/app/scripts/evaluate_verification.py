"""Evaluate retrieval plus NLI verdicts on a reproducible FEVER sample.

Run from backend/:
    python -m app.scripts.evaluate_verification --split paper_dev --limit 30 --top-k 5 --output ..\\reports\\verification_paper_dev_30.json

Use --limit 0 to evaluate the complete selected split. The end-to-end
classification metrics include all three FEVER labels. Retrieval metrics are
reported separately and only for claims with sentence-level gold evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from app.scripts.evaluate_retrieval import (
    MATCH_MODES,
    aggregate_metrics as aggregate_retrieval_metrics,
    extract_gold_evidence_sets,
)
from app.services.fever_loader import DEFAULT_FEVER_DIR, SPLIT_FILES, iter_jsonl
from app.services.nli_verifier import NliVerifier
from app.services.wiki_index import DEFAULT_INDEX_PATH, open_index_connection, search_index_with_connection

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def stratified_sample(
    records: list[dict[str, Any]],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a seeded, approximately label-proportional sample."""
    if limit < 0:
        raise ValueError("limit must be zero (all records) or a positive integer.")
    if limit == 0 or limit >= len(records):
        return list(records)
    if not records:
        return []

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get("label", "UNKNOWN"))].append(record)

    total = len(records)
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for label, bucket in buckets.items():
        exact_quota = limit * len(bucket) / total
        quotas[label] = min(len(bucket), int(exact_quota))
        fractions.append((exact_quota - int(exact_quota), label))

    remaining = limit - sum(quotas.values())
    for _, label in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if quotas[label] < len(buckets[label]):
            quotas[label] += 1
            remaining -= 1

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in sorted(buckets):
        selected.extend(rng.sample(buckets[label], quotas[label]))
    rng.shuffle(selected)
    return selected


def load_evaluation_records(
    split: str,
    limit: int,
    seed: int,
    data_dir: Path = DEFAULT_FEVER_DIR,
) -> tuple[int, list[dict[str, Any]]]:
    """Load labelled FEVER records and produce a deterministic stratified sample."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'. Choose from: {', '.join(SPLIT_FILES)}.")
    path = data_dir / SPLIT_FILES[split]
    records = [
        record for record in iter_jsonl(path)
        if record.get("label") in LABELS
    ]
    if not records:
        raise ValueError(f"No labelled FEVER records found in {path}.")
    return len(records), stratified_sample(records, limit, seed)


def classification_metrics(
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, Any]:
    """Calculate accuracy, macro metrics, per-class scores and a fixed-order confusion matrix."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted label lists must have equal length.")
    if not gold_labels:
        raise ValueError("At least one labelled prediction is required.")

    confusion = {
        gold: {predicted: 0 for predicted in LABELS}
        for gold in LABELS
    }
    for gold, predicted in zip(gold_labels, predicted_labels):
        if gold not in LABELS or predicted not in LABELS:
            raise ValueError("All labels must be one of the three FEVER classes.")
        confusion[gold][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in LABELS if other != label)
        false_negative = sum(confusion[label][other] for other in LABELS if other != label)
        support = sum(confusion[label].values())
        predicted_count = true_positive + false_positive
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_precision = sum(per_class[label]["precision"] for label in LABELS) / len(LABELS)
    macro_recall = sum(per_class[label]["recall"] for label in LABELS) / len(LABELS)
    macro_f1 = sum(per_class[label]["f1"] for label in LABELS) / len(LABELS)
    correct = sum(confusion[label][label] for label in LABELS)
    return {
        "claims": len(gold_labels),
        "accuracy": correct / len(gold_labels),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def run_evaluation(
    split: str,
    limit: int,
    seed: int,
    top_k: int,
    index_path: Path = DEFAULT_INDEX_PATH,
    data_dir: Path = DEFAULT_FEVER_DIR,
    verifier: Any | None = None,
) -> dict[str, Any]:
    """Run end-to-end retrieval and NLI classification on a FEVER sample."""
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    total_eligible, samples = load_evaluation_records(split, limit, seed, data_dir)
    if not samples:
        raise ValueError("No FEVER records selected for evaluation.")

    label_counts = Counter(str(record["label"]) for record in samples)
    predictions: list[str] = []
    gold_labels: list[str] = []
    retrieval_evaluated: list[tuple[list[set[tuple[str, int]]], list[tuple[str, int]]]] = []
    claim_details: list[dict[str, Any]] = []
    classifier = verifier or NliVerifier()
    started = time.time()

    connection = open_index_connection(index_path)
    try:
        for index, record in enumerate(samples, start=1):
            try:
                retrieved = search_index_with_connection(
                    connection,
                    record["claim"],
                    top_k=top_k,
                    operator=MATCH_MODES["or-filtered"],
                )
            except ValueError as exc:
                if "searchable word" not in str(exc):
                    raise
                retrieved = []

            nli_result = classifier.verify(record["claim"], retrieved)
            gold_label = str(record["label"])
            predicted_label = str(nli_result["verdict"])
            gold_labels.append(gold_label)
            predictions.append(predicted_label)

            gold_sets = extract_gold_evidence_sets(record.get("evidence"))
            if gold_sets:
                retrieved_refs = [
                    (item["page_id"], item["sentence_id"])
                    for item in retrieved
                ]
                retrieval_evaluated.append((gold_sets, retrieved_refs))

            claim_details.append({
                "id": record.get("id"),
                "claim": record["claim"],
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "correct": gold_label == predicted_label,
                "confidence": nli_result.get("confidence"),
                "decision_basis": nli_result.get("decision_basis"),
                "evidence_selection_basis": nli_result.get("evidence_selection_basis"),
                "decision_evidence": nli_result.get("decision_evidence"),
                "selected_evidence": nli_result.get("selected_evidence"),
                "retrieved_evidence": retrieved,
                "evidence_scores": nli_result.get("evidence_scores", []),
                "gold_evidence_sets": [
                    [list(reference) for reference in sorted(group)]
                    for group in gold_sets
                ],
            })

            if index % 10 == 0 or index == len(samples):
                print(f"Verified {index:,}/{len(samples):,} claims...", flush=True)
    finally:
        connection.close()

    classifier_status = classifier.status() if hasattr(classifier, "status") else {}
    retrieval_metrics = (
        aggregate_retrieval_metrics(retrieval_evaluated, top_k)
        if retrieval_evaluated else None
    )
    return {
        "evaluation_type": "end_to_end_retrieval_plus_nli",
        "split": split,
        "dataset_file": SPLIT_FILES[split],
        "eligible_labelled_claims_in_split": total_eligible,
        "evaluated_claims": len(samples),
        "sample_limit": limit,
        "random_seed": seed,
        "sampling": "seeded_stratified_by_label",
        "top_k": top_k,
        "retrieval_mode": "OR_FILTERED",
        "label_counts_in_sample": dict(sorted(label_counts.items())),
        "model_name": classifier_status.get("model_name", getattr(classifier, "model_name", None)),
        "device": classifier_status.get("device", getattr(classifier, "device", None)),
        "model_loaded": classifier_status.get("loaded"),
        "elapsed_seconds": round(time.time() - started, 2),
        "seconds_per_claim": round((time.time() - started) / len(samples), 3),
        "classification_metrics": classification_metrics(gold_labels, predictions),
        "retrieval_metrics_on_evidence_bearing_claims": retrieval_metrics,
        "retrieval_evaluated_claims": len(retrieval_evaluated),
        "claim_details": claim_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end FEVER retrieval and NLI classification."
    )
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="paper_dev")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Stratified sample size; use 0 to evaluate every labelled record.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_FEVER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    report = run_evaluation(
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        top_k=args.top_k,
        index_path=args.index,
        data_dir=args.data_dir,
    )

    metrics = report["classification_metrics"]
    print("\nFEVER end-to-end verification evaluation")
    print(f"Split: {report['split']}")
    print(f"Eligible labelled claims: {report['eligible_labelled_claims_in_split']:,}")
    print(f"Evaluated claims: {report['evaluated_claims']:,}")
    print(f"Sample labels: {report['label_counts_in_sample']}")
    print(f"Model: {report['model_name']} ({report['device']})")
    print(f"Elapsed: {report['elapsed_seconds']:.2f} seconds")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    for label in LABELS:
        values = metrics["per_class"][label]
        print(
            f"{label}: precision={values['precision'] * 100:.2f}% "
            f"recall={values['recall'] * 100:.2f}% "
            f"F1={values['f1'] * 100:.2f}% "
            f"(n={values['support']})"
        )

    retrieval = report["retrieval_metrics_on_evidence_bearing_claims"]
    if retrieval is not None:
        print(
            f"\nRetrieval metrics on {report['retrieval_evaluated_claims']} "
            f"evidence-bearing claims:"
        )
        print(f"Sentence precision@{args.top_k}: {retrieval['precision_at_k'] * 100:.2f}%")
        print(f"Sentence recall@{args.top_k}: {retrieval['sentence_recall_at_k'] * 100:.2f}%")
        print(f"Claim hit rate@{args.top_k}: {retrieval['claim_hit_rate_at_k'] * 100:.2f}%")
        print(
            "Complete evidence-set recall@"
            f"{args.top_k}: {retrieval['complete_evidence_set_recall_at_k'] * 100:.2f}%"
        )

    if args.show_errors:
        errors = [item for item in report["claim_details"] if not item["correct"]]
        print(f"\nMisclassified claims: {len(errors)}")
        for item in errors:
            print("-" * 78)
            print(
                f"ID {item['id']} | gold={item['gold_label']} "
                f"| predicted={item['predicted_label']}"
            )
            print(f"Claim: {item['claim']}")
            if item["selected_evidence"]:
                selected = item["selected_evidence"]
                print(
                    f"Selected: {selected['page_id']}#{selected['sentence_id']} "
                    f"- {selected['text'][:220]}"
                )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {args.output}")


if __name__ == "__main__":
    main()
