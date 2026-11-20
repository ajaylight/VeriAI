"""Evaluate FEVER evidence retrieval against annotated Wikipedia sentence IDs.

Run from the backend directory:
    python -m app.scripts.evaluate_retrieval --split paper_dev --limit 50 --top-k 5 10 --match-mode or

Use --match-mode and to compare strict all-term matching against broad OR matching.
The evaluator reuses one SQLite connection for the full batch.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.fever_loader import DEFAULT_FEVER_DIR, SPLIT_FILES, iter_jsonl
from app.services.wiki_index import (
    DEFAULT_INDEX_PATH,
    open_index_connection,
    search_index_with_connection,
)

EvidenceRef = tuple[str, int]
EvidenceSet = set[EvidenceRef]


def extract_gold_evidence_sets(evidence: Any) -> list[EvidenceSet]:
    """Extract valid (Wikipedia page ID, sentence ID) sets from FEVER evidence."""
    if not isinstance(evidence, list):
        return []

    evidence_sets: list[EvidenceSet] = []
    for group in evidence:
        if not isinstance(group, list):
            continue

        references: EvidenceSet = set()
        for item in group:
            if not isinstance(item, list) or len(item) < 4:
                continue
            page_id, sentence_id = item[2], item[3]
            if (
                isinstance(page_id, str)
                and page_id
                and isinstance(sentence_id, int)
                and not isinstance(sentence_id, bool)
                and sentence_id >= 0
            ):
                references.add((page_id, sentence_id))

        if references:
            evidence_sets.append(references)

    return evidence_sets


def aggregate_metrics(
    evaluated: list[tuple[list[EvidenceSet], list[EvidenceRef]]],
    k: int,
) -> dict[str, int | float]:
    """Calculate sentence-level precision/recall and claim-level retrieval rates."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not evaluated:
        raise ValueError("No evidence-bearing claims were evaluated.")

    gold_sentence_count = 0
    relevant_sentence_count = 0
    claim_hit_count = 0
    complete_evidence_set_count = 0
    returned_count = 0
    short_result_count = 0

    for gold_sets, retrieved in evaluated:
        gold_union: EvidenceSet = set().union(*gold_sets)
        retrieved_top_k = set(retrieved[:k])
        returned_count += len(retrieved[:k])
        if len(retrieved) < k:
            short_result_count += 1

        gold_sentence_count += len(gold_union)
        relevant_sentence_count += len(gold_union.intersection(retrieved_top_k))

        if gold_union.intersection(retrieved_top_k):
            claim_hit_count += 1
        if any(gold_set.issubset(retrieved_top_k) for gold_set in gold_sets):
            complete_evidence_set_count += 1

    claim_count = len(evaluated)
    precision_denominator = k * claim_count
    return {
        "claims": claim_count,
        "gold_sentence_references": gold_sentence_count,
        "retrieved_gold_sentence_references": relevant_sentence_count,
        "precision_at_k": relevant_sentence_count / precision_denominator,
        "sentence_recall_at_k": (
            relevant_sentence_count / gold_sentence_count if gold_sentence_count else 0.0
        ),
        "claim_hit_rate_at_k": claim_hit_count / claim_count,
        "complete_evidence_set_recall_at_k": complete_evidence_set_count / claim_count,
        "claims_with_any_gold_sentence_retrieved": claim_hit_count,
        "claims_with_a_complete_evidence_set_retrieved": complete_evidence_set_count,
        "total_results_returned_at_k": returned_count,
        "claims_returning_fewer_than_k_results": short_result_count,
    }


def load_evaluation_records(
    split: str,
    limit: int,
    seed: int,
) -> tuple[int, list[tuple[dict[str, Any], list[EvidenceSet]]]]:
    """Load evidence-bearing examples and select a reproducible random sample."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'. Choose from: {', '.join(SPLIT_FILES)}")
    if limit < 0:
        raise ValueError("--limit must be zero (all eligible records) or a positive integer.")

    dataset_path = DEFAULT_FEVER_DIR / SPLIT_FILES[split]
    eligible: list[tuple[dict[str, Any], list[EvidenceSet]]] = []

    for record in iter_jsonl(dataset_path):
        gold_sets = extract_gold_evidence_sets(record.get("evidence"))
        if gold_sets:
            eligible.append((record, gold_sets))

    total_eligible = len(eligible)
    if limit and total_eligible > limit:
        eligible = random.Random(seed).sample(eligible, limit)

    return total_eligible, eligible


def run_evaluation(
    split: str,
    limit: int,
    seed: int,
    top_ks: list[int],
    index_path: Path,
    match_mode: str = "or",
) -> dict[str, Any]:
    if not top_ks or any(k < 1 or k > 20 for k in top_ks):
        raise ValueError("Each --top-k value must be between 1 and 20.")
    if match_mode not in {"or", "and"}:
        raise ValueError("match_mode must be 'or' or 'and'.")
    top_ks = sorted(set(top_ks))

    total_eligible, samples = load_evaluation_records(split, limit, seed)
    if not samples:
        raise ValueError(
            f"No FEVER records with sentence-level evidence found in "
            f"{DEFAULT_FEVER_DIR / SPLIT_FILES[split]}."
        )

    label_counts = Counter(
        str(record.get("label", "UNKNOWN")) for record, _ in samples
    )
    max_k = max(top_ks)
    evaluated: list[tuple[list[EvidenceSet], list[EvidenceRef]]] = []
    started = time.time()

    # Reuse one open connection so SQLite's page cache remains warm across claims.
    connection = open_index_connection(index_path)
    try:
        for index, (record, gold_sets) in enumerate(samples, start=1):
            try:
                results = search_index_with_connection(
                    connection,
                    record["claim"],
                    top_k=max_k,
                    operator=match_mode.upper(),
                )
            except ValueError as exc:
                if "searchable word" not in str(exc):
                    raise
                results = []

            retrieved = [
                (item["page_id"], item["sentence_id"])
                for item in results
            ]
            evaluated.append((gold_sets, retrieved))

            if index % 25 == 0 or index == len(samples):
                print(f"Evaluated {index:,}/{len(samples):,} claims...", flush=True)
    finally:
        connection.close()

    metrics = {
        str(k): aggregate_metrics(evaluated, k)
        for k in top_ks
    }

    return {
        "split": split,
        "dataset_file": SPLIT_FILES[split],
        "eligible_claims_in_split": total_eligible,
        "evaluated_claims": len(samples),
        "sample_limit": limit,
        "random_seed": seed,
        "match_mode": match_mode,
        "label_counts_in_sample": dict(sorted(label_counts.items())),
        "index_path": str(index_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 evidence retrieval against FEVER sentence annotations."
    )
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="paper_dev")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of evidence-bearing claims to sample; use 0 for all eligible records.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed.")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Retrieval cutoffs to evaluate (each between 1 and 20).",
    )
    parser.add_argument(
        "--match-mode",
        choices=["or", "and"],
        default="or",
        help="OR is broad matching; AND requires all query terms.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the built SQLite FTS5 index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the full evaluation report as JSON.",
    )
    args = parser.parse_args()

    report = run_evaluation(
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        top_ks=args.top_k,
        index_path=args.index,
        match_mode=args.match_mode,
    )

    print("\nFEVER retrieval evaluation")
    print(f"Split: {report['split']}")
    print(f"Eligible evidence-bearing claims: {report['eligible_claims_in_split']:,}")
    print(f"Evaluated claims: {report['evaluated_claims']:,}")
    print(f"Query mode: {report['match_mode'].upper()}")
    print(f"Sample labels: {report['label_counts_in_sample']}")
    print(f"Elapsed: {report['elapsed_seconds']:.2f} seconds")
    for cutoff, values in report["metrics"].items():
        print(f"\n@{cutoff}")
        print(
            "  Sentence precision: "
            f"{values['precision_at_k'] * 100:.2f}% "
            f"({values['retrieved_gold_sentence_references']}/"
            f"{int(cutoff) * values['claims']})"
        )
        print(
            "  Sentence recall: "
            f"{values['sentence_recall_at_k'] * 100:.2f}% "
            f"({values['retrieved_gold_sentence_references']}/"
            f"{values['gold_sentence_references']})"
        )
        print(
            "  Claim hit rate: "
            f"{values['claim_hit_rate_at_k'] * 100:.2f}% "
            f"({values['claims_with_any_gold_sentence_retrieved']}/"
            f"{values['claims']})"
        )
        print(
            "  Complete evidence-set recall: "
            f"{values['complete_evidence_set_recall_at_k'] * 100:.2f}% "
            f"({values['claims_with_a_complete_evidence_set_retrieved']}/"
            f"{values['claims']})"
        )
        print(
            f"  Claims returning fewer than @{cutoff}: "
            f"{values['claims_returning_fewer_than_k_results']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {args.output}")


if __name__ == "__main__":
    main()
