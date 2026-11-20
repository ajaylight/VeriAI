"""Evaluate FEVER evidence retrieval against annotated Wikipedia sentence IDs.

Run from the backend directory:
    python -m app.scripts.evaluate_retrieval --split paper_dev --limit 10 --top-k 5 10 --match-mode or-filtered --show-misses

Modes:
  or          Broad baseline: every claim token is joined with OR.
  or-filtered Filtered baseline: remove common grammatical terms, then OR.
  and         Strict diagnostic: every claim token must match.

The evaluator reuses one SQLite connection for the full batch. Optional miss
diagnostics print gold references alongside the top retrieved sentences.
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
MATCH_MODES = {"or": "OR", "and": "AND", "or-filtered": "OR_FILTERED"}


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
    return {
        "claims": claim_count,
        "gold_sentence_references": gold_sentence_count,
        "retrieved_gold_sentence_references": relevant_sentence_count,
        "precision_at_k": relevant_sentence_count / (k * claim_count),
        "sentence_recall_at_k": relevant_sentence_count / gold_sentence_count if gold_sentence_count else 0.0,
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
    if match_mode not in MATCH_MODES:
        raise ValueError(f"match_mode must be one of: {', '.join(sorted(MATCH_MODES))}.")
    top_ks = sorted(set(top_ks))

    total_eligible, samples = load_evaluation_records(split, limit, seed)
    if not samples:
        raise ValueError(
            f"No FEVER records with sentence-level evidence found in "
            f"{DEFAULT_FEVER_DIR / SPLIT_FILES[split]}."
        )

    label_counts = Counter(str(record.get("label", "UNKNOWN")) for record, _ in samples)
    max_k = max(top_ks)
    evaluated: list[tuple[list[EvidenceSet], list[EvidenceRef]]] = []
    claim_details: list[dict[str, Any]] = []
    started = time.time()

    connection = open_index_connection(index_path)
    try:
        for index, (record, gold_sets) in enumerate(samples, start=1):
            try:
                results = search_index_with_connection(
                    connection,
                    record["claim"],
                    top_k=max_k,
                    operator=MATCH_MODES[match_mode],
                )
            except ValueError as exc:
                if "searchable word" not in str(exc):
                    raise
                results = []

            retrieved = [(item["page_id"], item["sentence_id"]) for item in results]
            evaluated.append((gold_sets, retrieved))

            retrieved_set = set(retrieved[:max_k])
            gold_union: EvidenceSet = set().union(*gold_sets)
            claim_details.append({
                "id": record.get("id"),
                "label": record.get("label"),
                "claim": record["claim"],
                "gold_evidence_sets": [
                    [list(reference) for reference in sorted(group)]
                    for group in gold_sets
                ],
                "retrieved": [
                    {
                        "page_id": item["page_id"],
                        "sentence_id": item["sentence_id"],
                        "text": item["text"],
                        "bm25_score": item["bm25_score"],
                    }
                    for item in results
                ],
                "hit_at_max_k": bool(gold_union.intersection(retrieved_set)),
                "complete_evidence_set_at_max_k": any(
                    group.issubset(retrieved_set) for group in gold_sets
                ),
            })

            if index % 25 == 0 or index == len(samples):
                print(f"Evaluated {index:,}/{len(samples):,} claims...", flush=True)
    finally:
        connection.close()

    metrics = {str(k): aggregate_metrics(evaluated, k) for k in top_ks}
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
        "claim_details": claim_details,
    }


def print_miss_diagnostics(report: dict[str, Any], display_k: int = 5) -> None:
    """Print misses at the largest requested cutoff with gold and retrieved evidence."""
    misses = [item for item in report["claim_details"] if not item["hit_at_max_k"]]
    print(f"\nMissed claims at @{max(map(int, report['metrics']))}: {len(misses)}")
    for item in misses:
        print("\n" + "-" * 78)
        print(f"Claim ID: {item['id']} | Label: {item['label']}")
        print(f"Claim: {item['claim']}")
        print("Gold evidence alternatives:")
        for group in item["gold_evidence_sets"]:
            print("  - " + ", ".join(f"{page}#{sentence}" for page, sentence in group))
        print(f"Top {display_k} retrieved sentences:")
        for result in item["retrieved"][:display_k]:
            snippet = " ".join(result["text"].split())
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            print(
                f"  - {result['page_id']}#{result['sentence_id']}: {snippet}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 evidence retrieval against FEVER sentence annotations."
    )
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="paper_dev")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--match-mode", choices=sorted(MATCH_MODES), default="or")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print gold references and top retrieved sentences for claims missed at max(top-k).",
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
            f"  Sentence precision: {values['precision_at_k'] * 100:.2f}% "
            f"({values['retrieved_gold_sentence_references']}/"
            f"{int(cutoff) * values['claims']})"
        )
        print(
            f"  Sentence recall: {values['sentence_recall_at_k'] * 100:.2f}% "
            f"({values['retrieved_gold_sentence_references']}/"
            f"{values['gold_sentence_references']})"
        )
        print(
            f"  Claim hit rate: {values['claim_hit_rate_at_k'] * 100:.2f}% "
            f"({values['claims_with_any_gold_sentence_retrieved']}/"
            f"{values['claims']})"
        )
        print(
            f"  Complete evidence-set recall: "
            f"{values['complete_evidence_set_recall_at_k'] * 100:.2f}% "
            f"({values['claims_with_a_complete_evidence_set_retrieved']}/"
            f"{values['claims']})"
        )
        print(
            f"  Claims returning fewer than @{cutoff}: "
            f"{values['claims_returning_fewer_than_k_results']}"
        )

    if args.show_misses:
        print_miss_diagnostics(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {args.output}")


if __name__ == "__main__":
    main()
