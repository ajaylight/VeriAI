from app.scripts.evaluate_retrieval import aggregate_metrics, extract_gold_evidence_sets


def test_extract_gold_evidence_sets_keeps_alternative_sentence_groups() -> None:
    evidence = [
        [[0, 0, "Page_A", 1], [0, 1, "Page_A", 2]],
        [[1, 0, "Page_B", 3]],
        [[2, 0, None, None]],
    ]

    result = extract_gold_evidence_sets(evidence)

    assert result == [
        {("Page_A", 1), ("Page_A", 2)},
        {("Page_B", 3)},
    ]


def test_aggregate_metrics_counts_sentence_hits_and_complete_alternative_sets() -> None:
    page_a_1 = ("Page_A", 1)
    page_a_2 = ("Page_A", 2)
    page_b_3 = ("Page_B", 3)
    page_c_4 = ("Page_C", 4)
    page_c_5 = ("Page_C", 5)

    evaluated = [
        (
            [{page_a_1, page_a_2}, {page_b_3}],
            [page_a_1, page_a_2, ("Distractor", 0)],
        ),
        (
            [{page_c_4, page_c_5}],
            [page_c_4, ("Distractor", 1)],
        ),
    ]

    result = aggregate_metrics(evaluated, k=2)

    assert result["claims"] == 2
    assert result["gold_sentence_references"] == 5
    assert result["retrieved_gold_sentence_references"] == 3
    assert result["precision_at_k"] == 0.75
    assert result["sentence_recall_at_k"] == 0.6
    assert result["claim_hit_rate_at_k"] == 1.0
    assert result["complete_evidence_set_recall_at_k"] == 0.5
