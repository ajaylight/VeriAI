from fastapi.testclient import TestClient

from app.main import app
from app.services.nli_verifier import DEFAULT_NLI_MODEL, NliVerifier


class FixedPredictor:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def __call__(self, evidence_texts, claim):
        self.calls += 1
        assert len(evidence_texts) == len(self.outputs)
        return self.outputs


def sample_evidence():
    return [
        {
            "page_id": "Example_Page",
            "sentence_id": 0,
            "text": "The evidence sentence.",
        }
    ]


def test_entailment_maps_to_supports():
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.91, "neutral": 0.06, "contradiction": 0.03}
        ])
    )

    result = verifier.verify("The claim.", sample_evidence())

    assert result["verdict"] == "SUPPORTS"
    assert result["confidence"] == 0.91
    assert result["selected_evidence"]["page_id"] == "Example_Page"


def test_contradiction_maps_to_refutes():
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90}
        ])
    )

    result = verifier.verify("The claim.", sample_evidence())

    assert result["verdict"] == "REFUTES"
    assert result["confidence"] == 0.90


def test_nli_receives_only_sentence_before_fever_link_annotations():
    class CapturingPredictor:
        def __init__(self):
            self.seen = None

        def __call__(self, evidence_texts, claim):
            self.seen = list(evidence_texts)
            return [
                {"entailment": 0.91, "neutral": 0.06, "contradiction": 0.03}
            ]

    predictor = CapturingPredictor()
    verifier = NliVerifier(predictor=predictor)
    evidence = [{
        "page_id": "Example_Page",
        "sentence_id": 0,
        "text": "Robert Reed was an English actor.\tEnglish\tEngland\tactor\tactor",
    }]

    result = verifier.verify("Robert Reed was an actor.", evidence)

    assert predictor.seen == ["Robert Reed was an English actor."]
    assert result["selected_evidence"]["text"] == "Robert Reed was an English actor."


def test_neutral_maps_to_not_enough_info():
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.12, "neutral": 0.81, "contradiction": 0.07}
        ])
    )

    result = verifier.verify("The claim.", sample_evidence())

    assert result["verdict"] == "NOT ENOUGH INFO"
    assert result["confidence"] == 0.81
    assert result["selected_evidence"] is None
    assert result["decision_evidence"] is None
    assert result["evidence_selection_basis"] == "no_decisive_evidence_pair"


def test_strongest_non_neutral_evidence_pair_decides_claim():
    evidence = sample_evidence() + [
        {
            "page_id": "Other_Page",
            "sentence_id": 2,
            "text": "A second evidence sentence.",
        }
    ]
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.72, "neutral": 0.20, "contradiction": 0.08},
            {"entailment": 0.04, "neutral": 0.10, "contradiction": 0.86},
        ])
    )

    result = verifier.verify("The claim.", evidence)

    assert result["verdict"] == "REFUTES"
    assert result["confidence"] == 0.86
    assert result["selected_evidence"]["page_id"] == "Other_Page"


def test_selected_evidence_uses_retrieval_order_when_multiple_pairs_agree():
    evidence = sample_evidence() + [
        {
            "page_id": "More_Confident_Page",
            "sentence_id": 4,
            "text": "A second supporting evidence sentence.",
        }
    ]
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.72, "neutral": 0.20, "contradiction": 0.08},
            {"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005},
        ])
    )

    result = verifier.verify("The claim.", evidence)

    assert result["verdict"] == "SUPPORTS"
    assert result["confidence"] == 0.99
    assert result["decision_evidence"]["page_id"] == "More_Confident_Page"
    assert result["selected_evidence"]["page_id"] == "Example_Page"
    assert result["evidence_selection_basis"] == (
        "earliest_retrieved_pair_agreeing_with_verdict"
    )

def test_no_evidence_returns_not_enough_info_without_loading_model():
    predictor = FixedPredictor([])
    verifier = NliVerifier(predictor=predictor)

    result = verifier.verify("The claim.", [])

    assert result["verdict"] == "NOT ENOUGH INFO"
    assert result["confidence"] == 0.0
    assert result["decision_basis"] == "no_evidence_retrieved"
    assert result["selected_evidence"] is None
    assert result["decision_evidence"] is None
    assert predictor.calls == 0


def test_verification_endpoint_connects_retrieval_and_nli(monkeypatch):
    import app.main as main_module

    predictor = FixedPredictor([
        {"entailment": 0.88, "neutral": 0.08, "contradiction": 0.04}
    ])
    monkeypatch.setattr(
        main_module,
        "search_index",
        lambda claim, top_k, operator: sample_evidence(),
    )
    monkeypatch.setattr(
        main_module,
        "_nli_verifier",
        NliVerifier(predictor=predictor),
    )

    response = TestClient(app).post(
        "/verification/verify",
        json={"claim": "The claim.", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "SUPPORTS"
    assert body["retrieved_evidence_count"] == 1
    assert body["retrieval_mode"] == "subject_page_first_bm25_or_filtered"


def test_verification_status_does_not_load_model_weights():
    response = TestClient(app).get("/verification/status")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == DEFAULT_NLI_MODEL
    assert body["loaded"] is False



def test_unrelated_entity_contradictions_are_ignored_when_subject_page_exists():
    evidence = [
        {
            "page_id": "Shawn_Carlson",
            "sentence_id": 0,
            "text": "Shawn Carlson was born in 1960.",
        },
        {
            "page_id": "Linda_Carlson",
            "sentence_id": 0,
            "text": "Linda Carlson was born May 12, 1945.",
        },
        {
            "page_id": "John_F._Carlson",
            "sentence_id": 0,
            "text": "John F. Carlson was born May 5, 1875.",
        },
        {
            "page_id": "Connie_Nielsen",
            "sentence_id": 0,
            "text": "Connie Nielsen is a Danish actress.",
        },
        {
            "page_id": "Chris_Stills",
            "sentence_id": 36,
            "text": "Chris played the role of Julius Caesar.",
        },
    ]
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.04, "neutral": 0.91, "contradiction": 0.05},
            {"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            {"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98},
            {"entailment": 0.02, "neutral": 0.92, "contradiction": 0.06},
            {"entailment": 0.01, "neutral": 0.10, "contradiction": 0.89},
        ])
    )

    shawn = verifier.verify(
        "Shawn Carlson was born in May of 1960.", evidence[:3]
    )
    connie = verifier.verify(
        "Connie Nielsen played the role of Julius Caesar.", evidence[3:]
    )

    assert shawn["verdict"] == "NOT ENOUGH INFO"
    assert shawn["decision_basis"] == "unrelated_entity_contradictions_ignored"
    assert shawn["evidence_scores"][1]["verdict_eligible"] is False
    assert shawn["evidence_scores"][2]["verdict_eligible"] is False

    assert connie["verdict"] == "NOT ENOUGH INFO"
    assert connie["decision_basis"] == "unrelated_entity_contradictions_ignored"
    assert connie["evidence_scores"][1]["verdict_eligible"] is False


def test_external_contradiction_explicitly_naming_subject_remains_eligible():
    evidence = [
        {
            "page_id": "Connie_Nielsen",
            "sentence_id": 0,
            "text": "Connie Nielsen is a Danish actress.",
        },
        {
            "page_id": "Cast_Interview",
            "sentence_id": 3,
            "text": "Connie Nielsen did not play Julius Caesar.",
        },
    ]
    verifier = NliVerifier(
        predictor=FixedPredictor([
            {"entailment": 0.02, "neutral": 0.92, "contradiction": 0.06},
            {"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
        ])
    )

    result = verifier.verify(
        "Connie Nielsen played the role of Julius Caesar.", evidence
    )

    assert result["verdict"] == "REFUTES"
    assert result["decision_evidence"]["page_id"] == "Cast_Interview"
    assert result["evidence_scores"][1]["verdict_eligible"] is True
