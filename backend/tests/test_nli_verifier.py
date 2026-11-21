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


def test_no_evidence_returns_not_enough_info_without_loading_model():
    predictor = FixedPredictor([])
    verifier = NliVerifier(predictor=predictor)

    result = verifier.verify("The claim.", [])

    assert result["verdict"] == "NOT ENOUGH INFO"
    assert result["confidence"] == 0.0
    assert result["decision_basis"] == "no_evidence_retrieved"
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
