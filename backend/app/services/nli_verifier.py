"""Three-way NLI verification using a pretrained Hugging Face sequence classifier.

The model is loaded lazily on the first verification request. Model weights are
cached by Hugging Face locally and are never stored in the Git repository.
"""
from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Any, Callable, Mapping, Sequence

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_LABELS = ("entailment", "neutral", "contradiction")
VERDICT_BY_NLI = {
    "entailment": "SUPPORTS",
    "neutral": "NOT ENOUGH INFO",
    "contradiction": "REFUTES",
}


class NliModelError(RuntimeError):
    """Raised when the NLI model or its dependencies cannot be loaded."""


class NliVerifier:
    """Score evidence-premise / claim-hypothesis pairs and produce a FEVER verdict."""

    def __init__(
        self,
        model_name: str | None = None,
        predictor: Callable[[Sequence[str], str], Sequence[Mapping[str, float]]] | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("VERIAI_NLI_MODEL", DEFAULT_NLI_MODEL)
        self._predictor = predictor
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._label_to_index: dict[str, int] = {}
        self.device: str | None = "test" if predictor is not None else None

    @property
    def loaded(self) -> bool:
        return self._predictor is not None or self._model is not None

    def status(self) -> dict[str, Any]:
        """Return model/dependency status without downloading or loading model weights."""
        return {
            "model_name": self.model_name,
            "loaded": self.loaded,
            "device": self.device,
            "torch_installed": find_spec("torch") is not None,
            "transformers_installed": find_spec("transformers") is not None,
            "sentencepiece_installed": find_spec("sentencepiece") is not None,
        }

    @staticmethod
    def _normalise_label(label: str) -> str | None:
        value = label.strip().lower().replace("-", "_").replace(" ", "_")
        if "entail" in value:
            return "entailment"
        if "neutral" in value:
            return "neutral"
        if "contradict" in value:
            return "contradiction"
        return None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                use_safetensors=True,
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            self._model.eval()

            raw_labels = getattr(self._model.config, "id2label", {}) or {}
            parsed = {
                int(index): self._normalise_label(str(label))
                for index, label in raw_labels.items()
            }
            resolved = {
                label: index
                for index, label in parsed.items()
                if label in NLI_LABELS
            }

            if set(resolved) == set(NLI_LABELS):
                self._label_to_index = resolved
            elif self.model_name == DEFAULT_NLI_MODEL and self._model.config.num_labels == 3:
                # The model card documents this checkpoint's output order.
                self._label_to_index = {
                    "entailment": 0,
                    "neutral": 1,
                    "contradiction": 2,
                }
            else:
                raise NliModelError(
                    "The selected checkpoint does not expose a recognized "
                    "entailment/neutral/contradiction label mapping."
                )
        except NliModelError:
            self._model = None
            raise
        except Exception as exc:
            self._model = None
            raise NliModelError(
                "Could not load the NLI model. Install backend/requirements-nli.txt "
                "and check your internet connection for the first model download. "
                f"Original error: {exc}"
            ) from exc

    def _predict_probabilities(
        self,
        evidence_texts: Sequence[str],
        claim: str,
    ) -> list[dict[str, float]]:
        if self._predictor is not None:
            raw = self._predictor(evidence_texts, claim)
            probabilities = [dict(item) for item in raw]
        else:
            self._load_model()
            assert self._tokenizer is not None
            assert self._model is not None
            assert self._torch is not None

            tokens = self._tokenizer(
                list(evidence_texts),
                [claim] * len(evidence_texts),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
            with self._torch.inference_mode():
                output = self._model(**tokens)
                rows = self._torch.softmax(output.logits, dim=-1).detach().cpu().tolist()

            probabilities = []
            for row in rows:
                probabilities.append({
                    label: float(row[index])
                    for label, index in self._label_to_index.items()
                })

        if len(probabilities) != len(evidence_texts):
            raise NliModelError("The NLI predictor returned an unexpected number of results.")

        validated: list[dict[str, float]] = []
        for item in probabilities:
            if any(label not in item for label in NLI_LABELS):
                raise NliModelError(
                    "NLI predictions must contain entailment, neutral and contradiction scores."
                )
            values = {label: float(item[label]) for label in NLI_LABELS}
            if any(value < 0.0 or value > 1.0 for value in values.values()):
                raise NliModelError("NLI probabilities must be between 0 and 1.")
            validated.append(values)
        return validated

    def verify(
        self,
        claim: str,
        evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Classify a claim using retrieved evidence; strongest decisive pair wins."""
        if not claim or not claim.strip():
            raise ValueError("claim must not be empty")

        usable_evidence = [
            item for item in evidence
            if isinstance(item.get("text"), str) and item["text"].strip()
        ]
        if not usable_evidence:
            return {
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "confidence_note": "No model score is available because no evidence was retrieved.",
                "decision_basis": "no_evidence_retrieved",
                "evidence_selection_basis": "no_evidence_retrieved",
                "decision_evidence": None,
                "selected_evidence": None,
                "model_name": self.model_name,
                "device": self.device,
                "evidence_scores": [],
            }

        evidence_texts = [str(item["text"]) for item in usable_evidence]
        pair_probabilities = self._predict_probabilities(evidence_texts, claim)

        evidence_scores: list[dict[str, Any]] = []
        decisive_candidates: list[tuple[float, int, str]] = []

        for index, (item, probabilities) in enumerate(zip(usable_evidence, pair_probabilities)):
            predicted_relation = max(probabilities, key=probabilities.get)
            predicted_verdict = VERDICT_BY_NLI[predicted_relation]
            evidence_scores.append({
                "page_id": item.get("page_id"),
                "sentence_id": item.get("sentence_id"),
                "text": item["text"],
                "predicted_relation": predicted_relation,
                "predicted_verdict": predicted_verdict,
                "probabilities": probabilities,
            })
            if predicted_relation in {"entailment", "contradiction"}:
                decisive_candidates.append((
                    probabilities[predicted_relation],
                    index,
                    predicted_relation,
                ))

        if decisive_candidates:
            confidence, winning_index, winning_relation = max(decisive_candidates)
            verdict = VERDICT_BY_NLI[winning_relation]
            decision_basis = "strongest_non_neutral_evidence_pair"
            decision_evidence = evidence_scores[winning_index]

            # Keep verdict aggregation based on the strongest NLI pair, but
            # choose the earliest retrieved pair with the same verdict as
            # the user-facing citation. Retrieval order is the relevance ranking
            # supplied by the subject-page/BM25 pipeline.
            selected_evidence = next(
                (
                    item for item in evidence_scores
                    if item["predicted_verdict"] == verdict
                ),
                decision_evidence,
            )
            evidence_selection_basis = (
                "earliest_retrieved_pair_agreeing_with_verdict"
            )
        else:
            verdict = "NOT ENOUGH INFO"
            confidence = max(
                (item["probabilities"]["neutral"] for item in evidence_scores),
                default=0.0,
            )
            decision_basis = "all_evidence_pairs_predicted_neutral"
            decision_evidence = None
            selected_evidence = None
            evidence_selection_basis = "no_decisive_evidence_pair"

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 6),
            "confidence_note": (
                "Uncalibrated softmax score for the strongest decisive pair "
                "(or strongest neutral pair); not a calibrated probability."
            ),
            "decision_basis": decision_basis,
            "evidence_selection_basis": evidence_selection_basis,
            "model_name": self.model_name,
            "device": self.device,
            "decision_evidence": decision_evidence,
            "selected_evidence": selected_evidence,
            "evidence_scores": evidence_scores,
        }
