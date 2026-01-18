"""Prediction pipeline for serving text classification models.

This module provides a lightweight, production-oriented wrapper for loading
the fine-tuned Hugging Face model from the artifacts directory and running
inference for incoming text inputs. It is intentionally framework-agnostic so
it can be used both from FastAPI (REST), Streamlit, notebooks, or batch jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_classification.config.configuration import ConfigurationManager
from text_classification.logging.logger import logger


class PredictionPipeline:
    """Load a fine-tuned model and run predictions on raw text.

    The pipeline reads configuration from config/config.yaml via
    ConfigurationManager and expects the trained model artifacts to be
    available under ``artifacts/model_trainer/final_model`` (or the
    configured path).
    """

    def __init__(self) -> None:
        self._config_manager = ConfigurationManager()
        self._model_config = self._config_manager.get_model_evaluation_config()

        self._model_dir: Path = self._model_config.model_dir
        self._tokenizer_dir: Path = self._model_config.tokenizer_dir

        self._model = None
        self._tokenizer = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_artifacts(self) -> None:
        """Lazily load model and tokenizer from disk.

        This method is idempotent and safe to call multiple times.
        """

        if self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading model and tokenizer for inference...")

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self._model_dir}. "
                "Make sure training has completed and artifacts are available."
            )

        if not self._tokenizer_dir.exists():
            raise FileNotFoundError(
                f"Tokenizer directory not found: {self._tokenizer_dir}. "
                "Make sure data transformation/training has completed."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self._model_dir)
        )

        self._model.to(self._device)
        self._model.eval()

        logger.info(
            "Model and tokenizer loaded successfully for prediction on %s",
            self._device,
        )

    def predict(self, texts: Sequence[str]) -> List[dict]:
        """Run prediction on a batch of texts.

        Args:
            texts: A sequence of raw text strings.

        Returns:
            List of dictionaries, one per input text, each containing:
                - ``label``: predicted class label (string if available)
                - ``score``: confidence score for the predicted label
        """

        if isinstance(texts, str):  # type: ignore[unreachable]
            # Defensive check in case a single string is passed accidentally
            texts = [texts]

        if not texts:
            return []

        self._load_artifacts()

        assert self._tokenizer is not None
        assert self._model is not None

        logger.info("Running prediction on %d texts", len(texts))

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)

        id2label = getattr(self._model.config, "id2label", None) or {}

        results: List[dict] = []
        for idx, (pred_id, score) in enumerate(zip(preds.tolist(), scores.tolist())):
            label = id2label.get(pred_id, str(pred_id))
            results.append({
                "text": texts[idx],
                "label": label,
                "score": float(score),
            })

        return results
