# ============================================================================
# src/text_classification/components/model_evaluation.py
# ============================================================================
"""Evaluates the fine-tuned text classification model."""

import json
import numpy as np
from pathlib import Path
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from text_classification.logging.logger import logger
from text_classification.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    """Handles evaluation of the trained model using 🤗 evaluate."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.test_dataset = None

    def load_model_and_tokenizer(self):
        """Load fine-tuned model and tokenizer."""
        try:
            logger.info(f"Loading fine-tuned model from: {self.config.model_dir}")
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.config.model_dir))

            logger.info(f"Loading tokenizer from: {self.config.tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_dir))

            logger.info("Model and tokenizer loaded successfully for evaluation.")
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise

    def load_test_data(self):
        """Load the test dataset."""
        try:
            logger.info(f"Loading test dataset from: {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            if "test" not in dataset:
                raise ValueError("Dataset must contain a 'test' split.")
            self.test_dataset = dataset["test"]
            logger.info(f"Loaded test dataset with {len(self.test_dataset)} samples.")
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            raise

    def evaluate(self):
        """Run evaluation and compute metrics using evaluate."""
        try:
            logger.info("Starting model evaluation...")

            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            if self.test_dataset is None:
                self.load_test_data()

            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )

            logger.info("Generating predictions...")
            predictions = trainer.predict(self.test_dataset)
            preds = np.argmax(predictions.predictions, axis=-1)
            labels = predictions.label_ids

            # Evaluate metrics using evaluate
            metrics = {}
            metrics["accuracy"] = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
            metrics["precision"] = evaluate.load("precision").compute(predictions=preds, references=labels, average="weighted")["precision"]
            metrics["recall"] = evaluate.load("recall").compute(predictions=preds, references=labels, average="weighted")["recall"]
            metrics["f1"] = evaluate.load("f1").compute(predictions=preds, references=labels, average="weighted")["f1"]

            # Confusion Matrix 
            cm_metric = evaluate.load("confusion_matrix")
            cm = cm_metric.compute(predictions=preds, references=labels)["confusion_matrix"]
            self._plot_confusion_matrix(cm, Path(self.config.root_dir) / "confusion_matrix.png")

            # Save metrics
            metrics_path = Path(self.config.root_dir) / ".json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Evaluation metrics saved to: {metrics_path}")
            logger.info(f"Metrics: {metrics}")

            logger.info("Model evaluation completed successfully.")
            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

    def _plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix."""
        try:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Confusion matrix saved to: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to plot confusion matrix: {e}")
