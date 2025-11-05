# ============================================================================
# src/text_classification/components/model_trainer.py
# ============================================================================
"""Fine-tunes Transformer models for text classification with robust metrics and logging."""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
import torch
from text_classification.logging.logger import logger
from text_classification.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    """Handles model fine-tuning for text classification tasks."""

    def __init__(self, config: ModelTrainerConfig, params: dict):
        self.config = config
        self.params = params
        self.model = None
        self.tokenizer = None
        self.dataset = None

        # Validate FP16 configuration
        if self.params.fp16 and not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA unavailable - using FP32")
            self.params.fp16 = False

    def load_data(self):
        """Load tokenized dataset and prepare train-validation splits."""
        try:
            logger.info(f"Loading tokenized dataset from: {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))

            # Use 'train'/'test' splits from previous stage
            self.dataset = DatasetDict({
                "train": dataset["train"],
                "validation": dataset["test"]
            })

            logger.info(f"Train samples: {len(self.dataset['train'])}")
            logger.info(f"Validation samples: {len(self.dataset['validation'])}")
            return self.dataset
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise


    def load_model_and_tokenizer(self):
        """Load pretrained model and tokenizer for classification."""
        try:
            logger.info(f"Loading tokenizer from: {self.config.tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_dir))

            # Detect number of labels
            labels = self.dataset["train"]["labels"]
            num_labels = len(set(labels))
            logger.info(f"Detected {num_labels} unique labels")

            logger.info(f"Loading model: {self.config.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_labels
            )

            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise


    def compute_metrics(self, eval_pred):
        """Compute classification metrics."""
        try:
            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")
            precision_metric = evaluate.load("precision")
            recall_metric = evaluate.load("recall")

            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            metrics = {
                "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
                "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"],
                "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
                "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
            }

            logger.info(f"Validation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Metric computation failed: {e}")
            return {}


    def train(self):
        """Train the Transformer model using Hugging Face Trainer."""
        try:
            logger.info("Starting model training...")

            # Prepare dataset
            if self.dataset is None:
                self.load_data()

            # Prepare model/tokenizer
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()

            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.params.output_dir),
                num_train_epochs=int(self.params.num_train_epochs),
                per_device_train_batch_size=int(self.params.per_device_train_batch_size),
                per_device_eval_batch_size=int(self.params.per_device_eval_batch_size),
                warmup_steps=int(self.params.warmup_steps),
                weight_decay=float(self.params.weight_decay),
                logging_dir=f"{self.config.root_dir}/logs",
                logging_steps=int(self.params.logging_steps),
                eval_strategy=self.params.eval_strategy,
                eval_steps=int(self.params.eval_steps),
                save_steps=int(self.params.save_steps),
                save_total_limit=int(self.params.save_total_limit),
                learning_rate=float(self.params.learning_rate),  # Already correct
                gradient_accumulation_steps=int(self.params.gradient_accumulation_steps),
                fp16=bool(self.params.fp16) if torch.cuda.is_available() else False,
                load_best_model_at_end=bool(self.params.load_best_model_at_end),
                metric_for_best_model=str(self.params.metric_for_best_model),
                greater_is_better=bool(self.params.greater_is_better),
                report_to=list(self.params.report_to),
                seed=self.config.seed,
            )

            logger.info(f"Training arguments:\n{training_args}")

            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["validation"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )

            # Train
            trainer.train()
            logger.info("Model training completed")

            # Save final model and tokenizer
            final_model_path = Path(self.config.root_dir) / "final_model"
            final_model_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))
            logger.info(f"Final model saved to: {final_model_path}")

            # Save evaluation metrics
            metrics = trainer.evaluate()
            metrics_path = Path(self.config.root_dir) / "training_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Training metrics saved to: {metrics_path}")

            return metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
