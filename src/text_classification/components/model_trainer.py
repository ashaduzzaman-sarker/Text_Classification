# ============================================================================
# src/text_classification/components/model_trainer.py
# ============================================================================
"""Fine-tunes Transformer models for text classification with LoRA/QLoRA.

This trainer supports standard full fine-tuning as well as parameter-efficient
fine-tuning via LoRA/QLoRA using the ``peft`` library. It also logs key
metrics to MLflow for experiment tracking.
"""

import json
from pathlib import Path
from typing import Optional

import evaluate
import mlflow
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from text_classification.logging.logger import logger
from text_classification.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    """Handles model fine-tuning for text classification tasks."""

    def __init__(self, config: ModelTrainerConfig, training_params, lora_params: Optional[object] = None):
        self.config = config
        self.training_params = training_params
        self.lora_params = lora_params
        self.model = None
        self.tokenizer = None
        self.dataset = None

        # Validate FP16 configuration
        if getattr(self.training_params, "fp16", False) and not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA unavailable - using FP32")
            self.training_params.fp16 = False

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

            logger.info(f"Loading base model: {self.config.model_name}")

            # ------------------------------------------------------------------
            # LoRA / QLoRA-aware model loading
            # ------------------------------------------------------------------
            use_lora = bool(getattr(self.lora_params, "enabled", False)) if self.lora_params is not None else False
            use_qlora = bool(getattr(self.lora_params, "use_qlora", False)) if self.lora_params is not None else False

            if use_lora and use_qlora:
                # QLoRA (4-bit quantisation) path
                logger.info("Initialising model with QLoRA (4-bit quantisation)")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels,
                    quantization_config=bnb_config,
                    device_map="auto",
                )

                base_model = prepare_model_for_kbit_training(base_model)
            else:
                # Standard full-precision model (optionally with LoRA on top)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels,
                )

            if use_lora:
                logger.info("Wrapping base model with LoRA adapters")
                lora_config = LoraConfig(
                    r=int(getattr(self.lora_params, "r", 8)),
                    lora_alpha=int(getattr(self.lora_params, "lora_alpha", 16)),
                    lora_dropout=float(getattr(self.lora_params, "lora_dropout", 0.05)),
                    bias=str(getattr(self.lora_params, "bias", "none")),
                    task_type=TaskType.SEQ_CLS,
                )
                self.model = get_peft_model(base_model, lora_config)
                self.model.print_trainable_parameters()
            else:
                self.model = base_model

            logger.info("Model and tokenizer loaded successfully (LoRA enabled: %s, QLoRA enabled: %s)", use_lora, use_qlora)
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
            tp = self.training_params
            training_args = TrainingArguments(
                output_dir=str(tp.output_dir),
                num_train_epochs=int(tp.num_train_epochs),
                per_device_train_batch_size=int(tp.per_device_train_batch_size),
                per_device_eval_batch_size=int(tp.per_device_eval_batch_size),
                warmup_steps=int(tp.warmup_steps),
                weight_decay=float(tp.weight_decay),
                logging_dir=f"{self.config.root_dir}/logs",
                logging_steps=int(tp.logging_steps),
                eval_strategy=tp.eval_strategy,
                eval_steps=int(tp.eval_steps),
                save_steps=int(tp.save_steps),
                save_total_limit=int(tp.save_total_limit),
                learning_rate=float(tp.learning_rate),
                gradient_accumulation_steps=int(tp.gradient_accumulation_steps),
                fp16=bool(tp.fp16) if torch.cuda.is_available() else False,
                load_best_model_at_end=bool(tp.load_best_model_at_end),
                metric_for_best_model=str(tp.metric_for_best_model),
                greater_is_better=bool(tp.greater_is_better),
                report_to=list(tp.report_to),
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

            # ------------------------------------------------------------------
            # MLflow experiment tracking
            # ------------------------------------------------------------------
            mlflow.set_tracking_uri(
                # Default to local ./mlruns if env var not set
                mlflow.get_tracking_uri() or "file:./mlruns"
            )
            mlflow.set_experiment(
                experiment_name="text_classification_lora"
            )

            with mlflow.start_run():
                # Log key hyperparameters
                mlflow.log_params({
                    "model_name": self.config.model_name,
                    "seed": self.config.seed,
                    "use_lora": bool(getattr(self.lora_params, "enabled", False)) if self.lora_params is not None else False,
                    "use_qlora": bool(getattr(self.lora_params, "use_qlora", False)) if self.lora_params is not None else False,
                    "num_train_epochs": int(tp.num_train_epochs),
                    "per_device_train_batch_size": int(tp.per_device_train_batch_size),
                    "learning_rate": float(tp.learning_rate),
                })

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

                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
