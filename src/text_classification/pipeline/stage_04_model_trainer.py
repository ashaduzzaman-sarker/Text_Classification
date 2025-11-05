# ============================================================================
# src/text_classification/pipeline/stage_04_model_trainer.py
# ============================================================================
"""Model training pipeline stage for text classification."""

from text_classification.config.configuration import ConfigurationManager
from text_classification.components.model_trainer import ModelTrainer
from text_classification.logging.logger import logger


class ModelTrainerPipeline:
    """Pipeline stage for model training."""

    def __init__(self):
        self.stage_name = "Model Training"

    def run(self) -> dict:
        """Execute model training pipeline."""
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")

            # Load configs
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            training_params = config_manager.params.TrainingArguments

            # Initialize trainer and train
            trainer = ModelTrainer(
                config=model_trainer_config,
                params=training_params
            )
            train_metrics = trainer.train()

            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return train_metrics

        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise
