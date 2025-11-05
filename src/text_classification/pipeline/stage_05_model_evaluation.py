# ============================================================================
# src/text_classification/pipeline/stage_05_model_evaluation.py
# ============================================================================
"""Pipeline stage for model evaluation."""

from text_classification.config.configuration import ConfigurationManager
from text_classification.components.model_evaluation import ModelEvaluation
from text_classification.logging.logger import logger


class ModelEvaluationPipeline:
    """Runs model evaluation stage."""

    def __init__(self):
        self.stage_name = "Model Evaluation"

    def run(self):
        """Execute model evaluation pipeline."""
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")

            # Load configurations
            config_manager = ConfigurationManager()
            model_eval_config = config_manager.get_model_evaluation_config()

            # Initialize and run evaluation
            evaluator = ModelEvaluation(config=model_eval_config)
            metrics = evaluator.evaluate()

            logger.info(f"Evaluation metrics: {metrics}")
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")

            return metrics

        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise
