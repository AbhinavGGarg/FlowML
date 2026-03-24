"""Orchestrator for the AutoML Pipeline.

This module provides the Orchestrator class that coordinates the
multi-agent AutoML pipeline for machine learning model development.
"""

import logging
from typing import Any, Callable, Optional

import pandas as pd

from agents.data_analyzer_agent import DataAnalyzerAgent
from agents.deployment_agent import DeploymentAgent
from agents.evaluation_agent import EvaluationAgent
from agents.explanation_generator_agent import ExplanationGeneratorAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessor_agent import PreprocessorAgent
from agents.training_agent import TrainingAgent
from config import settings
from core.exceptions import PipelineError
from core.memory_manager import MemoryManager
from core.message import Message
from utils.logger import get_logger


class Orchestrator:
    """Orchestrates the multi-agent AutoML pipeline.

    The orchestrator coordinates agents in sequence:
    1. DataAnalyzerAgent - analyzes dataset
    2. PreprocessorAgent - preprocesses data
    3. FeatureEngineeringAgent - engineers features
    4. ModelSelectionAgent - selects appropriate models
    5. TrainingAgent - trains the model
    6. EvaluationAgent - evaluates model performance
    7. DeploymentAgent - saves model for deployment
    8. ExplanationGeneratorAgent - generates explanations

    The orchestrator maintains memory of all interactions and handles
    error management across the pipeline.

    Attributes:
        memory: The memory manager for storing conversation history.
    """

    def __init__(self) -> None:
        """Initialize the Orchestrator with agent instances."""
        self._logger = get_logger(__name__)
        self.memory = MemoryManager()
        self.stage_results: dict[str, Any] = {}
        self.stage_statuses: dict[str, str] = {}

    def _initialize_agents(self):
        """Initialize all agent instances."""
        self.data_analyzer = DataAnalyzerAgent()
        self.preprocessor = PreprocessorAgent()
        self.feature_engineering = FeatureEngineeringAgent()
        self.model_selection = ModelSelectionAgent()
        self.training = TrainingAgent()
        self.evaluation = EvaluationAgent()
        self.deployment = DeploymentAgent()
        self.explanation = ExplanationGeneratorAgent()

    async def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
        cleanup_func: Optional[Callable[[], None]] = None,
    ) -> dict[str, Any]:
        """Run the complete AutoML pipeline.

        Executes the agent pipeline in sequence to build and evaluate
        a machine learning model.

        Args:
            df: The input DataFrame containing the dataset.
            target_column: The name of the target column.
            task_type: Type of task - "classification" or "regression".
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary containing results from all pipeline stages.

        Raises:
            PipelineError: If any stage of the pipeline fails.
        """
        preview_columns = ", ".join(str(column) for column in df.columns[:6])
        if len(df.columns) > 6:
            preview_columns += ", ..."
        # self._logger.info(
        #     "Starting AutoML pipeline | rows=%s | cols=%s | target=%s | task_type=%s | columns=[%s]",
        #     len(df),
        #     len(df.columns),
        #     target_column,
        #     task_type,
        #     preview_columns,
        # )
        self._initialize_agents()

        try:
            pipeline_config = {
                "test_size": test_size,
                "random_state": random_state,
                "cv_folds": settings.default_cv_folds,
                "enable_multi_model": settings.enable_multi_model,
                "optimize_hyperparameters": settings.enable_hpo,
                "n_trials_hpo": settings.n_hpo_trials,
                "enable_ensemble": settings.enable_ensemble,
                "ensemble_type": settings.ensemble_type,
                "ensemble_top_k": settings.ensemble_top_k,
            }

            # Stage 1: Data Analysis
            # self._logger.info("Stage 1: Analyzing dataset")
            self.stage_statuses["analysis"] = "running"
            analysis_result = await self.data_analyzer.run(df, target_column)
            self.stage_results["analysis"] = analysis_result
            self.stage_statuses["analysis"] = "completed"
            self.memory.add(Message(role="data_analyzer", content=str(analysis_result)))
            # self._logger.info(f"Analysis complete: {analysis_result.get('feature_count', 0)} features")

            # Stage 2: Preprocessing
            # self._logger.info("Stage 2: Preprocessing data")
            self.stage_statuses["preprocessing"] = "running"
            preprocessing_result = await self.preprocessor.run(
                df, analysis_result, target_column, test_size, random_state
            )
            self.stage_results["preprocessing"] = preprocessing_result
            self.stage_statuses["preprocessing"] = "completed"
            self.memory.add(Message(role="preprocessor", content=str(preprocessing_result)))

            # Stage 3: Feature Engineering
            # self._logger.info("Stage 3: Feature engineering")
            self.stage_statuses["features"] = "running"
            features_result = await self.feature_engineering.run(
                df, preprocessing_result, target_column
            )
            self.stage_results["features"] = features_result
            self.stage_statuses["features"] = "completed"
            self.memory.add(Message(role="feature_engineering", content=str(features_result)))
            # self._logger.info(f"Features: {features_result.get('final_feature_count', 0)} selected")

            # Stage 4: Model Selection
            # self._logger.info("Stage 4: Selecting model")
            self.stage_statuses["model_selection"] = "running"
            model_result = await self.model_selection.run(
                df, features_result, target_column, task_type, analysis_result
            )
            self.stage_results["model_selection"] = model_result
            self.stage_statuses["model_selection"] = "completed"
            self.memory.add(Message(role="model_selection", content=str(model_result)))
            # self._logger.info(f"Model selected: {model_result.get('selected_model')}")

            # Stage 5: Training
            # self._logger.info("Stage 5: Training model")
            self.stage_statuses["training"] = "running"
            model_result["target_column"] = target_column
            training_config = {
                **pipeline_config,
                "preprocessing_result": preprocessing_result,
            }
            training_result = await self.training.run(df, model_result, training_config)
            self.stage_results["training"] = training_result
            self.stage_statuses["training"] = "completed"
            self.memory.add(Message(role="training", content=str(training_result)))
            # self._logger.info(f"Training complete: CV score = {training_result.get('best_score', 0):.4f}")

            # Stage 6: Evaluation
            # self._logger.info("Stage 6: Evaluating model")
            self.stage_statuses["evaluation"] = "running"
            evaluation_result = await self.evaluation.run(training_result, task_type)
            self.stage_results["evaluation"] = evaluation_result
            self.stage_statuses["evaluation"] = "completed"
            self.memory.add(Message(role="evaluation", content=str(evaluation_result)))
            if evaluation_result.get("task_type") == "regression":
                self._logger.info(
                    "Evaluation complete: R2 = %.4f | RMSE = %.4f",
                    evaluation_result.get("r2", 0.0),
                    evaluation_result.get("rmse", 0.0),
                )
            else:
                self._logger.info(
                    "Evaluation complete: accuracy = %.4f | f1 = %.4f",
                    evaluation_result.get("accuracy", 0.0),
                    evaluation_result.get("f1", 0.0),
                )

            # Stage 7: Deployment
            # self._logger.info("Stage 7: Deploying model")
            self.stage_statuses["deployment"] = "running"
            deployment_result = await self.deployment.run(training_result, evaluation_result)
            self.stage_results["deployment"] = deployment_result
            self.stage_statuses["deployment"] = "completed"
            self.memory.add(Message(role="deployment", content=str(deployment_result)))

            # Stage 8: Explanation
            # self._logger.info("Stage 8: Generating explanations")
            self.stage_statuses["explanation"] = "running"
            explanation_result = await self.explanation.run(training_result, evaluation_result)
            self.stage_results["explanation"] = explanation_result
            self.stage_statuses["explanation"] = "completed"

            # self._logger.info("Pipeline completed successfully")
            return self.stage_results

        except Exception as e:
            # self._logger.exception(f"Pipeline failed: {e}")
            raise PipelineError(
                f"Pipeline failed: {str(e)}",
                failed_at_stage=list(self.stage_statuses.keys())[-1] if self.stage_statuses else "unknown",
                details={"error": str(e)},
            ) from e
        finally:
            if cleanup_func:
                try:
                    cleanup_func()
                except Exception as cleanup_err:  # pragma: no cover - best effort
                    self._logger.warning("Cleanup after pipeline failed: %s", cleanup_err)

    def get_pipeline_status(self) -> dict[str, str]:
        """Get the current pipeline status."""
        return self.stage_statuses.copy()

    def get_stage_result(self, stage: str) -> Optional[dict[str, Any]]:
        """Get the result of a specific pipeline stage."""
        return self.stage_results.get(stage)

    def run_pipeline_sync(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        cleanup_func: Optional[Callable[[], None]] = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for running the pipeline.

        Args:
            df: The input DataFrame containing the dataset.
            target_column: The name of the target column.
            task_type: Type of task - "classification" or "regression".

        Returns:
            Dictionary containing results from all pipeline stages.
        """
        import asyncio
        return asyncio.run(self.run_pipeline(df, target_column, task_type, cleanup_func=cleanup_func))
