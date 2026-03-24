"""Tests for the agent modules.

This module contains tests for the base agent and specific agent
implementations for the AutoML pipeline.
"""

import os
import re
import warnings

# Disable joblib memory mapping to prevent resource tracker warnings
os.environ['JOBLIB_MMAP_MODE'] = ''

# Suppress joblib resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked .* objects", category=UserWarning)
warnings.filterwarnings("ignore", message="resource_tracker: .*FileNotFoundError", category=UserWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base_agent import BaseAgent
from agents.data_analyzer_agent import DataAnalyzerAgent
from agents.preprocessor_agent import PreprocessorAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.deployment_agent import DeploymentAgent
from agents.report_generator import ReportGenerator
from core.exceptions import AgentExecutionError


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_base_agent_initialization(self) -> None:
        """Test that BaseAgent can be initialized."""

        class TestAgent(BaseAgent):
            async def execute(self, *args, **kwargs):
                return "test"

        agent = TestAgent(name="test_agent")
        assert agent.name == "test_agent"

    @pytest.mark.asyncio
    async def test_base_agent_run_wraps_execute(self) -> None:
        """Test that run() wraps execute() with error handling."""

        class TestAgent(BaseAgent):
            async def execute(self, *args, **kwargs):
                return f"processed: {args[0]}"

        agent = TestAgent(name="test")
        result = await agent.run("input")
        assert result == "processed: input"

    @pytest.mark.asyncio
    async def test_base_agent_run_handles_execution_error(self) -> None:
        """Test that run() handles execution errors."""

        class TestAgent(BaseAgent):
            async def execute(self, *args, **kwargs):
                raise ValueError("Test error")

        agent = TestAgent(name="test")
        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.run("input")

        assert "Test error" in str(exc_info.value)


class TestDataAnalyzerAgent:
    """Tests for the DataAnalyzerAgent class."""

    def test_data_analyzer_initialization(self) -> None:
        """Test DataAnalyzerAgent initialization."""
        agent = DataAnalyzerAgent()
        assert agent.name == "DataAnalyzer"

    @pytest.mark.asyncio
    async def test_data_analyzer_analyzes_dataframe(self) -> None:
        """Test that agent analyzes a DataFrame correctly."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "category": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        })

        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        assert result["row_count"] == 5
        assert result["column_count"] == 4
        assert result["feature_count"] == 3
        assert "numeric_columns" in result
        assert "categorical_columns" in result

    @pytest.mark.asyncio
    async def test_data_analyzer_handles_missing_values(self) -> None:
        """Test that agent handles missing values."""
        df = pd.DataFrame({
            "feature1": [1, 2, np.nan, 4, 5],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0],
        })

        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        assert "missing_values" in result
        assert result["missing_values"]["feature1"] == 0.2

    @pytest.mark.asyncio
    async def test_data_analyzer_returns_data_quality_key(self) -> None:
        """Test that data_quality and quality_flags keys are present in result."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0],
        })
        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        assert "data_quality" in result
        assert "quality_flags" in result
        dq = result["data_quality"]
        assert "duplicate_rows" in dq
        assert "missing_rows_pct" in dq
        assert "outlier_columns_count" in dq
        assert isinstance(result["quality_flags"], list)

    @pytest.mark.asyncio
    async def test_data_analyzer_detects_duplicates(self) -> None:
        """Test that the agent correctly counts duplicate rows."""
        df = pd.DataFrame({
            "feature1": [1, 1, 1, 2, 3],
            "feature2": [10.0, 10.0, 10.0, 20.0, 30.0],
            "target": [0, 0, 0, 1, 1],
        })
        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        dq = result["data_quality"]
        # rows 0, 1, 2 are all identical — 2 duplicates
        assert dq["duplicate_rows"] == 2
        assert dq["duplicate_pct"] == pytest.approx(40.0)

    @pytest.mark.asyncio
    async def test_data_analyzer_detects_high_cardinality(self) -> None:
        """Test that high-cardinality categorical columns are flagged."""
        unique_cats = [str(i) for i in range(100)]
        df = pd.DataFrame({
            "id_col": unique_cats,
            "numeric": list(range(100)),
            "target": [i % 2 for i in range(100)],
        })
        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        dq = result["data_quality"]
        assert dq["high_cardinality_count"] >= 1
        assert "id_col" in dq["high_cardinality_columns"]

    @pytest.mark.asyncio
    async def test_data_analyzer_detects_placeholder_invalids(self) -> None:
        """Test that placeholder sentinel values (-999) are detected in numeric columns."""
        df = pd.DataFrame({
            "price": [-999, -999, -999, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "area": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        dq = result["data_quality"]
        assert dq["placeholder_invalid_count"] >= 1
        assert "price" in dq["placeholder_invalid_columns"]

    @pytest.mark.asyncio
    async def test_data_analyzer_quality_flags_have_severity(self) -> None:
        """Test that quality flags include severity, message, and field keys."""
        df = pd.DataFrame({
            "a": [1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
            "b": [10.0] * 10,
            "target": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        for flag in result["quality_flags"]:
            assert "severity" in flag
            assert "message" in flag
            assert "field" in flag
            assert flag["severity"] in {"high", "medium", "low"}


class TestPreprocessorAgent:
    """Tests for the PreprocessorAgent class."""

    def test_preprocessor_initialization(self) -> None:
        """Test PreprocessorAgent initialization."""
        agent = PreprocessorAgent()
        assert agent.name == "Preprocessor"

    @pytest.mark.asyncio
    async def test_preprocessor_handles_categorical_columns(self) -> None:
        """Test that preprocessor handles categorical columns."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        })

        analysis = {
            "numeric_columns": ["feature1"],
            "categorical_columns": ["category"],
        }

        agent = PreprocessorAgent()
        result = await agent.execute(df, analysis, "target")

        assert result["train_size"] > 0
        assert result["test_size"] > 0
        assert "missing_summary" in result
        assert "categorical_summary" in result
        assert "scaling_summary" in result

    @pytest.mark.asyncio
    async def test_preprocessor_applies_automl_style_rules(self) -> None:
        """Test that preprocessing makes deterministic AutoML-style decisions."""
        rows = 1400
        rng = np.random.default_rng(21)
        rare_values = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"], dtype=object)
        rare_city = rare_values[rng.integers(0, len(rare_values), size=rows)]
        rare_city[:24] = np.array([f"rare_{index % 6}" for index in range(24)], dtype=object)

        df = pd.DataFrame({
            "Unnamed: 0": np.arange(rows),
            "customer_id": [f"CUST-{100000 + index}" for index in range(rows)],
            "value": rng.lognormal(mean=2.0, sigma=1.3, size=rows),
            "extreme": rng.normal(loc=10.0, scale=2.0, size=rows),
            "category_low": rng.choice(["bronze", "silver", "gold"], size=rows, p=[0.5, 0.3, 0.2]).astype(object),
            "zip_code": [f"ZIP-{index % 240}" for index in range(rows)],
            "rare_city": rare_city,
            "mostly_missing": np.where(np.arange(rows) % 10 == 0, rng.normal(size=rows), np.nan),
            "event_date": pd.date_range("2024-01-01", periods=rows, freq="D").astype(str),
        })
        df.loc[:19, "value"] = np.nan
        df.loc[:19, "category_low"] = None
        df.loc[:9, "extreme"] = df.loc[:9, "extreme"] * 40
        df["target"] = np.where(df["extreme"].fillna(0) > np.nanmedian(df["extreme"]), "yes", "no")

        analysis = {
            "numeric_columns": ["value", "extreme"],
            "categorical_columns": ["category_low", "zip_code", "rare_city", "mostly_missing", "event_date"],
        }

        agent = PreprocessorAgent()
        result = await agent.execute(df, analysis, "target")

        drop_lookup = {item["column"]: item["reason"] for item in result["dropped_columns"]}
        assert drop_lookup["Unnamed: 0"] == "identifier"
        assert drop_lookup["customer_id"] == "identifier"
        assert drop_lookup["mostly_missing"] == "sparse"
        assert result["missing_summary"]["strategy_used"] == "mixed"
        assert result["missing_summary"]["dropped_rows_count"] == 20
        assert "zip_code" in result["categorical_summary"]["high_cardinality_columns"]
        assert "rare_city" in result["categorical_summary"]["rare_category_grouped_columns"]
        assert result["scaling_summary"]["scaler"] == "RobustScaler"
        assert "value" in result["transform_summary"]["log_transformed_columns"]
        assert "event_date" in result["datetime_columns"]
        assert result["target_summary"]["task_type"] == "classification"
        assert result["transformed_feature_count"] >= result["feature_count_after_column_drops"]
        assert isinstance(result["_X_train_transformed"], pd.DataFrame)
        assert result["_X_train_transformed"].isnull().sum().sum() == 0

    @pytest.mark.asyncio
    async def test_preprocessor_prefers_imputation_on_small_datasets(self) -> None:
        """Test that small datasets favor imputation over dropping many rows."""
        df = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0],
            "cat": ["a", None, "b", "a", None, "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        })
        analysis = {
            "numeric_columns": ["num"],
            "categorical_columns": ["cat"],
        }

        agent = PreprocessorAgent()
        result = await agent.execute(df, analysis, "target", test_size=0.25, random_state=7)

        assert result["missing_summary"]["strategy_used"] == "impute"
        assert result["missing_summary"]["dropped_rows_count"] == 0
        assert result["missing_summary"]["imputed_numeric_columns"] == ["num"]
        assert result["missing_summary"]["imputed_categorical_columns"] == ["cat"]


class TestFeatureEngineeringAgent:
    """Tests for the FeatureEngineeringAgent class."""

    def test_feature_engineering_initialization(self) -> None:
        """Test FeatureEngineeringAgent initialization."""
        agent = FeatureEngineeringAgent()
        assert agent.name == "FeatureEngineering"

    @pytest.mark.asyncio
    async def test_feature_engineering_selects_features(self) -> None:
        """Test that agent selects features."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 10,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
            "feature3": [2, 3, 4, 5, 6] * 10,
            "target": [0, 1, 0, 1, 0] * 10,
        })

        preprocessing = {
            "numeric_columns": ["feature1", "feature2", "feature3"],
            "categorical_columns": [],
        }

        agent = FeatureEngineeringAgent()
        result = await agent.execute(df, preprocessing, "target", n_features_to_select=2)

        assert result["final_feature_count"] >= 1
        assert "target" not in result["selected_features"]
        assert "target" not in result["feature_scores"]

    @pytest.mark.asyncio
    async def test_feature_engineering_tracks_transformations_and_internal_dataset(self) -> None:
        """Test deterministic feature engineering behavior."""
        rng = np.random.default_rng(42)
        rows = 80
        signal = rng.lognormal(mean=1.2, sigma=1.0, size=rows)
        support = rng.normal(loc=2.0, scale=0.5, size=rows)
        x = rng.uniform(1.0, 5.0, size=rows)
        y_coord = rng.uniform(1.0, 5.0, size=rows)
        z = rng.uniform(1.0, 5.0, size=rows)
        category = np.where(signal > np.median(signal), "high", "low").astype(object)
        category[::9] = None

        df = pd.DataFrame({
            "Unnamed: 0": np.arange(rows),
            "row_id": np.arange(1000, 1000 + rows),
            "signal": signal,
            "signal_copy": signal * 1.001,
            "support": support,
            "constant_noise": np.ones(rows),
            "x": x,
            "y": y_coord,
            "z": z,
            "category": category,
            "target": (signal * 3.0) + (support * 1.5) + (x * y_coord * z * 0.05),
        })
        df.loc[::7, "signal"] = np.nan
        df.loc[::11, "category"] = None

        preprocessing = {
            "numeric_columns": ["signal", "signal_copy", "support", "constant_noise", "x", "y", "z"],
            "categorical_columns": ["category"],
        }

        agent = FeatureEngineeringAgent()
        result = await agent.execute(df, preprocessing, "target")

        transformation_types = {item["type"] for item in result["applied_transformations"]}
        engineered_df = result["_engineered_df"]
        llm_explanations = result["llm_explanations"]

        assert "Unnamed: 0" in result["dropped_columns"]
        assert "row_id" in result["dropped_columns"]
        assert "signal_copy" in result["dropped_columns"]
        assert "constant_noise" in result["dropped_columns"]
        assert "x__mul__y__mul__z" in result["generated_features"]
        assert any("__div__" in name for name in result["generated_features"])
        assert {"drop_index_like_columns", "numeric_imputation", "categorical_imputation", "log1p_transform", "generated_interactions", "correlation_filter", "feature_importance_filter"} <= transformation_types
        assert all(column in engineered_df.columns for column in result["selected_features"])
        assert engineered_df[result["selected_features"]].isnull().sum().sum() == 0
        assert "signal" in result["feature_scores"]
        assert "x__mul__y__mul__z" in engineered_df.columns
        assert "target" not in result["selected_features"]
        assert "target" not in result["feature_scores"]
        assert "stageSummary" in llm_explanations
        assert "whatHappened" in llm_explanations
        assert "whyItMattered" in llm_explanations
        assert "keyTakeaway" in llm_explanations
        assert isinstance(llm_explanations["llmUsed"], bool)
        assert isinstance(llm_explanations["featureExplanations"], dict)
        assert isinstance(llm_explanations["droppedFeatureExplanations"], dict)
        assert set(llm_explanations["featureExplanations"]).issubset(set(result["feature_scores"]).difference({"target"}))
        assert set(llm_explanations["droppedFeatureExplanations"]).issubset(set(result["dropped_columns"]))

    @pytest.mark.asyncio
    async def test_feature_engineering_limits_interactions_to_top_variance_features(self) -> None:
        """Test that interaction generation is bounded to top-variance numeric columns."""
        rows = 60
        df = pd.DataFrame({
            "n1": np.linspace(0, 100, rows),
            "n2": np.linspace(10, 80, rows),
            "n3": np.linspace(5, 45, rows),
            "n4": np.linspace(1, 21, rows),
            "n5": np.linspace(2, 12, rows),
            "n6": np.linspace(3, 6, rows),
            "n7": np.linspace(4, 5, rows),
            "target": np.linspace(0, 10, rows),
        })

        preprocessing = {
            "numeric_columns": ["n1", "n2", "n3", "n4", "n5", "n6", "n7"],
            "categorical_columns": [],
        }

        agent = FeatureEngineeringAgent()
        result = await agent.execute(df, preprocessing, "target")

        interaction_columns = [
            column for column in result["generated_features"]
            if "__mul__" in column or "__div__" in column
        ]
        expected_sources = {"n1", "n2", "n3", "n4", "n5"}

        assert len(interaction_columns) <= 20
        assert interaction_columns
        assert all(
            set(re.split(r"__(?:mul|div)__", column)).issubset(expected_sources)
            for column in interaction_columns
        )

    @pytest.mark.asyncio
    async def test_feature_engineering_preserves_base_features_for_selected_interactions(self) -> None:
        """Test that selected interactions keep their source features."""
        rng = np.random.default_rng(7)
        rows = 200
        age = rng.integers(18, 65, size=rows).astype(float)
        education_num = rng.integers(8, 20, size=rows).astype(float)
        noise = rng.normal(0, 1, size=rows)

        df = pd.DataFrame({
            "age": age,
            "educational-num": education_num,
            "hours-per-week": rng.integers(20, 60, size=rows).astype(float),
            "noise": noise,
            "target": age * education_num + noise,
        })

        preprocessing = {
            "numeric_columns": ["age", "educational-num", "hours-per-week", "noise"],
            "categorical_columns": [],
        }

        agent = FeatureEngineeringAgent()
        result = await agent.execute(df, preprocessing, "target")

        interaction_name = "age__mul__educational-num"
        if interaction_name in result["selected_features"]:
            assert "age" in result["selected_features"]
            assert "educational-num" in result["selected_features"]


class TestModelSelectionAgent:
    """Tests for the ModelSelectionAgent class."""

    def test_model_selection_initialization(self) -> None:
        """Test ModelSelectionAgent initialization."""
        agent = ModelSelectionAgent()
        assert agent.name == "ModelSelection"

    @pytest.mark.asyncio
    async def test_model_selection_recommends_model(self) -> None:
        """Test that agent recommends a model."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 10,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
            "target": [0, 1, 0, 1, 0] * 10,
        })

        features = {
            "final_feature_count": 2,
            "numeric_features": ["feature1", "feature2"],
        }

        agent = ModelSelectionAgent()
        result = await agent.execute(df, features, "target", "classification")

        assert "top_candidates" in result
        assert len(result["top_candidates"]) <= 3
        assert "selection_reasoning" in result
        assert "fixed_params" in result["top_candidates"][0]
        assert "search_space" in result["top_candidates"][0]


class TestTrainingAgent:
    """Tests for the TrainingAgent class."""

    def test_training_initialization(self) -> None:
        """Test TrainingAgent initialization."""
        agent = TrainingAgent()
        assert agent.name == "Training"

    @pytest.mark.asyncio
    async def test_training_trains_model(self) -> None:
        """Test that agent trains a model."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })

        model_selection = {
            "top_candidates": [
                {
                    "priority": 1,
                    "model_name": "RandomForest",
                    "model_family": "tree_ensemble",
                    "reasoning": "Primary recommendation.",
                    "fixed_params": {"n_estimators": 10, "random_state": 42},
                    "search_space": {},
                }
            ],
            "task_type": "classification",
            "target_column": "target",
        }

        agent = TrainingAgent()
        result = await agent.execute(df, model_selection, {"test_size": 0.2, "random_state": 42})

        assert "model" in result
        assert "best_score" in result

    @pytest.mark.asyncio
    async def test_training_uses_selected_engineered_features(self) -> None:
        """Test that training uses only selected engineered features."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })
        engineered_df = pd.DataFrame({
            "feature1": df["feature1"],
            "ignored_feature": df["feature2"] * 100,
        })

        model_selection = {
            "top_candidates": [
                {
                    "priority": 1,
                    "model_name": "RandomForest",
                    "model_family": "tree_ensemble",
                    "reasoning": "Primary recommendation.",
                    "fixed_params": {"n_estimators": 10, "random_state": 42},
                    "search_space": {},
                }
            ],
            "task_type": "classification",
            "target_column": "target",
            "selected_features": ["feature1"],
            "_engineered_df": engineered_df,
        }

        agent = TrainingAgent()
        result = await agent.execute(df, model_selection, {"test_size": 0.2, "random_state": 42})

        assert list(result["X_train"].columns) == ["feature1"]
        assert result["selected_features"] == ["feature1"]

    @pytest.mark.asyncio
    async def test_training_uses_preprocessing_split_artifacts(self) -> None:
        """Test that training reuses the preprocessing split when available."""
        df = pd.DataFrame({
            "feature1": np.linspace(1, 100, 120),
            "feature2": np.linspace(50, 150, 120),
            "target": [0, 1] * 60,
        })
        analysis = {
            "numeric_columns": ["feature1", "feature2"],
            "categorical_columns": [],
        }

        preprocessor = PreprocessorAgent()
        preprocessing_result = await preprocessor.execute(df, analysis, "target", test_size=0.25, random_state=11)

        model_selection = {
            "top_candidates": [
                {
                    "priority": 1,
                    "model_name": "RandomForest",
                    "model_family": "tree_ensemble",
                    "reasoning": "Primary recommendation.",
                    "fixed_params": {"n_estimators": 10, "random_state": 42},
                    "search_space": {},
                }
            ],
            "task_type": "classification",
            "target_column": "target",
        }

        agent = TrainingAgent()
        result = await agent.execute(
            df,
            model_selection,
            {
                "test_size": 0.25,
                "random_state": 11,
                "preprocessing_result": preprocessing_result,
            },
        )

        assert result["X_train"].equals(preprocessing_result["_X_train_transformed"])
        assert result["X_test"].equals(preprocessing_result["_X_test_transformed"])
        assert result["X_train"].shape[0] == preprocessing_result["train_size"]
        assert result["X_test"].shape[0] == preprocessing_result["test_size"]

    @pytest.mark.asyncio
    async def test_training_multi_model_uses_top_candidates(self) -> None:
        """Test that multi-model training compares candidates from model selection output."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })

        model_selection = {
            "top_candidates": [
                {
                    "priority": 1,
                    "model_name": "RandomForest",
                    "model_family": "tree_ensemble",
                    "reasoning": "Primary recommendation.",
                    "fixed_params": {"n_estimators": 10, "random_state": 42},
                    "search_space": {},
                },
                {
                    "priority": 2,
                    "model_name": "LogisticRegression",
                    "model_family": "linear",
                    "reasoning": "Fast backup baseline.",
                    "fixed_params": {"max_iter": 500, "C": 1.0, "random_state": 42},
                    "search_space": {},
                },
            ],
            "task_type": "classification",
            "target_column": "target",
        }

        agent = TrainingAgent()
        result = await agent.execute(
            df,
            model_selection,
            {
                "test_size": 0.2,
                "random_state": 42,
                "enable_multi_model": True,
                "optimize_hyperparameters": False,
                "cv_folds": 3,
            },
        )

        assert result["training_mode"] == "multi_model"
        compared_names = [item["model_name"] for item in result.get("compared_candidates", [])]
        assert compared_names == ["RandomForest", "LogisticRegression"]


class TestEvaluationAgent:
    """Tests for the EvaluationAgent class."""

    def test_evaluation_initialization(self) -> None:
        """Test EvaluationAgent initialization."""
        agent = EvaluationAgent()
        assert agent.name == "Evaluation"

    @pytest.mark.asyncio
    async def test_evaluation_computes_metrics(self) -> None:
        """Test that agent computes metrics."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })

        X = df[["feature1", "feature2"]]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        training_result = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_score": model.score(X_train, y_train),
        }

        agent = EvaluationAgent()
        result = await agent.execute(training_result, "classification")

        assert "accuracy" in result
        assert "f1" in result
        assert "deployment_decision" in result

    @pytest.mark.asyncio
    async def test_evaluation_computes_regression_metrics(self) -> None:
        """Test that agent computes regression metrics."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        df = pd.DataFrame({
            "feature1": np.linspace(1, 100, 100),
            "feature2": np.linspace(50, 150, 100),
        })
        df["target"] = (df["feature1"] * 2.5) + (df["feature2"] * 0.75)

        X = df[["feature1", "feature2"]]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        training_result = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_score": model.score(X_train, y_train),
            "test_score": model.score(X_test, y_test),
        }

        agent = EvaluationAgent()
        result = await agent.execute(training_result, "regression")

        assert result["task_type"] == "regression"
        assert "r2" in result
        assert "rmse" in result
        assert "mae" in result
        assert "deployment_decision" in result


class TestDeploymentAgent:
    """Tests for the DeploymentAgent class."""

    def test_deployment_initialization(self) -> None:
        """Test DeploymentAgent initialization."""
        agent = DeploymentAgent()
        assert agent.name == "Deployment"

    @pytest.mark.asyncio
    async def test_deployment_saves_model(self) -> None:
        """Test that agent saves the model and returns model_path."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        training_result = {
            "model": model,
            "model_name": "RandomForest",
        }

        evaluation_result = {
            "task_type": "classification",
            "accuracy": 0.9,
            "f1": 0.85,
            "deployment_decision": "deploy",
        }

        agent = DeploymentAgent()
        result = await agent.execute(training_result, evaluation_result, "test_pipeline")

        assert "model_path" in result
        assert result["deployment_success"] is True

    @pytest.mark.asyncio
    async def test_deployment_builds_package_zip(self) -> None:
        """Test that the deployment package zip is created with required files."""
        import zipfile
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        training_result = {
            "model": model,
            "model_name": "RandomForest",
            "feature_names": ["age", "income", "region_A", "region_B"],
        }
        evaluation_result = {
            "task_type": "classification",
            "accuracy": 0.92,
            "f1": 0.88,
            "deployment_decision": "deploy",
        }
        analysis_result = {
            "numeric_columns": ["age", "income"],
            "categorical_columns": ["region"],
        }
        preprocessing_result = {
            "numeric_columns": ["age", "income"],
            "categorical_columns": ["region"],
            "encoding_mapping": {"region": ["A", "B"]},
        }
        raw_df = pd.DataFrame({
            "age": [25, 35, 45, 55],
            "income": [30000.0, 50000.0, 70000.0, 90000.0],
            "region": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        })

        agent = DeploymentAgent()
        result = await agent.execute(
            training_result=training_result,
            evaluation_result=evaluation_result,
            pipeline_id="test_pkg_pipeline",
            analysis_result=analysis_result,
            preprocessing_result=preprocessing_result,
            raw_dataset=raw_df,
            target_column="target",
        )

        assert result["package_ready"] is True
        assert "package_path" in result

        zip_path = result["package_path"]
        assert Path(zip_path).exists(), "Package zip file should exist"

        required_files = {
            "app.py",
            "schema.json",
            "model.pkl",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            "README.md",
            "report.html",
        }
        with zipfile.ZipFile(zip_path, "r") as zf:
            actual_names = set(zf.namelist())
        assert required_files.issubset(actual_names), f"Missing: {required_files - actual_names}"

    @pytest.mark.asyncio
    async def test_deployment_schema_contains_column_info(self) -> None:
        """Test that schema.json in the package has the expected structure."""
        import json
        import zipfile
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        training_result = {"model": model, "model_name": "RF"}
        evaluation_result = {"task_type": "classification", "accuracy": 0.9, "deployment_decision": "deploy"}
        preprocessing_result = {
            "numeric_columns": ["age", "income"],
            "categorical_columns": ["region"],
            "encoding_mapping": {"region": ["A", "B"]},
        }
        raw_df = pd.DataFrame({
            "age": [20, 30, 40],
            "income": [20000.0, 40000.0, 60000.0],
            "region": ["A", "B", "A"],
            "label": [0, 1, 0],
        })

        agent = DeploymentAgent()
        result = await agent.execute(
            training_result=training_result,
            evaluation_result=evaluation_result,
            pipeline_id="test_schema_pipeline",
            preprocessing_result=preprocessing_result,
            raw_dataset=raw_df,
            target_column="label",
        )

        with zipfile.ZipFile(result["package_path"], "r") as zf:
            schema = json.loads(zf.read("schema.json"))

        assert "required_columns" in schema
        assert "numeric_columns" in schema
        assert "categorical_columns" in schema
        assert "column_types" in schema
        assert "column_ranges" in schema
        assert "allowed_categories" in schema
        assert "train_medians" in schema
        assert "feature_order" in schema

        # Numeric columns must have range entries
        assert "age" in schema["column_ranges"]
        assert "min" in schema["column_ranges"]["age"]
        assert "max" in schema["column_ranges"]["age"]

        # Categorical columns must have allowed_categories
        assert "region" in schema["allowed_categories"]
        assert set(schema["allowed_categories"]["region"]) == {"A", "B"}

    @pytest.mark.asyncio
    async def test_deployment_app_py_contains_validation_logic(self) -> None:
        """Test that the generated app.py contains schema-aware validation."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        training_result = {"model": model, "model_name": "RF"}
        evaluation_result = {"task_type": "classification", "accuracy": 0.9, "deployment_decision": "deploy"}

        agent = DeploymentAgent()
        result = await agent.execute(
            training_result=training_result,
            evaluation_result=evaluation_result,
            pipeline_id="test_apppy_pipeline",
        )

        import zipfile
        with zipfile.ZipFile(result["package_path"], "r") as zf:
            app_source = zf.read("app.py").decode()

        assert "validate_input" in app_source
        assert "validation_errors" in app_source
        assert "/predict" in app_source
        assert "preprocess" in app_source

    @pytest.mark.asyncio
    async def test_deployment_requirements_are_pinned(self) -> None:
        """Test that requirements.txt contains pinned version specifiers."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        agent = DeploymentAgent()
        result = await agent.execute(
            training_result={"model": model, "model_name": "RF"},
            evaluation_result={"task_type": "classification", "accuracy": 0.9, "deployment_decision": "deploy"},
            pipeline_id="test_reqs_pipeline",
        )

        import zipfile
        with zipfile.ZipFile(result["package_path"], "r") as zf:
            reqs = zf.read("requirements.txt").decode()

        # At least some lines should have version pins from runtime
        pinned = [line for line in reqs.splitlines() if "==" in line]
        assert len(pinned) > 0, "At least some packages should have pinned versions"

    @pytest.mark.asyncio
    async def test_deployment_schema_uses_model_feature_names_in(self) -> None:
        """When training_result.feature_names is absent, schema should use model.feature_names_in_."""
        import json
        import zipfile
        import pandas as pd

        class DummyModel:
            feature_names_in_ = np.array(["clarity", "color", "depth__div__carat"])

        train_df = pd.DataFrame({
            "clarity": ["SI2", "VS1", "SI1", "VVS2"],
            "color": ["E", "D", "F", "E"],
            "depth__div__carat": [24.0, 20.0, 18.0, 17.0],
            "target": [0, 1, 0, 1],
        })
        model = DummyModel()

        agent = DeploymentAgent()
        result = await agent.execute(
            training_result={"model": model, "model_name": "RF"},
            evaluation_result={"task_type": "classification", "accuracy": 0.9, "deployment_decision": "deploy"},
            pipeline_id="test_feature_names_in_pipeline",
            preprocessing_result={
                "numeric_columns": ["depth__div__carat"],
                "categorical_columns": ["clarity", "color"],
                "encoding_mapping": {},
            },
            raw_dataset=train_df,
            target_column="target",
        )

        with zipfile.ZipFile(result["package_path"], "r") as zf:
            schema = json.loads(zf.read("schema.json"))

        assert schema["feature_order"] == ["clarity", "color", "depth__div__carat"]

    def test_report_generator_outputs_html(self) -> None:
        """Report generator should return HTML with key panel sections."""
        generator = ReportGenerator()
        assets = generator.generate_assets(
            pipeline_id="abc123",
            dataset_name="demo.csv",
            target_column="target",
            analysis_result={"row_count": 100, "feature_count": 10, "data_quality": {}, "quality_flags": []},
            preprocessing_result={"train_size": 80, "test_size": 20, "transformed_feature_count": 12},
            features_result={"final_feature_count": 12, "selected_features": ["a", "b"]},
            model_selection_result={"top_candidates": [{"model_name": "RandomForest", "model_family": "tree", "reasoning": "robust"}]},
            training_result={"model_name": "RandomForest", "best_score": 0.9, "cv_scores": [0.88, 0.9, 0.89]},
            evaluation_result={"task_type": "classification", "accuracy": 0.92, "f1": 0.9, "confusion_matrix": [[40, 3], [4, 53]]},
            evaluation_insights={"stage_summary": "Great run", "performance_story": "Strong precision/recall"},
            explanation_result={"summary": "Model looks stable"},
        )

        html = assets.get("html")
        assert isinstance(html, str)
        assert "Pipeline Report" in html
        assert "Analysis Panel" in html
        assert "Preprocessing + Feature Engineering Panels" in html
        assert "Model Selection Panel" in html
        assert "Training Panel" in html
        assert "Evaluation Panel" in html

    def test_report_generator_hides_blank_training_chart(self) -> None:
        """Training chart should be omitted when model comparison values are effectively blank."""
        generator = ReportGenerator()
        assets = generator.generate_assets(
            pipeline_id="abc123",
            dataset_name="demo.csv",
            target_column="target",
            analysis_result={},
            preprocessing_result={},
            features_result={},
            model_selection_result={},
            training_result={
                "model_name": "RandomForest",
                "best_score": 0.0,
                "model_comparisons": [
                    {"model_name": "RandomForest", "mean_cv_score": 0.0},
                    {"model_name": "XGBoost", "mean_cv_score": 0.0},
                ],
            },
            evaluation_result={"task_type": "classification"},
            evaluation_insights={},
            explanation_result={},
        )

        html = assets.get("html")
        assert isinstance(html, str)
        assert "Training Panel" in html
        assert 'alt="cv scores"' not in html

