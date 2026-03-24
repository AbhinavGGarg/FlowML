"""Tests for the Orchestrator module.

This module contains tests for the Orchestrator class which
coordinates the multi-agent AutoML pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from core.orchestrator import Orchestrator


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    })


@pytest.mark.asyncio
async def test_orchestrator_pipeline_completes(sample_dataframe) -> None:
    """Test that the orchestrator completes the full pipeline."""
    orchestrator = Orchestrator()
    result = await orchestrator.run_pipeline(
        sample_dataframe,
        target_column="target",
        task_type="classification",
    )

    assert "analysis" in result
    assert "preprocessing" in result
    assert "features" in result
    assert "training" in result
    assert "evaluation" in result
    assert "deployment" in result


@pytest.mark.asyncio
async def test_orchestrator_analysis_stage(sample_dataframe) -> None:
    """Test that analysis stage produces expected results."""
    orchestrator = Orchestrator()
    result = await orchestrator.run_pipeline(
        sample_dataframe,
        target_column="target",
        task_type="classification",
    )

    analysis = result.get("analysis", {})
    assert analysis.get("row_count") == 100
    assert analysis.get("feature_count") == 3
    assert len(analysis.get("numeric_columns", [])) >= 2


@pytest.mark.asyncio
async def test_orchestrator_training_stage(sample_dataframe) -> None:
    """Test that training stage produces model and scores."""
    orchestrator = Orchestrator()
    result = await orchestrator.run_pipeline(
        sample_dataframe,
        target_column="target",
        task_type="classification",
    )

    training = result.get("training", {})
    assert "model" in training
    assert "best_score" in training
    assert "cv_scores" in training


@pytest.mark.asyncio
async def test_orchestrator_evaluation_stage(sample_dataframe) -> None:
    """Test that evaluation stage produces metrics."""
    orchestrator = Orchestrator()
    result = await orchestrator.run_pipeline(
        sample_dataframe,
        target_column="target",
        task_type="classification",
    )

    evaluation = result.get("evaluation", {})
    assert "accuracy" in evaluation
    assert "f1" in evaluation
    assert "confusion_matrix" in evaluation
    assert "deployment_decision" in evaluation


@pytest.mark.asyncio
async def test_orchestrator_deployment_stage(sample_dataframe) -> None:
    """Test that deployment stage saves model."""
    orchestrator = Orchestrator()
    result = await orchestrator.run_pipeline(
        sample_dataframe,
        target_column="target",
        task_type="classification",
    )

    deployment = result.get("deployment", {})
    assert "model_path" in deployment
    assert deployment.get("deployment_success") is True


def test_orchestrator_pipeline_status(sample_dataframe) -> None:
    """Test that pipeline status is tracked correctly."""
    orchestrator = Orchestrator()

    import asyncio
    asyncio.run(orchestrator.run_pipeline(
        sample_dataframe,
        target_column="target",
    ))

    status = orchestrator.get_pipeline_status()
    assert "analysis" in status
    assert "preprocessing" in status
    assert "training" in status
    assert status["analysis"] == "completed"


def test_orchestrator_initialization() -> None:
    """Test orchestrator can be initialized without running."""
    orchestrator = Orchestrator()

    assert orchestrator.memory is not None
    assert orchestrator.stage_results == {}
    assert orchestrator.stage_statuses == {}