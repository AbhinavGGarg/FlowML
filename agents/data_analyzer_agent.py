"""Data Analyzer Agent for AutoML Pipeline.

This agent analyzes the uploaded dataset to provide insights about
data quality, feature types, correlations, and recommends preprocessing strategies.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class DataAnalyzerAgent(BaseAgent):
    """Agent for analyzing datasets and providing preprocessing recommendations.

    This agent analyzes:
    - Row and column counts
    - Column data types (numeric, categorical, text)
    - Missing value patterns
    - Class distribution (for classification tasks)
    - Feature correlations
    - Outliers in numeric columns
    """

    def __init__(self) -> None:
        """Initialize the DataAnalyzerAgent."""
        super().__init__("DataAnalyzer")

    async def execute(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> dict[str, Any]:
        """Analyze the dataset and return comprehensive analysis results.

        Args:
            df: The input DataFrame to analyze.
            target_column: Optional target column name for classification/regression.

        Returns:
            Dictionary containing analysis results including:
            - row_count, column_count, feature_count
            - column_types: dict mapping column names to types
            - missing_values: dict with missing value percentages
            - numeric_columns, categorical_columns
            - correlations: correlation matrix for numeric features
            - class_distribution (if target provided)
            - recommendations: list of preprocessing recommendations
        """
        try:
            # logger.info(f"Analyzing dataset with {len(df)} rows and {len(df.columns)} columns")

            # Basic stats
            row_count = len(df)
            column_count = len(df.columns)
            feature_count = column_count - (1 if target_column else 0)

            # Classify columns by type
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Handle target column classification
            if target_column:
                if target_column in numeric_columns:
                    numeric_columns.remove(target_column)
                if target_column in categorical_columns:
                    categorical_columns.remove(target_column)

            # Missing value analysis
            missing_values = df.isnull().mean().to_dict()
            high_missing_cols = [col for col, pct in missing_values.items() if pct > 0.3]

            # Correlation analysis for numeric columns
            correlations = None
            if len(numeric_columns) > 1:
                corr_matrix = df[numeric_columns].corr()
                correlations = corr_matrix.to_dict()

                # Find highly correlated pairs
                high_corr_pairs = []
                for i, col1 in enumerate(numeric_columns):
                    for j, col2 in enumerate(numeric_columns):
                        if i < j and abs(corr_matrix.loc[col1, col2]) > 0.8:
                            high_corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
            else:
                high_corr_pairs = []

            # Class distribution for target column
            class_distribution = None
            if target_column:
                target_series = df[target_column]
                if target_series.dtype in ["object", "category"] or not np.issubdtype(
                    target_series.dtype, np.number
                ):
                    # Classification task
                    class_distribution = target_series.value_counts().to_dict()
                else:
                    # Regression task
                    class_distribution = {
                        "min": float(target_series.min()),
                        "max": float(target_series.max()),
                        "mean": float(target_series.mean()),
                        "std": float(target_series.std()),
                        "median": float(target_series.median()),
                    }

            # Outlier analysis for numeric columns
            outliers = {}
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outlier_count > 0:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(df) * 100),
                    }

            # Generate recommendations
            recommendations = self._generate_recommendations(
                missing_values=missing_values,
                high_missing_cols=high_missing_cols,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                high_corr_pairs=high_corr_pairs,
                outliers=outliers,
                class_distribution=class_distribution,
                target_column=target_column,
            )

            llm_insights = self._generate_llm_analysis_insights(
                row_count=row_count,
                column_count=column_count,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                high_missing_cols=high_missing_cols,
                high_corr_pairs=high_corr_pairs,
                outliers=outliers,
                class_distribution=class_distribution,
                target_column=target_column,
                deterministic_recommendations=recommendations,
            )

            if llm_insights.get("recommendations"):
                recommendations = self._merge_recommendations(
                    recommendations,
                    llm_insights["recommendations"],
                )

            # Data quality profile
            data_quality = self._build_data_quality_profile(
                df=df,
                missing_values=missing_values,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                outliers=outliers,
                target_column=target_column,
            )
            quality_flags = self._build_quality_flags(data_quality)

            result = {
                "row_count": row_count,
                "column_count": column_count,
                "feature_count": feature_count,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "missing_values": missing_values,
                "high_missing_columns": high_missing_cols,
                "correlations": correlations,
                "high_correlation_pairs": high_corr_pairs,
                "outliers": outliers,
                "class_distribution": class_distribution,
                "target_column": target_column,
                "recommendations": recommendations,
                "analysis_summary": llm_insights.get("analysis_summary"),
                "risk_level": llm_insights.get("risk_level"),
                "llm_used": llm_insights.get("llm_used", False),
                "data_quality": data_quality,
                "quality_flags": quality_flags,
            }

            # logger.info(f"Analysis complete: {feature_count} features, {len(high_missing_cols)} cols with high missing")
            return result

        except Exception as e:
            # logger.exception(f"Error analyzing dataset: {e}")
            raise AgentExecutionError(
                f"Data analysis failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _generate_recommendations(
        self,
        missing_values: dict[str, float],
        high_missing_cols: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict,
        class_distribution: Optional[dict],
        target_column: Optional[str],
    ) -> list[str]:
        """Generate preprocessing recommendations based on analysis."""
        recommendations = []

        # Missing value recommendations
        if high_missing_cols:
            recommendations.append(
                f"High missing values detected in {len(high_missing_cols)} columns. "
                "Consider dropping or imputing."
            )

        # Correlation recommendations
        if high_corr_pairs:
            recommendations.append(
                f"High correlation detected between {len(high_corr_pairs)} feature pairs. "
                "Consider removing redundant features."
            )

        # Outlier recommendations
        if outliers:
            recommendations.append(
                f"Outliers detected in {len(outliers)} columns. Consider winsorization or removal."
            )

        # Class imbalance
        if class_distribution and isinstance(class_distribution, dict):
            if all(isinstance(v, int) for v in class_distribution.values()):
                total = sum(class_distribution.values())
                max_pct = max(class_distribution.values()) / total
                min_pct = min(class_distribution.values()) / total
                if max_pct / min_pct > 10 if min_pct > 0 else False:
                    recommendations.append(
                        "Significant class imbalance detected. Consider SMOTE or class weights."
                    )

        # Default recommendations
        if not recommendations:
            recommendations.append("Data quality looks good. Proceed with standard preprocessing.")

        return recommendations

    def _generate_llm_analysis_insights(
        self,
        *,
        row_count: int,
        column_count: int,
        numeric_columns: list[str],
        categorical_columns: list[str],
        high_missing_cols: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict[str, Any],
        class_distribution: Optional[dict[str, Any]],
        target_column: Optional[str],
        deterministic_recommendations: list[str],
    ) -> dict[str, Any]:
        """Use an LLM to synthesize a concise analysis summary and recommendations."""
        payload = {
            "row_count": row_count,
            "column_count": column_count,
            "target_column": target_column,
            "numeric_columns": numeric_columns[:15],
            "categorical_columns": categorical_columns[:15],
            "high_missing_columns": high_missing_cols[:15],
            "high_correlation_pairs": [list(pair) for pair in high_corr_pairs[:10]],
            "outliers": outliers,
            "class_distribution": class_distribution,
            "deterministic_recommendations": deterministic_recommendations,
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an AutoML data analysis assistant. "
                "Given structured dataset diagnostics, produce a concise risk summary and "
                "up to three practical preprocessing recommendations. "
                "Return ONLY valid JSON with keys "
                "'analysis_summary', 'risk_level', and 'recommendations'. "
                "risk_level must be one of: low, medium, high. "
                "Recommendations must be short, concrete, and implementation-oriented."
            ),
            user_prompt=f"Dataset diagnostics:\n{self._safe_json(payload)}",
            temperature=0.1,
            max_tokens=500,
        )

        if not response:
            return {
                "analysis_summary": self._build_default_summary(
                    row_count,
                    column_count,
                    high_missing_cols,
                    high_corr_pairs,
                    outliers,
                ),
                "risk_level": self._default_risk_level(high_missing_cols, high_corr_pairs, outliers),
                "recommendations": [],
                "llm_used": False,
            }

        recommendations = response.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = []

        return {
            "analysis_summary": str(response.get("analysis_summary") or ""),
            "risk_level": str(response.get("risk_level") or self._default_risk_level(high_missing_cols, high_corr_pairs, outliers)).lower(),
            "recommendations": [str(item) for item in recommendations[:3] if str(item).strip()],
            "llm_used": True,
        }

    def _merge_recommendations(self, base: list[str], extra: list[str]) -> list[str]:
        """Merge deterministic and LLM recommendations while preserving order."""
        merged: list[str] = []
        for item in [*base, *extra]:
            normalized = item.strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    def _default_risk_level(
        self,
        high_missing_cols: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict[str, Any],
    ) -> str:
        """Generate a fallback risk level without an LLM."""
        issue_count = int(bool(high_missing_cols)) + int(bool(high_corr_pairs)) + int(bool(outliers))
        if issue_count >= 3:
            return "high"
        if issue_count >= 1:
            return "medium"
        return "low"

    def _build_default_summary(
        self,
        row_count: int,
        column_count: int,
        high_missing_cols: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict[str, Any],
    ) -> str:
        """Generate a concise deterministic fallback summary."""
        parts = [f"Dataset contains {row_count} rows and {column_count} columns."]
        if high_missing_cols:
            parts.append(f"{len(high_missing_cols)} columns have high missingness.")
        if high_corr_pairs:
            parts.append(f"{len(high_corr_pairs)} strongly correlated feature pairs were detected.")
        if outliers:
            parts.append(f"Outliers were found in {len(outliers)} numeric columns.")
        if len(parts) == 1:
            parts.append("Overall data quality looks stable for standard preprocessing.")
        return " ".join(parts)

    def _build_data_quality_profile(
        self,
        *,
        df: pd.DataFrame,
        missing_values: dict[str, float],
        numeric_columns: list[str],
        categorical_columns: list[str],
        outliers: dict[str, Any],
        target_column: Optional[str],
    ) -> dict[str, Any]:
        """Compute a structured data-quality profile for the dataset."""
        total_missing_cells = int(df.isnull().sum().sum())
        missing_rows = int(df.isnull().any(axis=1).sum())
        missing_rows_pct = round(missing_rows / max(len(df), 1) * 100, 2)
        missing_columns_count = sum(1 for v in missing_values.values() if v > 0)
        high_missing_columns = [col for col, pct in missing_values.items() if pct > 0.3]

        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / max(len(df), 1) * 100, 2)

        outlier_columns_count = len(outliers)
        max_outlier_pct = round(
            max((v["percentage"] for v in outliers.values()), default=0.0), 2
        )

        high_card_threshold = max(50, int(len(df) * 0.5))
        high_cardinality_columns = [
            col
            for col in categorical_columns
            if df[col].nunique(dropna=True) > high_card_threshold
        ]
        high_cardinality_count = len(high_cardinality_columns)

        placeholder_cols = self._detect_placeholder_invalids(df, numeric_columns, categorical_columns)
        placeholder_invalid_count = len(placeholder_cols)

        leakage_risk_columns = self._detect_leakage_signals(df, target_column, numeric_columns)

        return {
            "total_missing_cells": total_missing_cells,
            "missing_rows_pct": missing_rows_pct,
            "missing_columns_count": missing_columns_count,
            "high_missing_columns": high_missing_columns,
            "duplicate_rows": dup_count,
            "duplicate_pct": dup_pct,
            "outlier_columns_count": outlier_columns_count,
            "max_outlier_pct": max_outlier_pct,
            "high_cardinality_columns": high_cardinality_columns,
            "high_cardinality_count": high_cardinality_count,
            "placeholder_invalid_count": placeholder_invalid_count,
            "placeholder_invalid_columns": placeholder_cols,
            "leakage_risk_columns": leakage_risk_columns,
        }

    def _detect_placeholder_invalids(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> list[str]:
        """Identify columns with suspicious sentinel / placeholder values."""
        NUMERIC_SENTINELS = {-1, -999, -9999, -99, 9999, 99999, 0}
        TEXT_SENTINELS = {"n/a", "na", "none", "null", "missing", "unknown", "-", "", "?", "nan"}
        flagged: list[str] = []

        for col in numeric_columns:
            series = df[col].dropna()
            if series.empty:
                continue
            for sentinel in NUMERIC_SENTINELS:
                if (series == sentinel).mean() > 0.05:
                    flagged.append(col)
                    break

        for col in categorical_columns:
            series = df[col].dropna().astype(str).str.strip().str.lower()
            if series.empty:
                continue
            if series.isin(TEXT_SENTINELS).mean() > 0.05:
                flagged.append(col)

        return list(dict.fromkeys(flagged))

    def _detect_leakage_signals(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        numeric_columns: list[str],
    ) -> list[str]:
        """Detect columns with name similarity or near-perfect correlation with the target."""
        if not target_column or target_column not in df.columns:
            return []

        risk_cols: list[str] = []
        normalized_target = target_column.lower().replace(" ", "_")

        for col in df.columns:
            if col == target_column:
                continue
            normalized = col.lower().replace(" ", "_")
            if normalized_target in normalized or normalized in normalized_target:
                risk_cols.append(col)

        if pd.api.types.is_numeric_dtype(df[target_column]):
            for col in numeric_columns:
                if col == target_column or col in risk_cols:
                    continue
                try:
                    corr = abs(float(df[col].corr(df[target_column])))
                    if corr > 0.995:
                        risk_cols.append(col)
                except Exception:
                    pass

        return list(dict.fromkeys(risk_cols))

    def _build_quality_flags(self, data_quality: dict[str, Any]) -> list[dict[str, str]]:
        """Produce severity-tagged quality flags from a data quality profile."""
        flags: list[dict[str, str]] = []

        dup_pct = float(data_quality.get("duplicate_pct") or 0)
        if dup_pct > 5:
            flags.append({
                "severity": "high" if dup_pct > 15 else "medium",
                "message": f"{dup_pct:.1f}% of rows are duplicates — should be removed before training.",
                "field": "duplicate_rows",
            })

        missing_rows_pct = float(data_quality.get("missing_rows_pct") or 0)
        if missing_rows_pct > 20:
            flags.append({
                "severity": "high" if missing_rows_pct > 50 else "medium",
                "message": f"{missing_rows_pct:.1f}% of rows have at least one missing value.",
                "field": "missing_rows_pct",
            })

        outlier_cols = int(data_quality.get("outlier_columns_count") or 0)
        max_outlier_pct = float(data_quality.get("max_outlier_pct") or 0)
        if outlier_cols >= 3 or max_outlier_pct > 10:
            flags.append({
                "severity": "medium",
                "message": (
                    f"Outliers in {outlier_cols} column(s); "
                    f"worst column has {max_outlier_pct:.1f}% outlier rows."
                ),
                "field": "outliers",
            })

        high_card = int(data_quality.get("high_cardinality_count") or 0)
        if high_card > 0:
            cols_preview = ", ".join(
                (data_quality.get("high_cardinality_columns") or [])[:3]
            )
            flags.append({
                "severity": "low",
                "message": (
                    f"{high_card} high-cardinality column(s) ({cols_preview}) "
                    "may bloat one-hot encoding."
                ),
                "field": "high_cardinality",
            })

        placeholder_ct = int(data_quality.get("placeholder_invalid_count") or 0)
        if placeholder_ct > 0:
            cols_preview = ", ".join(
                (data_quality.get("placeholder_invalid_columns") or [])[:3]
            )
            flags.append({
                "severity": "medium",
                "message": (
                    f"Suspected placeholder values (e.g. -999, 'N/A') "
                    f"in {placeholder_ct} column(s): {cols_preview}."
                ),
                "field": "placeholder_invalids",
            })

        leakage_cols = list(data_quality.get("leakage_risk_columns") or [])
        if leakage_cols:
            flags.append({
                "severity": "high",
                "message": (
                    f"Possible target leakage in {len(leakage_cols)} column(s): "
                    f"{', '.join(leakage_cols[:3])}."
                ),
                "field": "leakage_risk",
            })

        return flags
