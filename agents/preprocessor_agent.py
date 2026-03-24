"""Preprocessor Agent for AutoML Pipeline.

This agent performs deterministic, AutoML-style preprocessing decisions and
returns UI-friendly metadata about what actually happened.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from agents.base_agent import BaseAgent
from agents.preprocessing_policies import (
    analyze_missingness,
    build_preprocessing_explanation,
    choose_encoding_strategy,
    choose_missing_value_strategy,
    choose_scaler,
    detect_column_types,
    detect_identifier_columns,
    detect_low_variance_columns,
    detect_rare_categories,
    detect_skewed_numeric_columns,
    detect_target_leakage_risks,
    expand_datetime_columns,
    get_default_preprocessing_config,
    infer_task_type,
    summarize_target_distribution,
)
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class PreprocessorAgent(BaseAgent):
    """Agent for deterministic preprocessing before model training."""

    def __init__(self) -> None:
        super().__init__("Preprocessor")

    async def execute(
        self,
        df: pd.DataFrame,
        analysis: dict[str, Any],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        try:
            logger.info("Starting preprocessing of dataset with %s rows", len(df))

            if target_column not in df.columns:
                raise AgentExecutionError(
                    f"Target column '{target_column}' not found in dataset",
                    agent_name=self.name,
                )

            working_df = df.copy()
            y_raw = working_df[target_column].copy()
            X_raw = working_df.drop(columns=[target_column]).copy()
            raw_feature_count = int(X_raw.shape[1])
            config = get_default_preprocessing_config({
                "test_size": test_size,
                "random_state": random_state,
            })
            if config_overrides:
                config.update(config_overrides)
            forced_keep_columns = {
                str(column) for column in config.get("force_keep_columns", [])
                if str(column)
            }
            forced_drop_columns = [
                str(column) for column in config.get("force_drop_columns", [])
                if str(column)
            ]

            initial_types = detect_column_types(X_raw)
            X_expanded, datetime_expansion_map = expand_datetime_columns(
                X_raw,
                initial_types.get("parsed_datetimes", {}),
            )

            leakage_risks = detect_target_leakage_risks(X_expanded, y_raw, target_column)
            identifier_columns = detect_identifier_columns(X_expanded, config=config)
            low_variance_columns = detect_low_variance_columns(X_expanded, config=config)

            missing_decision = choose_missing_value_strategy(
                X_expanded,
                config=config,
                protected_columns=forced_keep_columns,
            )

            drop_records = self._merge_drop_records(
                identifier_columns,
                low_variance_columns,
                leakage_risks.get("dropped_columns", []),
                [
                    {
                        "column": column,
                        "reason": "sparse",
                        "detail": "column had too many missing values to be useful",
                    }
                    for column in missing_decision.get("drop_columns", [])
                ],
            )
            dropped_column_names = [item["column"] for item in drop_records]
            if forced_keep_columns:
                drop_records = [
                    item for item in drop_records
                    if item["column"] not in forced_keep_columns
                ]
            if forced_drop_columns:
                existing = {item["column"] for item in drop_records}
                for column in forced_drop_columns:
                    if column in existing or column not in X_expanded.columns:
                        continue
                    drop_records.append({
                        "column": column,
                        "reason": "manual",
                        "detail": "column was manually dropped by the revision agent",
                    })

            dropped_column_names = [item["column"] for item in drop_records]

            X_reduced = X_expanded.drop(columns=dropped_column_names, errors="ignore")
            rows_dropped_for_target = int(y_raw.isna().sum())
            row_drop_mask = pd.Series(False, index=X_reduced.index)
            if bool(missing_decision.get("drop_rows")) and not X_reduced.empty:
                row_drop_mask = X_reduced.isna().any(axis=1)
            modeling_mask = y_raw.notna() & ~row_drop_mask

            X_model = X_reduced.loc[modeling_mask].copy()
            y_model = y_raw.loc[modeling_mask].copy()
            feature_count_after_column_drops = int(X_model.shape[1])
            retained_feature_columns = list(X_model.columns)
            final_types = detect_column_types(X_model)

            task_type = str(analysis.get("task_type") or infer_task_type(y_model))
            target_distribution = summarize_target_distribution(y_model, task_type)

            train_indices, test_indices = self._split_indices(
                y_model=y_model,
                task_type=task_type,
                test_size=test_size,
                random_state=random_state,
            )
            X_train = X_model.loc[train_indices].copy()
            X_test = X_model.loc[test_indices].copy()
            y_train = y_model.loc[train_indices].copy()
            y_test = y_model.loc[test_indices].copy()

            prep_state = self._fit_preprocessing_state(
                X_train=X_train,
                X_test=X_test,
                full_X=X_model.copy(),
                final_types=final_types,
                config=config,
            )

            transformed_train = prep_state["transformed_train"]
            transformed_test = prep_state["transformed_test"]
            transformed_feature_count = int(transformed_train.shape[1]) if not transformed_train.empty else 0

            missing_before = analyze_missingness(X_expanded)
            missing_after_drops = analyze_missingness(X_reduced)
            imputed_numeric_columns = sorted(prep_state["numeric_impute_values"].keys())
            imputed_categorical_columns = sorted(prep_state["categorical_impute_values"].keys())
            effective_strategy = str(missing_decision.get("strategy_used", "none"))
            if effective_strategy == "drop_columns" and (
                imputed_numeric_columns or imputed_categorical_columns
            ):
                effective_strategy = "mixed"
            if effective_strategy == "drop_rows" and any(
                item["reason"] == "sparse" for item in drop_records
            ):
                effective_strategy = "mixed"

            categorical_summary = self._build_categorical_summary(
                prep_state=prep_state,
                transformed_feature_count=transformed_feature_count,
            )
            scaling_summary = {
                "scaled_columns": prep_state["scaled_numeric_columns"],
                "scaler": prep_state["scaler_name"],
                "reason": prep_state["scaler_reason"],
            }
            transform_summary = {
                "skewed_columns_detected": prep_state["skewed_columns_detected"],
                "log_transformed_columns": prep_state["log_transformed_columns"],
                "outlier_heavy_columns": prep_state["outlier_heavy_columns"],
            }
            target_summary = {
                "target_column": target_column,
                "task_type": task_type,
                "class_distribution": target_distribution.get("class_distribution", {}),
                "imbalance_detected": bool(target_distribution.get("imbalance_detected", False)),
                "imbalance_ratio": target_distribution.get("imbalance_ratio"),
                "imbalance_severity": target_distribution.get("imbalance_severity", "none"),
                "suspicious_feature_columns": leakage_risks.get("suspicious_columns", []),
                "dropped_leakage_columns": [
                    item["column"] for item in drop_records if item["reason"] == "leakage_risk"
                ],
                "rows_removed_for_missing_target": rows_dropped_for_target,
            }

            result: dict[str, Any] = {
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "raw_feature_count": raw_feature_count,
                "feature_count_after_column_drops": feature_count_after_column_drops,
                "transformed_feature_count": transformed_feature_count,
                "feature_count": transformed_feature_count,
                "numeric_columns": final_types["numeric_columns"],
                "categorical_columns": final_types["categorical_columns"],
                "binary_columns": final_types["binary_columns"],
                "datetime_columns": initial_types["datetime_columns"],
                "kept_feature_columns": retained_feature_columns,
                "dropped_columns": drop_records,
                "missing_summary": {
                    "had_missing": missing_before["had_missing"],
                    "total_missing_cells": missing_before["total_missing_cells"],
                    "rows_with_missing": missing_before["rows_with_missing"],
                    "rows_with_missing_fraction": missing_before["rows_with_missing_fraction"],
                    "missing_by_column": missing_before["missing_by_column"],
                    "dropped_rows_count": int(row_drop_mask.sum()),
                    "dropped_columns": [
                        item["column"] for item in drop_records if item["reason"] == "sparse"
                    ],
                    "imputed_numeric_columns": imputed_numeric_columns,
                    "imputed_categorical_columns": imputed_categorical_columns,
                    "strategy_used": effective_strategy,
                    "strategy_reason": str(missing_decision.get("strategy_reason", "")),
                    "remaining_missing_cells_after_column_drops": missing_after_drops["total_missing_cells"],
                },
                "categorical_summary": categorical_summary,
                "scaling_summary": scaling_summary,
                "transform_summary": transform_summary,
                "target_summary": target_summary,
                "datetime_summary": {
                    "expanded_columns": datetime_expansion_map,
                },
                "preprocessing_config": {
                    **config,
                    "strategy_used": effective_strategy,
                    "strategy_reason": str(missing_decision.get("strategy_reason", "")),
                    "dropped_columns": drop_records,
                    "scaling_summary": scaling_summary,
                    "transform_summary": transform_summary,
                    "categorical_summary": categorical_summary,
                },
                "encoding_mapping": categorical_summary.get("categories", {}),
                "missing_handled": bool(missing_before["had_missing"]),
                "llm_used": False,
            }

            explanation_details, llm_used = self._generate_grounded_explanation(result)
            result["explanation"] = explanation_details["summary"]
            result["explanation_details"] = explanation_details
            result["llm_used"] = llm_used

            result["_train_indices"] = list(train_indices)
            result["_test_indices"] = list(test_indices)
            result["_modeling_indices"] = list(X_model.index)
            result["_X_train_transformed"] = transformed_train
            result["_X_test_transformed"] = transformed_test
            result["_y_train"] = y_train
            result["_y_test"] = y_test
            result["_feature_engineering_input_df"] = prep_state["feature_engineering_input_df"].assign(
                **{target_column: y_model}
            )

            logger.info(
                "Preprocessing complete: %s raw features -> %s transformed features",
                raw_feature_count,
                transformed_feature_count,
            )
            return result

        except Exception as exc:
            logger.exception("Error preprocessing data: %s", exc)
            raise AgentExecutionError(
                f"Preprocessing failed: {str(exc)}",
                agent_name=self.name,
                details={"error": str(exc)},
            ) from exc

    def _fit_preprocessing_state(
        self,
        *,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        full_X: pd.DataFrame,
        final_types: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit deterministic preprocessing transforms on training data only."""
        continuous_numeric = [
            column
            for column in final_types.get("numeric_columns", [])
            if column in X_train.columns
        ]
        categorical_columns = [
            column
            for column in final_types.get("categorical_columns", [])
            if column in X_train.columns
        ]
        binary_columns = [
            column
            for column in final_types.get("binary_columns", [])
            if column in X_train.columns
        ]

        numeric_impute_values = self._fit_numeric_imputers(X_train, continuous_numeric)
        categorical_impute_values = self._fit_categorical_imputers(
            X_train,
            categorical_columns + binary_columns,
        )

        train_base = self._apply_base_cleaning(
            X_train.copy(),
            continuous_numeric=continuous_numeric,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            numeric_impute_values=numeric_impute_values,
            categorical_impute_values=categorical_impute_values,
        )
        test_base = self._apply_base_cleaning(
            X_test.copy(),
            continuous_numeric=continuous_numeric,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            numeric_impute_values=numeric_impute_values,
            categorical_impute_values=categorical_impute_values,
        )
        full_base = self._apply_base_cleaning(
            full_X.copy(),
            continuous_numeric=continuous_numeric,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            numeric_impute_values=numeric_impute_values,
            categorical_impute_values=categorical_impute_values,
        )

        rare_category_maps: dict[str, list[str]] = {}
        if bool(config.get("rare_category_grouping", True)):
            for column in categorical_columns:
                if column not in train_base.columns:
                    continue
                rare_values = detect_rare_categories(
                    train_base[column],
                    rare_fraction=float(config["rare_category_fraction"]),
                    min_levels=int(config["rare_category_min_levels"]),
                    max_group_fraction=float(config["rare_category_max_group_fraction"]),
                )
                if rare_values:
                    rare_category_maps[column] = rare_values

        train_base = self._apply_rare_category_grouping(train_base, rare_category_maps)
        test_base = self._apply_rare_category_grouping(test_base, rare_category_maps)
        full_base = self._apply_rare_category_grouping(full_base, rare_category_maps)

        skewed_columns_detected = detect_skewed_numeric_columns(
            train_base,
            continuous_numeric,
            skew_threshold=float(config["skew_threshold"]),
        )
        scaler_decision = choose_scaler(train_base, continuous_numeric, config=config)
        train_base = self._apply_log_transforms(train_base, skewed_columns_detected)
        test_base = self._apply_log_transforms(test_base, skewed_columns_detected)
        full_base = self._apply_log_transforms(full_base, skewed_columns_detected)

        transformed_train, transformed_test, categorical_state = self._encode_features(
            train_df=train_base,
            test_df=test_base,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            continuous_numeric=continuous_numeric,
            config=config,
        )
        transformed_train, transformed_test, scaled_columns = self._scale_numeric_features(
            train_df=transformed_train,
            test_df=transformed_test,
            numeric_columns=continuous_numeric,
            scaler_name=str(scaler_decision["scaler"]),
        )

        return {
            "numeric_impute_values": numeric_impute_values,
            "categorical_impute_values": categorical_impute_values,
            "rare_category_maps": rare_category_maps,
            "log_transformed_columns": skewed_columns_detected,
            "skewed_columns_detected": skewed_columns_detected,
            "outlier_heavy_columns": scaler_decision["outlier_heavy_columns"],
            "scaler_name": scaler_decision["scaler"],
            "scaler_reason": scaler_decision["reason"],
            "scaled_numeric_columns": scaled_columns,
            "categorical_state": categorical_state,
            "transformed_train": transformed_train,
            "transformed_test": transformed_test,
            "feature_engineering_input_df": full_base,
        }

    def _fit_numeric_imputers(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
    ) -> dict[str, float]:
        """Fit median imputers for numeric columns on the training split."""
        values: dict[str, float] = {}
        for column in numeric_columns:
            if column not in df.columns or not df[column].isna().any():
                continue
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            fill_value = float(numeric_series.median()) if numeric_series.notna().any() else 0.0
            values[column] = fill_value
        return values

    def _fit_categorical_imputers(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, str]:
        """Fit most-frequent imputers for categorical columns on the training split."""
        values: dict[str, str] = {}
        for column in columns:
            if column not in df.columns or not df[column].isna().any():
                continue
            mode = df[column].dropna().astype(str).mode()
            values[column] = str(mode.iloc[0]) if not mode.empty else "missing"
        return values

    def _apply_base_cleaning(
        self,
        df: pd.DataFrame,
        *,
        continuous_numeric: list[str],
        categorical_columns: list[str],
        binary_columns: list[str],
        numeric_impute_values: dict[str, float],
        categorical_impute_values: dict[str, str],
    ) -> pd.DataFrame:
        """Apply train-fitted imputers before feature engineering or encoding."""
        cleaned = df.copy()

        for column in continuous_numeric:
            if column not in cleaned.columns:
                continue
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            if column in numeric_impute_values:
                cleaned[column] = cleaned[column].fillna(numeric_impute_values[column])

        for column in categorical_columns + binary_columns:
            if column not in cleaned.columns:
                continue
            if column in categorical_impute_values:
                cleaned[column] = cleaned[column].fillna(categorical_impute_values[column])
            cleaned[column] = cleaned[column].astype(str)

        return cleaned

    def _apply_rare_category_grouping(
        self,
        df: pd.DataFrame,
        rare_category_maps: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Group train-detected rare categories into 'Other'."""
        grouped = df.copy()
        for column, rare_values in rare_category_maps.items():
            if column not in grouped.columns or not rare_values:
                continue
            rare_lookup = set(str(value) for value in rare_values)
            grouped[column] = grouped[column].astype(str).apply(
                lambda value: "Other" if value in rare_lookup else value
            )
        return grouped

    def _apply_log_transforms(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """Apply safe log1p transforms to selected numeric columns."""
        transformed = df.copy()
        for column in columns:
            if column not in transformed.columns:
                continue
            numeric_series = pd.to_numeric(transformed[column], errors="coerce")
            transformed[column] = np.log1p(numeric_series.clip(lower=0.0))
        return transformed

    def _encode_features(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_columns: list[str],
        binary_columns: list[str],
        continuous_numeric: list[str],
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Encode categorical columns deterministically from the training split."""
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        categories: dict[str, list[str]] = {}
        encoding_strategy_by_column = choose_encoding_strategy(
            train_df,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            config=config,
        )
        high_cardinality_columns: list[str] = []
        encoded_columns: list[str] = []

        if continuous_numeric:
            train_parts.append(train_df[continuous_numeric].copy())
            test_parts.append(test_df[continuous_numeric].copy())

        for column in binary_columns:
            if column not in train_df.columns:
                continue
            encoded_columns.append(column)
            observed = sorted(train_df[column].astype(str).dropna().unique().tolist())
            categories[column] = observed[:2]
            mapping = {value: float(index) for index, value in enumerate(observed[:2])}
            train_parts.append(
                pd.DataFrame(
                    {column: train_df[column].astype(str).map(mapping).fillna(0.0).astype(float)},
                    index=train_df.index,
                )
            )
            test_parts.append(
                pd.DataFrame(
                    {column: test_df[column].astype(str).map(mapping).fillna(0.0).astype(float)},
                    index=test_df.index,
                )
            )

        for column in categorical_columns:
            if column not in train_df.columns:
                continue
            strategy = encoding_strategy_by_column.get(column, "onehot")
            encoded_columns.append(column)
            if strategy == "frequency":
                high_cardinality_columns.append(column)
                frequencies = (
                    train_df[column].astype(str).value_counts(normalize=True).sort_index()
                )
                categories[column] = list(frequencies.index[: min(10, len(frequencies))])
                output_name = f"{column}__frequency"
                train_parts.append(
                    pd.DataFrame(
                        {output_name: train_df[column].astype(str).map(frequencies).fillna(0.0)},
                        index=train_df.index,
                    )
                )
                test_parts.append(
                    pd.DataFrame(
                        {output_name: test_df[column].astype(str).map(frequencies).fillna(0.0)},
                        index=test_df.index,
                    )
                )
                continue

            levels = sorted(train_df[column].astype(str).dropna().unique().tolist())
            categories[column] = levels
            train_categorical = pd.Categorical(train_df[column].astype(str), categories=levels)
            test_categorical = pd.Categorical(test_df[column].astype(str), categories=levels)
            train_onehot = pd.get_dummies(train_categorical, prefix=column, prefix_sep="__", dtype=float)
            test_onehot = pd.get_dummies(test_categorical, prefix=column, prefix_sep="__", dtype=float)
            train_onehot.index = train_df.index
            test_onehot.index = test_df.index
            train_parts.append(train_onehot)
            test_parts.append(test_onehot)

        transformed_train = (
            pd.concat(train_parts, axis=1)
            if train_parts
            else pd.DataFrame(index=train_df.index)
        )
        transformed_test = (
            pd.concat(test_parts, axis=1)
            if test_parts
            else pd.DataFrame(index=test_df.index)
        )
        transformed_test = transformed_test.reindex(columns=transformed_train.columns, fill_value=0.0)

        return transformed_train, transformed_test, {
            "categories": categories,
            "encoding_strategy_by_column": encoding_strategy_by_column,
            "high_cardinality_columns": sorted(high_cardinality_columns),
            "encoded_columns": sorted(encoded_columns),
        }

    def _scale_numeric_features(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numeric_columns: list[str],
        scaler_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Scale continuous numeric columns using train-fitted statistics only."""
        if scaler_name == "None" or not numeric_columns:
            return train_df, test_df, []

        scaled_columns = [column for column in numeric_columns if column in train_df.columns]
        if not scaled_columns:
            return train_df, test_df, []

        scaler = RobustScaler() if scaler_name == "RobustScaler" else StandardScaler()
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        train_scaled[scaled_columns] = scaler.fit_transform(train_scaled[scaled_columns])
        test_scaled[scaled_columns] = scaler.transform(test_scaled[scaled_columns])
        return train_scaled, test_scaled, scaled_columns

    def _build_categorical_summary(
        self,
        *,
        prep_state: dict[str, Any],
        transformed_feature_count: int,
    ) -> dict[str, Any]:
        """Build a structured categorical summary for the UI."""
        categorical_state = prep_state["categorical_state"]
        categories = categorical_state["categories"]
        strategies = categorical_state["encoding_strategy_by_column"]
        encoded_columns = categorical_state["encoded_columns"]
        output_columns_created = 0
        for column in encoded_columns:
            strategy = strategies.get(column, "")
            if strategy == "onehot":
                output_columns_created += max(len(categories.get(column, [])) - 1, 0)

        return {
            "encoded_columns": encoded_columns,
            "high_cardinality_columns": categorical_state["high_cardinality_columns"],
            "rare_category_grouped_columns": sorted(prep_state["rare_category_maps"].keys()),
            "encoding_strategy_by_column": strategies,
            "categories": categories,
            "output_columns_created": output_columns_created,
            "binary_encoded_columns": [
                column for column, strategy in strategies.items() if strategy == "binary"
            ],
            "transformed_feature_count": transformed_feature_count,
        }

    def _merge_drop_records(self, *groups: list[dict[str, str]]) -> list[dict[str, str]]:
        """Merge drop reasons while keeping the first reason for each column."""
        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                column = str(item.get("column", ""))
                if not column or column in seen:
                    continue
                merged.append({
                    "column": column,
                    "reason": str(item.get("reason", "other")),
                    "detail": str(item.get("detail", "")),
                })
                seen.add(column)
        return merged

    def _split_indices(
        self,
        *,
        y_model: pd.Series,
        task_type: str,
        test_size: float,
        random_state: int,
    ) -> tuple[list[Any], list[Any]]:
        """Create a leakage-safe train/test split with optional stratification."""
        indices = list(y_model.index)
        stratify_target: Optional[pd.Series] = None
        if task_type == "classification":
            class_counts = y_model.value_counts()
            if not class_counts.empty and int(class_counts.min()) >= 2:
                stratify_target = y_model

        try:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_target,
            )
        except ValueError:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
        return list(train_indices), list(test_indices)

    def _generate_grounded_explanation(
        self,
        result: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Use structured decisions as the source of truth for explanations."""
        fallback = build_preprocessing_explanation(result)
        payload = {
            "summary": fallback["summary"],
            "decisions": fallback["decisions"],
            "why": fallback["why"],
            "missing_summary": result.get("missing_summary", {}),
            "categorical_summary": result.get("categorical_summary", {}),
            "scaling_summary": result.get("scaling_summary", {}),
            "transform_summary": result.get("transform_summary", {}),
            "dropped_columns": result.get("dropped_columns", []),
            "target_summary": result.get("target_summary", {}),
        }
        response = self._generate_llm_json(
            system_prompt=(
                "You explain preprocessing decisions to a beginner. "
                "Use ONLY the structured facts you are given. "
                "Do not invent steps, counts, or strategies. "
                "Return JSON only with keys summary, decisions, and why. "
                "summary must be 2-3 sentences. decisions must be a short list of factual actions. "
                "why must explain the overall benefit in plain language."
            ),
            user_prompt=f"Grounded preprocessing facts:\n{self._safe_json(payload)}",
            temperature=0.1,
            max_tokens=500,
        )
        if not response:
            return fallback, False

        summary = str(response.get("summary") or "").strip()
        decisions = response.get("decisions", [])
        why = str(response.get("why") or "").strip()
        if not summary:
            return fallback, False

        cleaned_decisions = [
            str(item).strip() for item in decisions
            if str(item).strip()
        ]
        return {
            "summary": summary,
            "decisions": cleaned_decisions[:6] if cleaned_decisions else fallback["decisions"],
            "why": why or fallback["why"],
        }, True

    def _build_agent_summary(  # type: ignore[override]
        self,
        result: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dataset_summary: Optional[str],
    ) -> dict[str, Any]:
        details = result.get("explanation_details", {}) if isinstance(result, dict) else {}
        summary = str(details.get("summary") or result.get("explanation") or "Preprocessing completed.").strip()
        decisions = details.get("decisions") if isinstance(details, dict) else []
        why = str(details.get("why") or "These steps clean the data and make it ready for training.").strip()

        return {
            "agent": self.name,
            "step_summary": summary,
            "decisions_made": [str(item).strip() for item in decisions][:4] if isinstance(decisions, list) else [],
            "why": why,
            "overall_summary": summary,
            "llm_used": bool(result.get("llm_used", False)),
        }
