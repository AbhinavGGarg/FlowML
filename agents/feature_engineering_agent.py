"""Feature Engineering Agent for AutoML Pipeline.

This agent performs deterministic feature engineering before model training.
"""

import logging
import re
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseAgent):
    """Agent for deterministic feature engineering and feature selection."""

    INDEX_LIKE_NAMES = {"index", "row_id"}
    CORRELATION_THRESHOLD = 0.95
    IMPORTANCE_THRESHOLD = 0.01
    MAX_INTERACTION_NUMERIC_COLUMNS = 5
    RANDOM_STATE = 42
    IMPORTANCE_TREES = 50

    def __init__(self) -> None:
        """Initialize the FeatureEngineeringAgent."""
        super().__init__("FeatureEngineering")

    async def execute(
        self,
        df: pd.DataFrame,
        preprocessing: dict[str, Any],
        target_column: str,
        n_features_to_select: int = 0,
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Engineer features for the dataset."""
        try:
            logger.info("Engineering features for dataset with %s columns", len(df.columns))
            config_overrides = config_overrides or {}
            raw_correlation_threshold = config_overrides.get("correlation_threshold")
            correlation_threshold = float(
                self.CORRELATION_THRESHOLD
                if raw_correlation_threshold is None
                else raw_correlation_threshold
            )
            raw_importance_threshold = config_overrides.get("importance_threshold")
            importance_threshold = float(
                self.IMPORTANCE_THRESHOLD
                if raw_importance_threshold is None
                else raw_importance_threshold
            )
            raw_max_interactions = config_overrides.get("max_interaction_numeric_columns")
            max_interaction_numeric_columns = int(
                self.MAX_INTERACTION_NUMERIC_COLUMNS
                if raw_max_interactions is None
                else raw_max_interactions
            )
            include_features = [
                str(feature) for feature in config_overrides.get("include_features", [])
            ]
            exclude_features = {
                str(feature) for feature in config_overrides.get("exclude_features", [])
            }
            force_keep_engineered = {
                str(feature) for feature in config_overrides.get("force_keep_engineered_features", [])
            }
            force_drop_engineered = {
                str(feature) for feature in config_overrides.get("force_drop_engineered_features", [])
            }
            raw_use_interactions = config_overrides.get("use_interactions")
            use_interactions = True if raw_use_interactions is None else bool(raw_use_interactions)
            raw_use_pca = config_overrides.get("use_pca")
            use_pca = use_pca if raw_use_pca is None else bool(raw_use_pca)
            if "n_pca_components" in config_overrides and config_overrides.get("n_pca_components") is not None:
                n_pca_components = config_overrides.get("n_pca_components")

            prepared_df = preprocessing.get("_feature_engineering_input_df")
            if isinstance(prepared_df, pd.DataFrame):
                df = prepared_df.copy()
            else:
                df = df.copy()
                modeling_indices = preprocessing.get("_modeling_indices", [])
                if isinstance(modeling_indices, list) and modeling_indices:
                    available_indices = [index for index in modeling_indices if index in df.index]
                    if available_indices:
                        df = df.loc[available_indices].copy()

                kept_feature_columns = preprocessing.get("kept_feature_columns", [])
                if isinstance(kept_feature_columns, list) and kept_feature_columns:
                    allowed_columns = [column for column in kept_feature_columns if column in df.columns]
                    if target_column in df.columns:
                        df = df[allowed_columns + [target_column]].copy()
                    else:
                        df = df[allowed_columns].copy()

            preprocessing_numeric = preprocessing.get("numeric_columns", [])
            preprocessing_categorical = preprocessing.get("categorical_columns", [])

            if target_column in df.columns:
                y = df[target_column].copy()
                X = df.drop(columns=[target_column]).copy()
            else:
                y = None
                X = df.copy()

            dropped_columns: list[str] = []
            generated_features: list[str] = []
            applied_transformations: list[dict[str, Any]] = []
            pca_result: dict[str, Any] = {}

            X = self._drop_target_column_if_present(X, target_column)

            index_like_columns = [column for column in X.columns if self._is_index_like(column)]
            if index_like_columns:
                X = X.drop(columns=index_like_columns)
                dropped_columns.extend(index_like_columns)
                applied_transformations.append({
                    "type": "drop_index_like_columns",
                    "columns": sorted(index_like_columns),
                })

            numeric_columns, categorical_columns = self._get_feature_types(X)

            duplicate_like_columns = self._find_duplicate_like_columns(
                X,
                numeric_columns,
                correlation_threshold=correlation_threshold,
            )
            if duplicate_like_columns:
                X = X.drop(columns=duplicate_like_columns)
                dropped_columns.extend(duplicate_like_columns)
                applied_transformations.append({
                    "type": "correlation_filter",
                    "threshold": correlation_threshold,
                    "columns": duplicate_like_columns,
                })

            numeric_columns, categorical_columns = self._get_feature_types(X)

            numeric_imputations: dict[str, float] = {}
            for column in numeric_columns:
                if X[column].isna().any():
                    fill_value = float(X[column].median()) if X[column].notna().any() else 0.0
                    X[column] = X[column].fillna(fill_value)
                    numeric_imputations[column] = fill_value
            if numeric_imputations:
                applied_transformations.append({
                    "type": "numeric_imputation",
                    "strategy": "median",
                    "fill_values": numeric_imputations,
                })

            categorical_imputations: dict[str, str] = {}
            for column in categorical_columns:
                if X[column].isna().any():
                    mode = X[column].mode(dropna=True)
                    fill_value = str(mode.iloc[0]) if not mode.empty else "missing"
                    X[column] = X[column].fillna(fill_value)
                    categorical_imputations[column] = fill_value
            if categorical_imputations:
                applied_transformations.append({
                    "type": "categorical_imputation",
                    "strategy": "most_frequent",
                    "fill_values": categorical_imputations,
                })

            log_transforms: list[dict[str, Any]] = []
            for column in numeric_columns:
                skewness = X[column].skew()
                if pd.notna(skewness) and float(skewness) > 1.0:
                    minimum = float(X[column].min())
                    shift = abs(minimum) + 1.0 if minimum <= -1.0 else 0.0
                    X[column] = np.log1p(X[column] + shift)
                    log_transforms.append({
                        "column": column,
                        "skewness": float(skewness),
                        "shift": shift,
                    })
            if log_transforms:
                applied_transformations.append({
                    "type": "log1p_transform",
                    "columns": log_transforms,
                })

            numeric_columns, categorical_columns = self._get_feature_types(X)

            interaction_source_columns = (
                self._select_interaction_columns(
                    X,
                    numeric_columns,
                    max_interaction_numeric_columns=max_interaction_numeric_columns,
                )
                if use_interactions
                else []
            )
            if interaction_source_columns:
                new_features = self._generate_interactions(X, interaction_source_columns)
                if new_features:
                    generated_features.extend(new_features)
                    applied_transformations.append({
                        "type": "generated_interactions",
                        "source_columns": interaction_source_columns,
                        "columns": new_features,
                    })

            correlated_columns = self._find_high_correlation_columns(
                X,
                correlation_threshold=correlation_threshold,
            )
            if correlated_columns:
                X = X.drop(columns=correlated_columns)
                dropped_columns.extend(correlated_columns)
                applied_transformations.append({
                    "type": "correlation_filter",
                    "threshold": correlation_threshold,
                    "columns": correlated_columns,
                })

            if generated_features:
                generated_features = [column for column in generated_features if column in X.columns]

            numeric_columns, categorical_columns = self._get_feature_types(X)

            if y is not None and y.isna().any():
                valid_mask = y.notna()
                X = X.loc[valid_mask].copy()
                y = y.loc[valid_mask].copy()
                applied_transformations.append({
                    "type": "drop_missing_target_rows",
                    "rows_removed": int((~valid_mask).sum()),
                })

            feature_scores = self._compute_feature_importance_scores(X, y)
            feature_scores.pop(target_column, None)

            selected_features = self._select_features_from_scores(
                X=X,
                feature_scores=feature_scores,
                n_features_to_select=n_features_to_select,
                target_column=target_column,
                importance_threshold=importance_threshold,
            )
            for feature_name in include_features:
                if feature_name in X.columns and feature_name not in selected_features:
                    selected_features.append(feature_name)
            if force_keep_engineered:
                for feature_name in sorted(force_keep_engineered):
                    if feature_name in X.columns and feature_name not in selected_features:
                        selected_features.append(feature_name)

            protected_base_features = self._collect_protected_base_features(selected_features, X.columns)
            low_importance_columns = [
                column for column in X.columns
                if column not in selected_features
                and column not in protected_base_features
                and column not in force_keep_engineered
            ]
            if low_importance_columns:
                dropped_columns.extend(low_importance_columns)
                applied_transformations.append({
                    "type": "feature_importance_filter",
                    "threshold": importance_threshold,
                    "columns": sorted(low_importance_columns),
                })

            selected_features = sorted(
                set(selected_features).union(protected_base_features),
                key=lambda column: list(X.columns).index(column),
            )
            if exclude_features or force_drop_engineered:
                blocked = exclude_features.union(force_drop_engineered)
                selected_features = [
                    column for column in selected_features
                    if column not in blocked
                ]
                dropped_columns.extend(sorted(blocked.intersection(set(X.columns))))

            if n_features_to_select > 0 and len(selected_features) > n_features_to_select:
                capped_features = self._apply_selection_cap(
                    selected_features=selected_features,
                    feature_scores=feature_scores,
                    feature_order=list(X.columns),
                    n_features_to_select=n_features_to_select,
                )
                capped_out_columns = [
                    column for column in selected_features
                    if column not in capped_features
                ]
                if capped_out_columns:
                    dropped_columns.extend(capped_out_columns)
                    applied_transformations.append({
                        "type": "selection_cap",
                        "max_features": n_features_to_select,
                        "columns": sorted(capped_out_columns),
                    })
                selected_features = capped_features

            if use_pca and selected_features:
                X, selected_features, generated_features, pca_result = self._apply_pca(
                    X=X,
                    selected_features=selected_features,
                    generated_features=generated_features,
                    n_pca_components=n_pca_components,
                )
                if pca_result:
                    applied_transformations.append({
                        "type": "pca",
                        "n_components": pca_result["n_components"],
                        "components": pca_result["components"],
                    })

            selected_features = [
                column for column in selected_features
                if column in X.columns and column != target_column
            ]
            feature_scores = {
                column: float(score)
                for column, score in self._rank_features(feature_scores)
                if column in X.columns and column != target_column
            }
            generated_features = [
                column for column in generated_features
                if column in X.columns and column != target_column
            ]

            selected_numeric, selected_categorical = self._split_selected_features(
                X,
                selected_features,
                preprocessing_numeric,
                preprocessing_categorical,
            )

            result = {
                "dropped_columns": sorted(dict.fromkeys(dropped_columns)),
                "generated_features": generated_features,
                "selected_features": selected_features,
                "feature_scores": feature_scores,
                "applied_transformations": applied_transformations,
                "final_feature_count": len(selected_features),
                "pca_result": pca_result if pca_result else None,
                "feature_engineering_config": {
                    "n_features_to_select": n_features_to_select,
                    "use_pca": use_pca,
                    "n_pca_components": n_pca_components,
                    "original_feature_count": len(df.columns) - (1 if target_column in df.columns else 0),
                    "post_engineering_feature_count": len(X.columns),
                    "importance_threshold": importance_threshold,
                    "correlation_threshold": correlation_threshold,
                    "max_interaction_numeric_columns": max_interaction_numeric_columns,
                    "include_features": include_features,
                    "exclude_features": sorted(exclude_features),
                    "use_interactions": use_interactions,
                },
                "numeric_features": selected_numeric,
                "categorical_features": selected_categorical,
                "_engineered_df": X,
            }
            result["llm_explanations"] = self._build_feature_llm_explanations(
                result=result,
                target_column=target_column,
                y=y,
            )

            logger.info(
                "Feature engineering complete: %s selected, %s generated, %s dropped",
                len(selected_features),
                len(generated_features),
                len(result["dropped_columns"]),
            )
            return result

        except Exception as e:
            logger.exception("Error in feature engineering: %s", e)
            raise AgentExecutionError(
                f"Feature engineering failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _drop_target_column_if_present(self, X: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Defensively remove the target column from the feature matrix."""
        if target_column in X.columns:
            return X.drop(columns=[target_column])
        return X

    def _is_index_like(self, column_name: str) -> bool:
        """Return True when the column looks like an index or identifier."""
        normalized = column_name.strip().lower()
        return (
            normalized.startswith("unnamed")
            or normalized in self.INDEX_LIKE_NAMES
            or normalized.endswith("_id")
        )

    def _get_feature_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Split features into numeric and categorical groups."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [column for column in X.columns if column not in numeric_columns]
        return numeric_columns, categorical_columns

    def _select_interaction_columns(
        self,
        X: pd.DataFrame,
        numeric_columns: list[str],
        *,
        max_interaction_numeric_columns: int,
    ) -> list[str]:
        """Choose the top numeric columns by variance for interaction generation."""
        if not numeric_columns:
            return []

        variances = X[numeric_columns].var(ddof=0).fillna(0.0)
        ranked = sorted(
            numeric_columns,
            key=lambda column: (-float(variances.get(column, 0.0)), column),
        )
        ranked = [column for column in ranked if float(variances.get(column, 0.0)) > 0.0]
        return ranked[: max_interaction_numeric_columns]

    def _find_duplicate_like_columns(
        self,
        X: pd.DataFrame,
        numeric_columns: list[str],
        *,
        correlation_threshold: float,
    ) -> list[str]:
        """Drop obvious copy-like numeric columns when a base feature already exists."""
        duplicates: list[str] = []
        numeric_frame = X[numeric_columns].copy() if numeric_columns else pd.DataFrame(index=X.index)
        if numeric_frame.empty:
            return duplicates

        for column in numeric_columns:
            normalized = column.lower()
            if not any(token in normalized for token in ("copy", "duplicate", "clone")):
                continue

            base_name = (
                normalized.replace("_copy", "")
                .replace("copy_", "")
                .replace("_duplicate", "")
                .replace("duplicate_", "")
                .replace("_clone", "")
                .replace("clone_", "")
            )
            candidates = [
                other for other in numeric_columns
                if other != column and other.lower() == base_name
            ]
            if not candidates:
                continue

            candidate = candidates[0]
            aligned = numeric_frame[[column, candidate]].dropna()
            if aligned.empty:
                continue
            correlation = aligned[column].corr(aligned[candidate])
            if pd.notna(correlation) and float(abs(correlation)) >= correlation_threshold:
                duplicates.append(column)

        return sorted(dict.fromkeys(duplicates))

    def _generate_interactions(self, X: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
        """Generate deterministic pairwise interactions among selected numeric columns."""
        generated: list[str] = []
        epsilon = 1e-6

        for left, right in combinations(numeric_columns, 2):
            product_name = f"{left}__mul__{right}"
            ratio_name = f"{left}__div__{right}"

            X[product_name] = X[left] * X[right]
            denominator = X[right].where(X[right].abs() > epsilon, np.nan)
            X[ratio_name] = (
                X[left] / denominator
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            generated.extend([product_name, ratio_name])

        spatial_columns = [column for column in ("x", "y", "z") if column in X.columns]
        if len(spatial_columns) == 3:
            volume_name = "x__mul__y__mul__z"
            if volume_name not in X.columns:
                X[volume_name] = X["x"] * X["y"] * X["z"]
                generated.append(volume_name)

        return generated

    def _find_high_correlation_columns(
        self,
        X: pd.DataFrame,
        *,
        correlation_threshold: float,
    ) -> list[str]:
        """Find redundant columns while preferring simpler/original features."""
        numeric_frame = X.select_dtypes(include=[np.number])
        if numeric_frame.empty:
            return []

        correlation_matrix = numeric_frame.corr().abs()
        variances = numeric_frame.var(ddof=0).fillna(0.0).to_dict()
        columns_to_drop: set[str] = set()
        numeric_columns = list(numeric_frame.columns)

        for index, left in enumerate(numeric_columns):
            if left in columns_to_drop:
                continue
            for right in numeric_columns[index + 1:]:
                if right in columns_to_drop:
                    continue
                correlation = correlation_matrix.loc[left, right]
                if pd.isna(correlation) or float(correlation) <= correlation_threshold:
                    continue
                columns_to_drop.add(self._choose_correlated_drop(left, right, variances))

        return sorted(columns_to_drop)

    def _choose_correlated_drop(
        self,
        left: str,
        right: str,
        variances: dict[str, float],
    ) -> str:
        """Pick which correlated feature to drop, preferring simpler/original features."""
        left_is_derived = self._is_derived_feature(left)
        right_is_derived = self._is_derived_feature(right)

        if left_is_derived != right_is_derived:
            return left if left_is_derived else right

        left_copy_like = any(token in left.lower() for token in ("copy", "duplicate", "clone"))
        right_copy_like = any(token in right.lower() for token in ("copy", "duplicate", "clone"))
        if left_copy_like != right_copy_like:
            return left if left_copy_like else right

        left_variance = float(variances.get(left, 0.0))
        right_variance = float(variances.get(right, 0.0))
        if left_variance != right_variance:
            return left if left_variance < right_variance else right

        left_complexity = len(self._extract_base_features(left))
        right_complexity = len(self._extract_base_features(right))
        if left_complexity != right_complexity:
            return left if left_complexity > right_complexity else right

        return max(left, right)

    def _compute_feature_importance_scores(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
    ) -> dict[str, float]:
        """Train a lightweight RandomForest model and return feature importances."""
        if X.empty:
            return {}

        if y is None:
            return {column: 1.0 for column in X.columns}

        encoded_X = self._encode_features_for_model(X)
        encoded_y, is_classification = self._encode_target(y)

        if is_classification:
            model = RandomForestClassifier(
                n_estimators=self.IMPORTANCE_TREES,
                max_depth=8,
                random_state=self.RANDOM_STATE,
                n_jobs=1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.IMPORTANCE_TREES,
                max_depth=8,
                random_state=self.RANDOM_STATE,
                n_jobs=1,
            )

        model.fit(encoded_X, encoded_y)
        return {
            column: float(score)
            for column, score in zip(encoded_X.columns, model.feature_importances_)
        }

    def _select_features_from_scores(
        self,
        *,
        X: pd.DataFrame,
        feature_scores: dict[str, float],
        n_features_to_select: int,
        target_column: str,
        importance_threshold: float,
    ) -> list[str]:
        """Select important features and retain base columns for selected interactions."""
        initial_selected = [
            column for column in X.columns
            if column != target_column and feature_scores.get(column, 0.0) >= importance_threshold
        ]

        if not initial_selected and len(X.columns) > 0:
            ranked_features = self._rank_features(feature_scores)
            fallback_count = n_features_to_select if n_features_to_select > 0 else 1
            initial_selected = [column for column, _ in ranked_features[:fallback_count]]

        protected_base_features = self._collect_protected_base_features(initial_selected, X.columns)
        selected = [
            column for column in X.columns
            if column in set(initial_selected).union(protected_base_features)
        ]
        return selected

    def _collect_protected_base_features(
        self,
        selected_features: list[str],
        available_columns: Any,
    ) -> set[str]:
        """Keep source features when a derived interaction is selected."""
        available = set(available_columns)
        protected: set[str] = set()
        for feature in selected_features:
            if not self._is_derived_feature(feature):
                continue
            protected.update(
                base_feature for base_feature in self._extract_base_features(feature)
                if base_feature in available
            )
        return protected

    def _apply_selection_cap(
        self,
        *,
        selected_features: list[str],
        feature_scores: dict[str, float],
        feature_order: list[str],
        n_features_to_select: int,
    ) -> list[str]:
        """Cap selected features while preserving bases for retained interactions."""
        order_index = {column: index for index, column in enumerate(feature_order)}
        ranked = sorted(
            selected_features,
            key=lambda column: (-feature_scores.get(column, 0.0), order_index.get(column, len(feature_order)), column),
        )
        capped = ranked[:n_features_to_select]
        protected = self._collect_protected_base_features(capped, feature_order)
        kept = [
            column for column in feature_order
            if column in set(capped).union(protected)
        ]
        return kept

    def _encode_features_for_model(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features deterministically for model-based scoring."""
        encoded = X.copy()
        for column in encoded.columns:
            if pd.api.types.is_numeric_dtype(encoded[column]):
                continue
            values = encoded[column].astype(str)
            categories = sorted(values.unique().tolist())
            encoded[column] = pd.Categorical(values, categories=categories).codes
        return encoded

    def _encode_target(self, y: pd.Series) -> tuple[pd.Series, bool]:
        """Encode target values and infer task type."""
        if self._is_classification_target(y):
            values = y.astype(str)
            categories = sorted(values.unique().tolist())
            encoded_y = pd.Series(
                pd.Categorical(values, categories=categories).codes,
                index=y.index,
            )
            return encoded_y, True
        return y.astype(float), False

    def _is_classification_target(self, y: pd.Series) -> bool:
        """Infer whether the target is classification-like."""
        if (
            pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_object_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
        ):
            return True
        if pd.api.types.is_integer_dtype(y):
            unique_count = int(y.nunique(dropna=True))
            return unique_count <= min(20, max(2, len(y) // 5))
        return False

    def _rank_features(self, feature_scores: dict[str, float]) -> list[tuple[str, float]]:
        """Return features sorted by descending importance, then by name."""
        return sorted(
            feature_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

    def _is_derived_feature(self, feature_name: str) -> bool:
        """Return True if the feature is a generated interaction or PCA output."""
        return "__mul__" in feature_name or "__div__" in feature_name or feature_name.startswith("PC")

    def _extract_base_features(self, feature_name: str) -> list[str]:
        """Extract source feature names from a derived feature."""
        if feature_name.startswith("PC"):
            return []
        if not self._is_derived_feature(feature_name):
            return [feature_name]
        return [part for part in re.split(r"__(?:mul|div)__", feature_name) if part]

    def _apply_pca(
        self,
        *,
        X: pd.DataFrame,
        selected_features: list[str],
        generated_features: list[str],
        n_pca_components: Optional[int],
    ) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
        """Apply PCA to selected numeric features while keeping categorical selections."""
        selected_numeric = [
            column for column in selected_features
            if column in X.columns and pd.api.types.is_numeric_dtype(X[column])
        ]
        selected_categorical = [
            column for column in selected_features
            if column in X.columns and column not in selected_numeric
        ]
        if len(selected_numeric) < 2:
            return X, selected_features, generated_features, {}

        n_components = n_pca_components or min(10, len(selected_numeric))
        n_components = min(n_components, len(selected_numeric))
        pca = PCA(n_components=n_components, svd_solver="full")
        components = pca.fit_transform(X[selected_numeric])
        component_names = [f"PC{i + 1}" for i in range(n_components)]

        pca_frame = pd.DataFrame(components, index=X.index, columns=component_names)
        X = X.drop(columns=selected_numeric)
        X = pd.concat([X, pca_frame], axis=1)

        generated_features = [column for column in generated_features if column not in selected_numeric]
        generated_features.extend(component_names)
        selected_features = selected_categorical + component_names
        pca_result = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": component_names,
        }
        return X, selected_features, generated_features, pca_result

    def _split_selected_features(
        self,
        X: pd.DataFrame,
        selected_features: list[str],
        preprocessing_numeric: list[str],
        preprocessing_categorical: list[str],
    ) -> tuple[list[str], list[str]]:
        """Return selected feature groups after engineering."""
        numeric_selected = [
            column for column in selected_features
            if column in X.columns and pd.api.types.is_numeric_dtype(X[column])
        ]
        categorical_selected = [
            column for column in selected_features
            if column in X.columns and column not in numeric_selected
        ]

        for column in preprocessing_numeric:
            if column in selected_features and column not in numeric_selected:
                numeric_selected.append(column)
        for column in preprocessing_categorical:
            if column in selected_features and column not in categorical_selected:
                categorical_selected.append(column)

        return numeric_selected, categorical_selected

    def _build_feature_llm_explanations(
        self,
        *,
        result: dict[str, Any],
        target_column: str,
        y: Optional[pd.Series],
    ) -> dict[str, Any]:
        """Create LLM-backed educational explanations with deterministic fallback."""
        fallback = self._build_fallback_feature_explanations(
            result=result,
            target_column=target_column,
            y=y,
        )

        task_type = fallback["task_type"]
        top_feature_names = fallback["top_feature_names"]
        dropped_feature_names = fallback["dropped_feature_names"]
        feature_scores = result.get("feature_scores", {})
        if not isinstance(feature_scores, dict):
            feature_scores = {}
        payload = {
            "task_type": task_type,
            "target_column": target_column,
            "target_style": (
                "categorical target; explain decisions in terms of class separation"
                if task_type == "classification"
                else "numeric target; explain decisions in terms of predicting a value"
            ),
            "target_preview": self._summarize_target_values(y),
            "feature_engineering_result": {
                "original_feature_count": result.get("feature_engineering_config", {}).get("original_feature_count"),
                "post_engineering_feature_count": result.get("feature_engineering_config", {}).get("post_engineering_feature_count"),
                "final_feature_count": result.get("final_feature_count", 0),
                "selected_features": result.get("selected_features", []),
                "generated_features": result.get("generated_features", []),
                "dropped_columns": result.get("dropped_columns", []),
                "feature_scores": [
                    {"feature": str(name), "score": float(score)}
                    for name, score in self._rank_features({
                        str(name): float(score) for name, score in feature_scores.items()
                    })
                ],
                "applied_transformations": [
                    {
                        "type": str(item.get("type") or ""),
                        "columns": item.get("columns"),
                        "source_columns": item.get("source_columns"),
                        "threshold": item.get("threshold"),
                        "strategy": item.get("strategy"),
                    }
                    for item in result.get("applied_transformations", [])
                    if isinstance(item, dict)
                ],
                "numeric_features": result.get("numeric_features", []),
                "categorical_features": result.get("categorical_features", []),
                "pca_result": result.get("pca_result"),
            },
            "explanation_targets": {
                "top_feature_names": top_feature_names,
                "dropped_feature_names": dropped_feature_names,
            },
            "dropped_feature_reasons": fallback["dropped_reason_lookup"],
            "fallback_guidance": fallback["explanations"],
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an ML tutor writing dashboard copy for a feature-engineering stage. "
                "Use ONLY the structured context provided. Do not invent transformations or claims. "
                "Match the task_type exactly: classification explanations must talk about class separation; "
                "regression explanations must talk about predicting a numeric value. "
                "You are allowed to reason over the full feature-engineering result, including selected features, "
                "generated interactions, dropped features, feature scores, and transformation history, and you should recap the choices clearly. "
                "Make the writing dataset-specific, not generic: each narrative field should mention concrete feature names or target details from the provided context when possible. "
                "Prefer referencing which features stood out, which interactions were created, and which dropped columns were removed. "
                "Return ONLY valid JSON with keys "
                "'stageSummary', 'whatHappened', 'whyItMattered', 'keyTakeaway', 'llmUsed', "
                "'featureExplanations', and 'droppedFeatureExplanations'. "
                "'stageSummary', 'whatHappened', 'whyItMattered', and 'keyTakeaway' must each be 1 to 2 sentences. "
                "'featureExplanations' must be an object keyed only by the listed top_feature_names. "
                "'droppedFeatureExplanations' must be an object keyed only by the listed dropped_feature_names. "
                "'llmUsed' must be true. "
                "Keep the tone beginner-friendly, encouraging, and precise."
            ),
            user_prompt=f"Feature engineering context:\n{self._safe_json(payload)}",
            temperature=0.2,
            max_tokens=1600,
        )

        if not response:
            return fallback["explanations"]

        explanations = {
            "stageSummary": str(response.get("stageSummary") or fallback["explanations"]["stageSummary"]).strip(),
            "whatHappened": str(response.get("whatHappened") or fallback["explanations"]["whatHappened"]).strip(),
            "whyItMattered": str(response.get("whyItMattered") or fallback["explanations"]["whyItMattered"]).strip(),
            "keyTakeaway": str(response.get("keyTakeaway") or fallback["explanations"]["keyTakeaway"]).strip(),
            "llmUsed": True,
            "featureExplanations": self._sanitize_named_explanations(
                response.get("featureExplanations"),
                allowed_names=top_feature_names,
                fallback=fallback["explanations"]["featureExplanations"],
            ),
            "droppedFeatureExplanations": self._sanitize_named_explanations(
                response.get("droppedFeatureExplanations"),
                allowed_names=dropped_feature_names,
                fallback=fallback["explanations"]["droppedFeatureExplanations"],
            ),
        }
        return explanations

    def _build_fallback_feature_explanations(
        self,
        *,
        result: dict[str, Any],
        target_column: str,
        y: Optional[pd.Series],
    ) -> dict[str, Any]:
        """Build deterministic feature-stage explanations when the LLM is unavailable."""
        feature_scores = result.get("feature_scores", {})
        if not isinstance(feature_scores, dict):
            feature_scores = {}

        selected_features = [
            str(feature) for feature in result.get("selected_features", [])
            if str(feature) != target_column
        ]
        generated_features = [str(feature) for feature in result.get("generated_features", [])]
        dropped_columns = [str(feature) for feature in result.get("dropped_columns", [])]
        transformations = [
            item for item in result.get("applied_transformations", [])
            if isinstance(item, dict)
        ]
        task_type = "classification" if y is not None and self._is_classification_target(y) else "regression"

        ranked_features = [
            name for name, _ in sorted(
                ((str(name), float(score)) for name, score in feature_scores.items() if str(name) != target_column),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        top_feature_names = ranked_features[:8]
        dropped_reason_lookup = self._build_dropped_reason_lookup(transformations)
        selected_examples = selected_features[:5]
        generated_examples = generated_features[:4]
        dropped_examples = dropped_columns[:4]

        transformation_types = {str(item.get("type")) for item in transformations if item.get("type")}
        stage_summary = (
            f"We kept {len(selected_features)} final features after engineering {len(generated_features)} new signals "
            f"and removing {len(dropped_columns)} weaker or redundant ones. "
            f"Strong signals included {self._join_examples(top_feature_names[:3])}."
        )
        what_happened = (
            f"The pipeline cleaned the feature set, then kept features like {self._join_examples(selected_examples[:4])} "
            f"for {'separating classes' if task_type == 'classification' else 'predicting the target value'}."
        )
        if generated_examples:
            what_happened += f" It also created interactions such as {self._join_examples(generated_examples[:3])}."
        why_happened_parts = []
        if "generated_interactions" in transformation_types:
            why_happened_parts.append(
                f"Interaction features like {self._join_examples(generated_examples[:2])} let the model notice relationships between columns."
            )
        if "correlation_filter" in transformation_types:
            correlated_examples = [
                column for column in dropped_examples
                if dropped_reason_lookup.get(column) == "correlation_filter"
            ][:2]
            if correlated_examples:
                why_happened_parts.append(
                    f"Correlation filtering removed redundant columns such as {self._join_examples(correlated_examples)}."
                )
            else:
                why_happened_parts.append("Correlation filtering removed features that were telling almost the same story.")
        if "feature_importance_filter" in transformation_types:
            importance_examples = [
                column for column in dropped_examples
                if dropped_reason_lookup.get(column) == "feature_importance_filter"
            ][:2]
            if importance_examples:
                why_happened_parts.append(
                    f"Importance filtering dropped low-signal columns like {self._join_examples(importance_examples)} so the model could focus on stronger patterns."
                )
            else:
                why_happened_parts.append("Importance filtering kept the model focused on features with stronger signal.")
        if not why_happened_parts:
            why_happened_parts.append("This step helps the model focus on stronger signals and ignore noise.")
        why_it_mattered = " ".join(why_happened_parts)
        key_takeaway = (
            f"For this dataset, features like {self._join_examples(top_feature_names[:3])} carried more useful signal than dropped columns like "
            f"{self._join_examples(dropped_examples[:2]) if dropped_examples else 'the weaker candidates'}, giving the model a cleaner set of clues before training."
        )

        feature_explanations = {
            name: self._build_fallback_feature_explanation(name, task_type)
            for name in top_feature_names
        }
        dropped_feature_explanations = {
            name: self._build_fallback_dropped_feature_explanation(
                feature_name=name,
                reason_key=dropped_reason_lookup.get(name, "other"),
                task_type=task_type,
            )
            for name in dropped_columns[:12]
        }

        return {
            "task_type": task_type,
            "top_feature_names": top_feature_names,
            "dropped_feature_names": list(dropped_feature_explanations.keys()),
            "dropped_reason_lookup": {name: dropped_reason_lookup.get(name, "other") for name in dropped_columns[:12]},
            "explanations": {
                "stageSummary": stage_summary,
                "whatHappened": what_happened,
                "whyItMattered": why_it_mattered,
                "keyTakeaway": key_takeaway,
                "llmUsed": False,
                "featureExplanations": feature_explanations,
                "droppedFeatureExplanations": dropped_feature_explanations,
            },
        }

    def _summarize_target_values(self, y: Optional[pd.Series]) -> list[str]:
        """Return a short preview of target values for explanation prompts."""
        if y is None:
            return []
        try:
            unique_values = y.dropna().astype(str).unique().tolist()
        except Exception:
            return []
        return [str(value) for value in unique_values[:4]]

    def _join_examples(self, feature_names: list[str]) -> str:
        """Create a readable comma-separated example list."""
        clean = [name for name in feature_names if name]
        if not clean:
            return "the strongest features"
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        return f"{', '.join(clean[:-1])}, and {clean[-1]}"

    def _build_dropped_reason_lookup(
        self,
        transformations: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Map dropped features to the transformation that removed them."""
        lookup: dict[str, str] = {}
        for transformation in transformations:
            reason_key = str(transformation.get("type") or "other")
            columns = transformation.get("columns", [])
            if not isinstance(columns, list):
                continue
            for column in columns:
                lookup[str(column)] = reason_key
        return lookup

    def _build_fallback_feature_explanation(self, feature_name: str, task_type: str) -> str:
        """Describe a selected feature without using the LLM."""
        if "__mul__" in feature_name:
            left, right = feature_name.split("__mul__", 1)
            return (
                f"This engineered feature multiplies {left} and {right}, which helps the model see how those signals work together "
                f"when {('separating classes' if task_type == 'classification' else 'predicting the target value')}."
            )
        if "__div__" in feature_name:
            left, right = feature_name.split("__div__", 1)
            return (
                f"This engineered feature compares {left} to {right}, giving the model a relative measurement that can sharpen "
                f"{'classification' if task_type == 'classification' else 'regression'} predictions."
            )
        if feature_name.startswith("PC"):
            return (
                f"{feature_name} is a compressed numeric summary that keeps important variation while simplifying the feature space."
            )
        return (
            f"{feature_name} stayed in the final set because it appeared to provide a useful direct signal for "
            f"{'classification' if task_type == 'classification' else 'regression'}."
        )

    def _build_fallback_dropped_feature_explanation(
        self,
        *,
        feature_name: str,
        reason_key: str,
        task_type: str,
    ) -> str:
        """Explain why a feature was dropped without using the LLM."""
        if reason_key == "feature_importance_filter":
            return (
                f"{feature_name} was removed because it added very little extra signal for this {task_type} problem."
            )
        if reason_key == "correlation_filter":
            return (
                f"{feature_name} was removed because it overlapped strongly with another numeric feature."
            )
        if reason_key == "selection_cap":
            return (
                f"{feature_name} was a reasonable candidate, but it did not rank high enough to survive the final feature cap."
            )
        if reason_key == "drop_index_like_columns":
            return (
                f"{feature_name} looked more like an ID or bookkeeping column than a real predictive signal."
            )
        return f"{feature_name} was filtered out during feature cleanup."

    def _sanitize_named_explanations(
        self,
        raw_value: Any,
        *,
        allowed_names: list[str],
        fallback: dict[str, str],
    ) -> dict[str, str]:
        """Keep only LLM explanations for the expected feature names."""
        if not isinstance(raw_value, dict):
            return fallback

        cleaned: dict[str, str] = {}
        for name in allowed_names:
            value = raw_value.get(name)
            if isinstance(value, str) and value.strip():
                cleaned[name] = value.strip()

        if not cleaned:
            return fallback

        for name, value in fallback.items():
            cleaned.setdefault(name, value)
        return cleaned
