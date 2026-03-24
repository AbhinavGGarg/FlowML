"""Deterministic preprocessing policies for the AutoML pipeline."""

from __future__ import annotations

import math
import re
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "small_dataset_row_threshold": 1000,
    "drop_rows_max_fraction": 0.04,
    "drop_column_missing_fraction": 0.5,
    "onehot_max_categories": 20,
    "rare_category_fraction": 0.01,
    "rare_category_min_levels": 10,
    "rare_category_max_group_fraction": 0.2,
    "near_constant_fraction": 0.99,
    "skew_threshold": 1.0,
    "outlier_fraction_threshold": 0.05,
    "robust_scaler_min_outlier_columns": 1,
    "identifier_unique_fraction": 0.98,
}

IDENTIFIER_NAME_TOKENS = (
    "id",
    "uuid",
    "guid",
    "row_id",
    "customer_id",
    "user_id",
    "account_id",
    "member_id",
    "order_id",
    "transaction_id",
    "record_id",
    "index",
)

BOOLEAN_STRINGS = {
    "0",
    "1",
    "false",
    "true",
    "n",
    "no",
    "y",
    "yes",
    "f",
    "t",
}


def get_default_preprocessing_config(
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return the default preprocessing configuration with optional overrides."""
    config = dict(DEFAULT_PREPROCESSING_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def normalize_name(value: str) -> str:
    """Normalize a column name for deterministic string matching."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return cleaned.strip("_")


def detect_column_types(df: pd.DataFrame) -> dict[str, Any]:
    """Detect numeric, categorical, binary, and datetime columns."""
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    binary_columns: list[str] = []
    datetime_columns: list[str] = []
    parsed_datetimes: dict[str, pd.Series] = {}

    for column in df.columns:
        series = df[column]
        if _looks_like_datetime(series):
            datetime_columns.append(column)
            parsed_datetimes[column] = pd.to_datetime(series, errors="coerce")
            continue

        if _is_binary_series(series):
            binary_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_columns": binary_columns,
        "datetime_columns": datetime_columns,
        "parsed_datetimes": parsed_datetimes,
    }


def expand_datetime_columns(
    df: pd.DataFrame,
    parsed_datetimes: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Expand datetime columns into numeric calendar components."""
    if not parsed_datetimes:
        return df.copy(), {}

    expanded = df.copy()
    created_columns: dict[str, list[str]] = {}

    for column, parsed in parsed_datetimes.items():
        if column not in expanded.columns:
            continue

        parts: list[str] = []
        parsed_series = parsed.reindex(expanded.index)
        candidates: list[tuple[str, pd.Series]] = [
            ("year", parsed_series.dt.year),
            ("month", parsed_series.dt.month),
            ("day", parsed_series.dt.day),
            ("dayofweek", parsed_series.dt.dayofweek),
        ]

        has_clock_signal = bool(
            parsed_series.dt.hour.fillna(0).any()
            or parsed_series.dt.minute.fillna(0).any()
            or parsed_series.dt.second.fillna(0).any()
        )
        if has_clock_signal:
            candidates.extend(
                [
                    ("hour", parsed_series.dt.hour),
                    ("minute", parsed_series.dt.minute),
                ]
            )

        for suffix, values in candidates:
            if values.notna().sum() == 0:
                continue
            if values.nunique(dropna=True) <= 1:
                continue
            new_column = f"{column}__{suffix}"
            expanded[new_column] = values.astype(float)
            parts.append(new_column)

        if not parts:
            ordinal_column = f"{column}__ordinal"
            expanded[ordinal_column] = parsed_series.map(
                lambda value: float(value.toordinal()) if pd.notna(value) else np.nan
            )
            parts.append(ordinal_column)

        expanded = expanded.drop(columns=[column])
        created_columns[column] = parts

    return expanded, created_columns


def analyze_missingness(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize missingness across the dataframe."""
    missing_counts = df.isna().sum()
    rows_with_missing_mask = df.isna().any(axis=1) if not df.empty else pd.Series(dtype=bool)
    row_count = len(df) or 1
    return {
        "had_missing": bool(int(missing_counts.sum()) > 0),
        "total_missing_cells": int(missing_counts.sum()),
        "rows_with_missing": int(rows_with_missing_mask.sum()) if not df.empty else 0,
        "rows_with_missing_fraction": float(rows_with_missing_mask.mean()) if not df.empty else 0.0,
        "missing_by_column": {
            str(column): float(missing_counts[column] / row_count)
            for column in df.columns
            if int(missing_counts[column]) > 0
        },
    }


def choose_missing_value_strategy(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    protected_columns: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Choose a deterministic missing-value strategy for the dataset."""
    protected_columns = protected_columns or set()
    analysis = analyze_missingness(df)
    missing_by_column = analysis["missing_by_column"]
    row_count = len(df)

    if not analysis["had_missing"]:
        return {
            "strategy_used": "none",
            "strategy_reason": "No missing values were found in the feature columns.",
            "drop_columns": [],
            "drop_rows": False,
        }

    sparse_candidates = [
        column
        for column, fraction in missing_by_column.items()
        if fraction >= float(config["drop_column_missing_fraction"]) and column not in protected_columns
    ]
    drop_columns = _limit_drop_columns(df.columns.tolist(), sparse_candidates)
    remaining = df.drop(columns=drop_columns, errors="ignore")
    rows_with_missing_fraction = (
        float(remaining.isna().any(axis=1).mean()) if not remaining.empty else 0.0
    )

    large_dataset = row_count >= int(config["small_dataset_row_threshold"])
    safe_row_drop = 0.0 < rows_with_missing_fraction <= float(config["drop_rows_max_fraction"])
    drop_rows = large_dataset and safe_row_drop
    forced_strategy = str(config.get("missing_value_strategy") or "auto")
    protect_rows = bool(config.get("protect_rows_from_drop", False))

    if drop_columns and drop_rows:
        strategy_used = "mixed"
        strategy_reason = (
            f"Dropped {len(drop_columns)} sparse column(s) first, then removed a small share of rows "
            f"({rows_with_missing_fraction:.1%}) with remaining missing values because the dataset is large."
        )
    elif drop_columns and remaining.isna().sum().sum() > 0:
        strategy_used = "mixed"
        strategy_reason = (
            f"Dropped {len(drop_columns)} sparse column(s), then kept the remaining rows and imputed the rest "
            "because dropping more rows would remove too much data."
        )
    elif drop_columns:
        strategy_used = "drop_columns"
        strategy_reason = (
            f"Dropped {len(drop_columns)} sparse column(s) with at least "
            f"{float(config['drop_column_missing_fraction']):.0%} missing values."
        )
    elif drop_rows:
        strategy_used = "drop_rows"
        strategy_reason = (
            f"Only {rows_with_missing_fraction:.1%} of rows had missing values after reviewing the columns, "
            "so dropping those rows preserves most of the dataset."
        )
    else:
        strategy_used = "impute"
        strategy_reason = (
            "Missing values were imputed to preserve rows because the dataset is relatively small or "
            "dropping rows would discard too much information."
        )

    if protect_rows:
        drop_rows = False
        strategy_used = "impute"
        strategy_reason = "Row dropping was disabled by the revision agent, so remaining missing values were imputed."
    elif forced_strategy != "auto":
        if forced_strategy == "impute":
            drop_rows = False
            strategy_used = "impute"
            strategy_reason = "The revision agent forced an imputation-first missing-value strategy."
        elif forced_strategy == "drop_rows":
            drop_rows = analysis["rows_with_missing"] > 0
            strategy_used = "drop_rows"
            strategy_reason = "The revision agent requested row dropping for missing values."
        elif forced_strategy == "drop_columns":
            drop_rows = False
            strategy_used = "drop_columns"
            strategy_reason = "The revision agent requested column dropping for sparse columns only."
        elif forced_strategy == "mixed":
            strategy_used = "mixed"
            strategy_reason = "The revision agent requested the mixed missing-value handling policy."

    return {
        "strategy_used": strategy_used,
        "strategy_reason": strategy_reason,
        "drop_columns": drop_columns,
        "drop_rows": drop_rows,
    }


def detect_identifier_columns(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Detect likely identifier or bookkeeping columns."""
    findings: list[dict[str, str]] = []
    row_count = len(df)
    unique_fraction_threshold = float(config["identifier_unique_fraction"])

    for column in df.columns:
        series = df[column]
        normalized = normalize_name(column)
        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_count = int(non_null.nunique(dropna=True))
        unique_fraction = float(unique_count / max(len(non_null), 1))
        name_is_identifier = (
            normalized.startswith("unnamed")
            or normalized in IDENTIFIER_NAME_TOKENS
            or normalized.endswith("_id")
            or any(token in normalized for token in ("uuid", "guid"))
        )

        if normalized.startswith("unnamed"):
            findings.append({
                "column": column,
                "reason": "identifier",
                "detail": "looks like an imported index column",
            })
            continue

        if name_is_identifier:
            findings.append({
                "column": column,
                "reason": "identifier",
                "detail": "name strongly suggests an identifier column",
            })
            continue

        looks_like_code = _looks_like_code_series(non_null)
        if (
            row_count >= 20
            and unique_count >= max(20, int(row_count * 0.5))
            and unique_fraction >= unique_fraction_threshold
            and (_looks_like_sequential_index(non_null) or looks_like_code)
        ):
            findings.append({
                "column": column,
                "reason": "identifier",
                "detail": "values are near-unique and look like record identifiers",
            })

    return findings


def detect_low_variance_columns(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Detect constant and near-constant columns."""
    findings: list[dict[str, str]] = []
    threshold = float(config["near_constant_fraction"])

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_count = int(non_null.nunique(dropna=True))
        if unique_count <= 1:
            findings.append({
                "column": column,
                "reason": "constant",
                "detail": "all observed values are the same",
            })
            continue

        top_fraction = float(non_null.astype(str).value_counts(normalize=True, dropna=False).iloc[0])
        if top_fraction >= threshold and unique_count <= min(10, len(non_null)):
            findings.append({
                "column": column,
                "reason": "other",
                "detail": "near-constant values leave almost no useful variation",
            })

    return findings


def detect_target_leakage_risks(
    df: pd.DataFrame,
    target: pd.Series,
    target_column: str,
) -> dict[str, Any]:
    """Detect suspicious columns that may leak target information."""
    suspicious_columns: list[dict[str, str]] = []
    dropped_columns: list[dict[str, str]] = []
    normalized_target = normalize_name(target_column)
    target_as_text = target.astype(str)

    for column in df.columns:
        normalized = normalize_name(column)
        series = df[column]
        series_as_text = series.astype(str)

        if normalized == normalized_target:
            dropped_columns.append({
                "column": column,
                "reason": "leakage_risk",
                "detail": "feature name matches the target name",
            })
            continue

        if (
            normalized_target
            and normalized != normalized_target
            and (normalized_target in normalized or normalized in normalized_target)
        ):
            suspicious_columns.append({
                "column": column,
                "reason": "name_similarity",
                "detail": "feature name is unusually similar to the target name",
            })

        aligned = pd.concat(
            [series_as_text.rename("feature"), target_as_text.rename("target")],
            axis=1,
        ).dropna()
        if not aligned.empty and aligned["feature"].equals(aligned["target"]):
            dropped_columns.append({
                "column": column,
                "reason": "leakage_risk",
                "detail": "feature values duplicate the target values",
            })

    return {
        "suspicious_columns": suspicious_columns,
        "dropped_columns": dropped_columns,
    }


def choose_encoding_strategy(
    train_df: pd.DataFrame,
    categorical_columns: list[str],
    binary_columns: list[str],
    *,
    config: dict[str, Any],
) -> dict[str, str]:
    """Choose a deterministic encoding strategy for each categorical column."""
    strategies: dict[str, str] = {}

    for column in binary_columns:
        if column in train_df.columns:
            strategies[column] = "binary"

    for column in categorical_columns:
        if column not in train_df.columns:
            continue
        cardinality = int(train_df[column].nunique(dropna=True))
        override = str(config.get("encoding_strategy_overrides", {}).get(column, "")).strip()
        if override in {"onehot", "frequency"}:
            strategies[column] = override
        elif cardinality <= int(config["onehot_max_categories"]):
            strategies[column] = "onehot"
        else:
            strategies[column] = "frequency"

    return strategies


def detect_rare_categories(
    series: pd.Series,
    *,
    rare_fraction: float,
    min_levels: int,
    max_group_fraction: float,
) -> list[str]:
    """Detect rare levels that can be safely grouped into 'Other'."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return []

    frequencies = non_null.value_counts(normalize=True)
    if len(frequencies) < min_levels:
        return []

    rare = frequencies[frequencies < rare_fraction]
    if len(rare) < 2:
        return []

    if float(rare.sum()) > max_group_fraction:
        return []

    return sorted(str(item) for item in rare.index.tolist())


def detect_skewed_numeric_columns(
    train_df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    skew_threshold: float,
) -> list[str]:
    """Detect strongly positively skewed numeric columns."""
    skewed: list[str] = []

    for column in numeric_columns:
        if column not in train_df.columns:
            continue
        series = pd.to_numeric(train_df[column], errors="coerce").dropna()
        if len(series) < 8:
            continue
        if series.nunique(dropna=True) <= 5:
            continue
        skewness = float(series.skew())
        if math.isfinite(skewness) and skewness >= skew_threshold and float(series.min()) >= 0.0:
            skewed.append(column)

    return skewed


def choose_scaler(
    train_df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Choose between standard and robust scaling for numeric columns."""
    scaler_override = str(config.get("scaler") or "auto")
    if scaler_override in {"StandardScaler", "RobustScaler", "None"}:
        return {
            "scaler": scaler_override,
            "reason": f"Used {scaler_override} because the revision agent overrode scaler selection." if scaler_override != "None" else "Skipped scaling because the revision agent disabled it.",
            "outlier_heavy_columns": [],
        }

    outlier_heavy_columns: list[str] = []
    threshold = float(config["outlier_fraction_threshold"])

    for column in numeric_columns:
        if column not in train_df.columns:
            continue
        series = pd.to_numeric(train_df[column], errors="coerce").dropna()
        if len(series) < 8:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        outlier_fraction = float(((series < lower) | (series > upper)).mean())
        if outlier_fraction >= threshold:
            outlier_heavy_columns.append(column)

    use_robust = len(outlier_heavy_columns) >= int(config["robust_scaler_min_outlier_columns"])
    scaler_name = "RobustScaler" if use_robust and numeric_columns else "StandardScaler" if numeric_columns else "None"
    reason = (
        f"Used RobustScaler because {len(outlier_heavy_columns)} numeric column(s) showed strong outlier behavior."
        if scaler_name == "RobustScaler"
        else "Used StandardScaler because numeric columns did not show strong outlier pressure."
        if scaler_name == "StandardScaler"
        else "No numeric columns needed scaling."
    )

    return {
        "scaler": scaler_name,
        "reason": reason,
        "outlier_heavy_columns": outlier_heavy_columns,
    }


def summarize_target_distribution(
    target: pd.Series,
    task_type: str,
) -> dict[str, Any]:
    """Summarize the target distribution and imbalance status."""
    if task_type == "classification":
        distribution = {
            str(label): int(count)
            for label, count in target.value_counts(dropna=False).to_dict().items()
        }
        counts = list(distribution.values())
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        imbalance_ratio = float(max_count / min_count) if min_count else math.inf if max_count else 0.0
        severity = "none"
        if imbalance_ratio >= 10:
            severity = "severe"
        elif imbalance_ratio >= 3:
            severity = "moderate"
        elif imbalance_ratio >= 1.5:
            severity = "mild"

        return {
            "class_distribution": distribution,
            "imbalance_detected": severity in {"mild", "moderate", "severe"},
            "imbalance_ratio": imbalance_ratio,
            "imbalance_severity": severity,
        }

    numeric_target = pd.to_numeric(target, errors="coerce")
    return {
        "class_distribution": {},
        "imbalance_detected": False,
        "target_range": {
            "min": float(numeric_target.min()) if numeric_target.notna().any() else None,
            "max": float(numeric_target.max()) if numeric_target.notna().any() else None,
            "mean": float(numeric_target.mean()) if numeric_target.notna().any() else None,
            "std": float(numeric_target.std()) if numeric_target.notna().any() else None,
        },
    }


def infer_task_type(target: pd.Series) -> str:
    """Infer the task type from the target column."""
    if (
        pd.api.types.is_bool_dtype(target)
        or pd.api.types.is_object_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        return "classification"

    non_null = target.dropna()
    if non_null.empty:
        return "regression"

    if pd.api.types.is_integer_dtype(non_null) and int(non_null.nunique(dropna=True)) <= min(20, max(2, len(non_null) // 10)):
        return "classification"

    return "regression"


def build_preprocessing_explanation(result: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic beginner-friendly explanation from actual decisions."""
    dropped_columns = result.get("dropped_columns", [])
    missing_summary = result.get("missing_summary", {})
    categorical_summary = result.get("categorical_summary", {})
    scaling_summary = result.get("scaling_summary", {})
    transform_summary = result.get("transform_summary", {})
    target_summary = result.get("target_summary", {})
    datetime_summary = result.get("datetime_summary", {})

    decisions: list[str] = []

    if dropped_columns:
        examples = ", ".join(item["column"] for item in dropped_columns[:3] if isinstance(item, dict))
        decisions.append(
            f"Removed {len(dropped_columns)} non-useful column(s), including {examples}, before modeling."
        )

    strategy_used = str(missing_summary.get("strategy_used") or "none")
    if strategy_used == "none":
        decisions.append("No missing values were found after initial cleanup, so no imputation was needed.")
    else:
        decisions.append(str(missing_summary.get("strategy_reason") or "Missing-value rules were applied deterministically."))

    high_cardinality = categorical_summary.get("high_cardinality_columns", []) or []
    rare_grouped = categorical_summary.get("rare_category_grouped_columns", []) or []
    if high_cardinality:
        decisions.append(
            f"Used safer categorical handling for high-cardinality column(s) like {', '.join(high_cardinality[:3])}."
        )
    elif categorical_summary.get("encoded_columns"):
        decisions.append(
            f"Encoded categorical column(s) like {', '.join(categorical_summary.get('encoded_columns', [])[:3])}."
        )

    if rare_grouped:
        decisions.append(
            f"Grouped rare levels into 'Other' for {', '.join(rare_grouped[:3])} to keep category noise under control."
        )

    if datetime_summary.get("expanded_columns"):
        expanded_names = list(datetime_summary["expanded_columns"].keys())
        decisions.append(
            f"Expanded datetime column(s) like {', '.join(expanded_names[:2])} into calendar features."
        )

    scaler = scaling_summary.get("scaler", "None")
    if scaler != "None":
        decisions.append(str(scaling_summary.get("reason") or f"Applied {scaler} to numeric columns."))

    log_columns = transform_summary.get("log_transformed_columns", []) or []
    if log_columns:
        decisions.append(
            f"Applied log1p to skewed numeric column(s) like {', '.join(log_columns[:3])} before scaling."
        )

    target_line = ""
    if target_summary.get("task_type") == "classification":
        severity = target_summary.get("imbalance_severity", "none")
        distribution = target_summary.get("class_distribution", {})
        target_line = (
            f" The target looked like a classification problem with {len(distribution)} class(es) and {severity} imbalance."
        )

    summary_parts = [
        f"We prepared {result.get('raw_feature_count', 0)} raw feature(s) for modeling and ended with "
        f"{result.get('transformed_feature_count', 0)} transformed feature(s).",
        f"The final train/test split used {result.get('train_size', 0)} training rows and {result.get('test_size', 0)} test rows.",
    ]
    summary = " ".join(summary_parts) + target_line

    why = (
        "These deterministic steps remove columns that are unlikely to help, keep missing-value handling conservative, "
        "and transform features in a way that is safe to explain to beginners."
    )

    return {
        "summary": summary.strip(),
        "decisions": decisions[:6],
        "why": why,
    }


def _looks_like_datetime(series: pd.Series) -> bool:
    """Return True when an object-like series parses cleanly as datetimes."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) < 3:
        return False

    sample = non_null.astype(str).head(100)
    if sample.str.fullmatch(r"\d+").mean() > 0.9:
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`.*",
            category=UserWarning,
        )
        parsed = pd.to_datetime(sample, errors="coerce")
    success_rate = float(parsed.notna().mean())
    return success_rate >= 0.8


def _is_binary_series(series: pd.Series) -> bool:
    """Return True when a series looks binary."""
    if pd.api.types.is_bool_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_values = pd.unique(non_null)
    if len(unique_values) > 2:
        return False

    if pd.api.types.is_numeric_dtype(non_null):
        return True

    normalized = {str(value).strip().lower() for value in unique_values}
    return normalized.issubset(BOOLEAN_STRINGS)


def _looks_like_code_series(series: pd.Series) -> bool:
    """Return True when the values resemble IDs or compact codes."""
    sample = series.astype(str).head(50)
    if sample.empty:
        return False
    return bool((sample.str.contains(r"^[A-Za-z0-9\\-_]+$", regex=True)).mean() >= 0.9)


def _looks_like_sequential_index(series: pd.Series) -> bool:
    """Return True when numeric values behave like a generated row index."""
    if not pd.api.types.is_numeric_dtype(series):
        return False

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 5:
        return False
    if not np.allclose(numeric, np.round(numeric)):
        return False

    diffs = np.diff(numeric.to_numpy(dtype=float))
    if diffs.size == 0:
        return False
    return bool(np.allclose(diffs, diffs[0]) and abs(float(diffs[0])) == 1.0)


def _limit_drop_columns(columns: list[str], candidates: list[str]) -> list[str]:
    """Avoid dropping every feature column in extreme cases."""
    if len(candidates) >= len(columns):
        return sorted(candidates[:-1]) if len(candidates) > 1 else []
    return sorted(candidates)
