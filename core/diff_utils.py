"""Utilities for producing safe pipeline diffs and comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def public_value(value: Any) -> Any:
    """Convert complex values into public, JSON-safe shapes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): public_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [public_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(len(value)),
            "columns": [str(column) for column in value.columns],
        }
    return str(value)


def diff_mapping(before: Any, after: Any) -> Any:
    """Return a nested diff for two values."""
    left = public_value(before)
    right = public_value(after)

    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left).union(right))
        diff: dict[str, Any] = {}
        for key in keys:
            if key not in left:
                diff[key] = {"before": None, "after": right[key]}
                continue
            if key not in right:
                diff[key] = {"before": left[key], "after": None}
                continue
            nested = diff_mapping(left[key], right[key])
            if nested not in ({}, None):
                diff[key] = nested
        return diff

    if left != right:
        return {"before": left, "after": right}
    return {}


def list_diff(before: list[str], after: list[str]) -> dict[str, list[str]]:
    """Summarize list additions and removals."""
    left = [str(item) for item in before]
    right = [str(item) for item in after]
    return {
        "added": [item for item in right if item not in left],
        "removed": [item for item in left if item not in right],
    }


def build_comparison_summary(
    changed_stages: list[str],
    metric_diffs: dict[str, Any],
    feature_diffs: dict[str, list[str]],
) -> str:
    """Create a compact comparison summary."""
    parts: list[str] = []

    if changed_stages:
        parts.append(f"Changed stages: {', '.join(changed_stages)}.")

    highlighted_metrics = [
        name
        for name, diff in metric_diffs.items()
        if isinstance(diff, dict) and "before" in diff and "after" in diff
    ]
    if highlighted_metrics:
        parts.append(
            "Metrics changed for "
            + ", ".join(highlighted_metrics[:4])
            + "."
        )

    added = feature_diffs.get("added", [])
    removed = feature_diffs.get("removed", [])
    if added:
        parts.append(f"Added feature(s): {', '.join(added[:4])}.")
    if removed:
        parts.append(f"Removed feature(s): {', '.join(removed[:4])}.")

    if not parts:
        return "The two runs produced the same tracked configuration and summary outputs."
    return " ".join(parts)
