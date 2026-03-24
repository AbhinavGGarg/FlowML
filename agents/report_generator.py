"""Pipeline report generator for deployment packages.

Builds a site-themed report.html summarizing all pipeline panels and embeds
server-side generated charts as base64 PNGs.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from agents.base_agent import BaseAgent


class ReportGenerator(BaseAgent):
    """Generate HTML pipeline reports using stage results."""

    def __init__(self) -> None:
        super().__init__("ReportGenerator")

    async def execute(self, *args, **kwargs) -> dict[str, Any]:
      """BaseAgent compatibility wrapper."""
      return self.generate_assets(*args, **kwargs)

    def generate_assets(
        self,
        *,
        pipeline_id: str,
        dataset_name: Optional[str],
        target_column: Optional[str],
        analysis_result: Optional[dict[str, Any]],
        preprocessing_result: Optional[dict[str, Any]],
        features_result: Optional[dict[str, Any]],
        model_selection_result: Optional[dict[str, Any]],
        training_result: Optional[dict[str, Any]],
        evaluation_result: Optional[dict[str, Any]],
        evaluation_insights: Optional[dict[str, Any]],
        explanation_result: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return HTML report assets."""
        analysis = analysis_result or {}
        preprocessing = preprocessing_result or {}
        features = features_result or {}
        model_selection = model_selection_result or {}
        training = training_result or {}
        evaluation = evaluation_result or {}
        insights = evaluation_insights or {}
        explanation = explanation_result or {}

        charts = self._build_charts(
            analysis=analysis,
            features=features,
            training=training,
            evaluation=evaluation,
        )

        context = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "pipeline_id": pipeline_id,
            "dataset_name": dataset_name or "unknown",
            "target_column": target_column or "unknown",
            "model_name": str(training.get("model_name") or "Unknown"),
            "task_type": str(evaluation.get("task_type") or "classification"),
            "deployment_decision": str(evaluation.get("deployment_decision") or "review"),
            "analysis": analysis,
            "preprocessing": preprocessing,
            "features": features,
            "model_selection": model_selection,
            "training": training,
            "evaluation": evaluation,
            "insights": insights,
            "explanation": explanation,
            "charts": charts,
        }

        html = self._render_html(context)
        return {"html": html}

    def _build_charts(
        self,
        *,
        analysis: dict[str, Any],
        features: dict[str, Any],
        training: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Optional[str]]:
        """Build all charts as base64 data URIs."""
        return {
            "correlation_heatmap": self._chart_correlation_heatmap(analysis.get("correlations")),
            "feature_importance": self._chart_feature_importance(features.get("feature_scores")),
            "cv_scores": self._chart_cv_scores(training.get("cv_scores"), training.get("model_comparisons")),
            "confusion_matrix": self._chart_confusion_matrix(evaluation.get("confusion_matrix"), evaluation.get("class_labels")),
            "actual_vs_predicted": self._chart_actual_vs_predicted(evaluation.get("y_test"), evaluation.get("predictions")),
        }

    def _figure_to_data_uri(self, fig: Any) -> str:
        import matplotlib.pyplot as plt

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _chart_correlation_heatmap(self, correlations: Any) -> Optional[str]:
        if not isinstance(correlations, dict) or not correlations:
            return None
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            corr_df = pd.DataFrame(correlations)
            if corr_df.empty:
                return None
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#111827")
            image = ax.imshow(corr_df.values, cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_df.columns)))
            ax.set_yticks(range(len(corr_df.index)))
            ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=7, color="#e5e7eb")
            ax.set_yticklabels(corr_df.index, fontsize=7, color="#e5e7eb")
            ax.set_title("Correlation Heatmap", color="#f3f4f6", fontsize=12)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="#9ca3af")
            plt.setp(cbar.ax.get_yticklabels(), color="#d1d5db", fontsize=8)
            return self._figure_to_data_uri(fig)
        except Exception:
            return None

    def _chart_feature_importance(self, feature_scores: Any) -> Optional[str]:
        if not isinstance(feature_scores, dict) or not feature_scores:
            return None
        try:
            import matplotlib.pyplot as plt

            items = sorted(feature_scores.items(), key=lambda item: float(item[1]), reverse=True)[:15]
            labels = [str(key) for key, _ in items][::-1]
            values = [float(value) for _, value in items][::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#111827")
            ax.barh(labels, values, color="#8b5cf6")
            ax.set_title("Top Feature Importance", color="#f3f4f6", fontsize=12)
            ax.tick_params(axis="x", colors="#d1d5db")
            ax.tick_params(axis="y", colors="#d1d5db", labelsize=8)
            ax.grid(axis="x", alpha=0.2, color="#374151")
            return self._figure_to_data_uri(fig)
        except Exception:
            return None

    def _chart_cv_scores(self, cv_scores: Any, model_comparisons: Any) -> Optional[str]:
        try:
            import matplotlib.pyplot as plt

            if isinstance(model_comparisons, list) and model_comparisons:
                pairs: list[tuple[str, float]] = []
                for item in model_comparisons[:6]:
                    if not isinstance(item, dict):
                        continue
                    raw_value = item.get("cv_mean", item.get("mean_cv_score"))
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        continue
                    pairs.append((str(item.get("model_name", "model")), value))

                if not pairs or all(abs(value) < 1e-12 for _, value in pairs):
                    return None

                labels = [label for label, _ in pairs]
                values = [value for _, value in pairs]
                fig, ax = plt.subplots(figsize=(8, 4.5))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#111827")
                colors = ["#10b981" if i == 0 else "#8b5cf6" for i in range(len(labels))]
                ax.bar(labels, values, color=colors)
                ax.set_title("Model Comparison (CV)", color="#f3f4f6", fontsize=12)
                ax.tick_params(axis="x", rotation=25, colors="#d1d5db", labelsize=8)
                ax.tick_params(axis="y", colors="#d1d5db")
                ax.grid(axis="y", alpha=0.2, color="#374151")
                return self._figure_to_data_uri(fig)

            if isinstance(cv_scores, list) and cv_scores:
                values = []
                for value in cv_scores:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(numeric):
                        values.append(numeric)

                if not values or all(abs(value) < 1e-12 for value in values):
                    return None

                labels = [f"Fold {i + 1}" for i in range(len(values))]
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#111827")
                ax.bar(labels, values, color="#8b5cf6")
                ax.axhline(float(np.mean(values)), color="#10b981", linestyle="--", linewidth=1.2, label="Mean")
                ax.legend(facecolor="#0f172a", edgecolor="#1f2937", labelcolor="#d1d5db")
                ax.set_title("Cross-Validation Scores", color="#f3f4f6", fontsize=12)
                ax.tick_params(axis="x", colors="#d1d5db", labelsize=8)
                ax.tick_params(axis="y", colors="#d1d5db")
                ax.grid(axis="y", alpha=0.2, color="#374151")
                return self._figure_to_data_uri(fig)
        except Exception:
            return None
        return None

    def _chart_confusion_matrix(self, matrix: Any, class_labels: Any) -> Optional[str]:
        if not isinstance(matrix, list) or not matrix:
            return None
        try:
            import matplotlib.pyplot as plt

            mat = np.array(matrix)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#111827")
            image = ax.imshow(mat, cmap="magma")
            labels = class_labels if isinstance(class_labels, list) and class_labels else list(range(mat.shape[0]))
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels([str(label) for label in labels], color="#d1d5db")
            ax.set_yticklabels([str(label) for label in labels], color="#d1d5db")
            ax.set_xlabel("Predicted", color="#d1d5db")
            ax.set_ylabel("Actual", color="#d1d5db")
            ax.set_title("Confusion Matrix", color="#f3f4f6", fontsize=12)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="#f9fafb", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            return self._figure_to_data_uri(fig)
        except Exception:
            return None

    def _chart_actual_vs_predicted(self, y_true: Any, y_pred: Any) -> Optional[str]:
        if not isinstance(y_true, list) or not isinstance(y_pred, list) or not y_true or not y_pred:
            return None
        try:
            import matplotlib.pyplot as plt

            x = np.array([float(v) for v in y_true])
            y = np.array([float(v) for v in y_pred])
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#111827")
            ax.scatter(x, y, s=14, alpha=0.7, color="#10b981")
            ax.plot([lo, hi], [lo, hi], "--", color="#8b5cf6", linewidth=1)
            ax.set_xlabel("Actual", color="#d1d5db")
            ax.set_ylabel("Predicted", color="#d1d5db")
            ax.set_title("Actual vs Predicted", color="#f3f4f6", fontsize=12)
            ax.tick_params(axis="both", colors="#d1d5db")
            ax.grid(alpha=0.2, color="#374151")
            return self._figure_to_data_uri(fig)
        except Exception:
            return None

    def _render_html(self, context: dict[str, Any]) -> str:
        """Render the report HTML using a template."""
        template = _REPORT_TEMPLATE
        try:
            from jinja2 import Template

            return Template(template).render(**context)
        except Exception:
            # Basic fallback rendering if jinja2 is unavailable
            return template.replace("{{ pipeline_id }}", str(context.get("pipeline_id", "unknown")))

_REPORT_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>FlowML Pipeline Report</title>
  <style>
    :root {
      --background: hsl(220, 24%, 8%);
      --foreground: hsl(210, 40%, 98%);
      --card: hsl(220, 22%, 11%);
      --secondary: hsl(220, 18%, 16%);
      --muted: hsl(220, 16%, 18%);
      --muted-foreground: hsl(215, 20%, 67%);
      --accent: hsl(154, 77%, 49%);
      --primary: hsl(266, 92%, 67%);
      --border: hsl(220, 18%, 22%);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.45;
    }
    .wrap { max-width: 1120px; margin: 0 auto; padding: 28px; }
    .glass {
      background: rgba(23, 28, 44, 0.78);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      margin-bottom: 16px;
    }
    h1, h2, h3 { margin: 0 0 10px 0; }
    h1 { font-size: 28px; }
    h2 { font-size: 18px; color: var(--accent); margin-top: 8px; }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .kpi { background: var(--secondary); border-radius: 12px; padding: 10px; border: 1px solid var(--border); }
    .kpi .label { color: var(--muted-foreground); font-size: 12px; }
    .kpi .val { font-weight: 700; font-size: 18px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 11px; background: var(--muted); margin-right: 6px; margin-bottom: 6px; }
    .table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .table th, .table td { border-bottom: 1px solid var(--border); padding: 7px 8px; text-align: left; }
    .muted { color: var(--muted-foreground); }
    .img { width: 100%; border-radius: 12px; border: 1px solid var(--border); background: #0f172a; }
    .flag { border-radius: 10px; padding: 8px 10px; margin-bottom: 8px; font-size: 13px; }
    .flag.high { background: rgba(239,68,68,0.16); border: 1px solid rgba(239,68,68,0.35); }
    .flag.medium { background: rgba(245,158,11,0.16); border: 1px solid rgba(245,158,11,0.35); }
    .flag.low { background: rgba(59,130,246,0.16); border: 1px solid rgba(59,130,246,0.35); }
    .split { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    @media (max-width: 960px) {
      .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"glass\">
      <div class=\"split\">
        <div>
          <h1>Pipeline Report</h1>
          <p class=\"muted\">Pipeline {{ pipeline_id }} · Generated {{ generated_at }}</p>
          <p><strong>Dataset:</strong> {{ dataset_name }} · <strong>Target:</strong> {{ target_column }}</p>
        </div>
        <div>
          <p><strong>Model:</strong> {{ model_name }}</p>
          <p><strong>Task:</strong> {{ task_type }}</p>
          <p><strong>Decision:</strong> {{ deployment_decision }}</p>
        </div>
      </div>
    </div>

    <div class=\"glass\">
      <h2>1) Analysis Panel</h2>
      <div class=\"kpi-grid\">
        <div class=\"kpi\"><div class=\"label\">Rows</div><div class=\"val\">{{ analysis.row_count or 0 }}</div></div>
        <div class=\"kpi\"><div class=\"label\">Features</div><div class=\"val\">{{ analysis.feature_count or 0 }}</div></div>
        <div class=\"kpi\"><div class=\"label\">Missing rows %</div><div class=\"val\">{{ analysis.data_quality.missing_rows_pct if analysis.data_quality else 0 }}</div></div>
        <div class=\"kpi\"><div class=\"label\">Duplicate rows</div><div class=\"val\">{{ analysis.data_quality.duplicate_rows if analysis.data_quality else 0 }}</div></div>
      </div>
      {% if analysis.quality_flags %}
        <div style=\"margin-top:10px;\">
          {% for flag in analysis.quality_flags %}
            <div class=\"flag {{ flag.severity }}\">{{ flag.message }}</div>
          {% endfor %}
        </div>
      {% endif %}
      {% if charts.correlation_heatmap %}
        <div style=\"margin-top:12px;\"><img class=\"img\" src=\"{{ charts.correlation_heatmap }}\" alt=\"correlation heatmap\"/></div>
      {% endif %}
      {% if analysis.analysis_summary %}<p class=\"muted\" style=\"margin-top:10px;\">{{ analysis.analysis_summary }}</p>{% endif %}
    </div>

    <div class=\"glass\">
      <h2>2) Preprocessing + Feature Engineering Panels</h2>
      <div class=\"row\">
        <div>
          <h3>Preprocessing</h3>
          <p class=\"muted\">Train rows: {{ preprocessing.train_size or 0 }} · Test rows: {{ preprocessing.test_size or 0 }}</p>
          <p class=\"muted\">Raw features: {{ preprocessing.raw_feature_count or 0 }} → Transformed: {{ preprocessing.transformed_feature_count or preprocessing.feature_count or 0 }}</p>
          {% if preprocessing.explanation %}<p>{{ preprocessing.explanation }}</p>{% endif %}
        </div>
        <div>
          <h3>Feature engineering</h3>
          <p class=\"muted\">Final feature count: {{ features.final_feature_count or 0 }}</p>
          {% if features.selected_features %}
            <div>
              {% for feat in features.selected_features[:20] %}
                <span class=\"pill\">{{ feat }}</span>
              {% endfor %}
            </div>
          {% endif %}
        </div>
      </div>
      {% if charts.feature_importance %}
        <div style=\"margin-top:12px;\"><img class=\"img\" src=\"{{ charts.feature_importance }}\" alt=\"feature importance\"/></div>
      {% endif %}
    </div>

    <div class=\"glass\">
      <h2>3) Model Selection Panel</h2>
      {% if model_selection.top_candidates %}
        <table class=\"table\">
          <thead><tr><th>#</th><th>Model</th><th>Family</th><th>Reasoning</th></tr></thead>
          <tbody>
            {% for cand in model_selection.top_candidates[:5] %}
              <tr>
                <td>{{ loop.index }}</td>
                <td>{{ cand.model_name }}</td>
                <td>{{ cand.model_family }}</td>
                <td>{{ cand.reasoning }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <p class=\"muted\">No model selection candidates available.</p>
      {% endif %}
      {% if model_selection.selection_reasoning %}<p style=\"margin-top:10px;\">{{ model_selection.selection_reasoning }}</p>{% endif %}
    </div>

    <div class=\"glass\">
      <h2>4) Training Panel</h2>
      <p class=\"muted\">Best CV score: {{ training.best_score if training.best_score is not none else 'n/a' }} · Training time: {{ training.training_time if training.training_time is not none else 'n/a' }}s</p>
      {% if charts.cv_scores %}
        <div style=\"margin-top:12px;\"><img class=\"img\" src=\"{{ charts.cv_scores }}\" alt=\"cv scores\"/></div>
      {% endif %}
      {% if training.best_params %}
        <h3 style=\"margin-top:12px;\">Best parameters</h3>
        <table class=\"table\">
          <tbody>
            {% for k, v in training.best_params.items() %}
              <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
      {% endif %}
    </div>

    <div class=\"glass\">
      <h2>5) Evaluation Panel</h2>
      {% if task_type == 'regression' %}
        <div class=\"kpi-grid\">
          <div class=\"kpi\"><div class=\"label\">R²</div><div class=\"val\">{{ evaluation.r2 if evaluation.r2 is not none else 'n/a' }}</div></div>
          <div class=\"kpi\"><div class=\"label\">RMSE</div><div class=\"val\">{{ evaluation.rmse if evaluation.rmse is not none else 'n/a' }}</div></div>
          <div class=\"kpi\"><div class=\"label\">MAE</div><div class=\"val\">{{ evaluation.mae if evaluation.mae is not none else 'n/a' }}</div></div>
          <div class=\"kpi\"><div class=\"label\">Test score</div><div class=\"val\">{{ training.test_score if training.test_score is not none else 'n/a' }}</div></div>
        </div>
      {% else %}
        <div class=\"kpi-grid\">
          <div class=\"kpi\"><div class=\"label\">Accuracy</div><div class=\"val\">{{ evaluation.accuracy if evaluation.accuracy is not none else 'n/a' }}</div></div>
          <div class=\"kpi\"><div class=\"label\">F1</div><div class=\"val\">{{ evaluation.f1 if evaluation.f1 is not none else 'n/a' }}</div></div>
          <div class=\"kpi\"><div class=\"label\">Precision</div><div class=\"val\">{{ evaluation.precision if evaluation.precision is not none else 'n/a' }}</div></div>
          <div class=\"kpi\"><div class=\"label\">Recall</div><div class=\"val\">{{ evaluation.recall if evaluation.recall is not none else 'n/a' }}</div></div>
        </div>
      {% endif %}
      {% if charts.confusion_matrix and task_type != 'regression' %}
        <div style=\"margin-top:12px;\"><img class=\"img\" src=\"{{ charts.confusion_matrix }}\" alt=\"confusion matrix\"/></div>
      {% endif %}
      {% if charts.actual_vs_predicted and task_type == 'regression' %}
        <div style=\"margin-top:12px;\"><img class=\"img\" src=\"{{ charts.actual_vs_predicted }}\" alt=\"actual vs predicted\"/></div>
      {% endif %}
    </div>

    {% if insights %}
    <div class=\"glass\">
      <h2>6) Narrative Panel (LLM)</h2>
      {% if insights.stage_summary %}<p><strong>Summary:</strong> {{ insights.stage_summary }}</p>{% endif %}
      {% if insights.performance_story %}<p><strong>Performance story:</strong> {{ insights.performance_story }}</p>{% endif %}
      {% if insights.cross_validation_explanation %}<p><strong>CV explanation:</strong> {{ insights.cross_validation_explanation }}</p>{% endif %}
      {% if insights.deployment_reasoning %}
        <p><strong>Deployment reasoning:</strong> {{ insights.deployment_reasoning.reason }}</p>
        <p class=\"muted\">Confidence: {{ insights.deployment_reasoning.confidence }} · Risk: {{ insights.deployment_reasoning.risk_note }}</p>
      {% endif %}
    </div>
    {% endif %}
  </div>
</body>
</html>
"""
