import type {
  BaselineMetrics,
  EvaluationInsightsResponse,
  MetricsResponse,
  TaskType,
} from "@/lib/api";

export interface EvaluationStageResultLike extends Record<string, unknown> {
  task_type?: TaskType;
  roc_auc?: number | null;
  predictions?: unknown[];
  y_test?: unknown[];
  residuals?: number[];
  absolute_errors?: number[];
  prediction_confidence?: number[];
  confusion_matrix?: number[][];
  classification_report?: Record<string, Record<string, number>>;
  class_labels?: Array<string | number>;
  baseline_metrics?: BaselineMetrics | null;
  llm_insights?: EvaluationInsightsResponse;
  llm_insights_path?: string | null;
}

export interface LossStageResultLike extends Record<string, unknown> {
  train_loss?: number[];
  val_loss?: number[];
  best_epoch?: number | null;
}

export interface MetricHighlightCard {
  key: string;
  label: string;
  value: string;
  rawValue: number | null;
  badge: string;
  tone: string;
  statement: string;
  tooltip: string;
}

export interface EvaluationTabOption<T extends string> {
  id: T;
  label: string;
}

export interface RegressionScatterPoint {
  actual: number;
  predicted: number;
  residual: number;
}

export interface HistogramBin {
  label: string;
  value: number;
}

export interface ClassMetricBar {
  label: string;
  precision: number;
  recall: number;
  f1: number;
}

export interface LossCurvePoint {
  epoch: string;
  trainLoss: number;
  valLoss: number;
}

export const NOTES_TABS: EvaluationTabOption<"notes" | "technical">[] = [
  { id: "notes", label: "Notes" },
  { id: "technical", label: "Technical Logs" },
];

export const CODE_TABS: EvaluationTabOption<"simple" | "pseudocode" | "real">[] = [
  { id: "simple", label: "Simple explanation" },
  { id: "pseudocode", label: "Pseudocode" },
  { id: "real", label: "Real code" },
];

export function resolveTaskType(
  stageResult: EvaluationStageResultLike | null,
  metrics: MetricsResponse | null,
  fallbackTaskType: TaskType,
): TaskType {
  const candidate = stageResult?.task_type ?? metrics?.task_type ?? fallbackTaskType;
  return candidate === "regression" ? "regression" : "classification";
}

export function getEvaluationSubtitle(taskType: TaskType): string {
  return taskType === "regression"
    ? "Measure how close predicted values are to the real values on unseen data."
    : "Measure how well the model predicts classes on unseen data.";
}

export function getFallbackInsights(taskType: TaskType): EvaluationInsightsResponse {
  return {
    stage_summary:
      taskType === "regression"
        ? "This evaluation run shows how close the model's numeric predictions were on unseen data."
        : "This evaluation run shows how well the model classified unseen examples.",
    about_stage_text:
      taskType === "regression"
        ? "Evaluation checks whether predicted values stay close to the true values on data the model did not train on."
        : "Evaluation checks how often the model predicts the right class and whether those predictions stay balanced across classes.",
    performance_story:
      taskType === "regression"
        ? "Use R2 to judge explained variation, and use RMSE or MAE to judge the size of the mistakes."
        : "Use accuracy for overall correctness, then look at F1 and ROC-AUC for a fuller picture of model quality.",
    loss_explanation: "The loss curves show how training and validation behavior changed over time and whether the model started to overfit.",
    generalization_explanation: "Train and test performance help show whether the model learned a real pattern or just memorized the training set.",
    cross_validation_explanation: "Cross-validation checks whether the model stays reliable across several training and validation splits.",
    baseline_explanation: "A baseline comparison shows whether the model learned more than the simplest reasonable guess.",
    deployment_reasoning: {
      recommendation: "review",
      confidence: "medium",
      reason: "The model should be reviewed with its metrics, baseline, and generalization checks together.",
      risk_note: "A single metric rarely tells the whole story.",
      next_step: "Inspect the weakest metrics and the most common errors before deployment.",
    },
    metric_tooltips: {
      r2: "R2 shows how much of the target variation the model explains.",
      rmse: "RMSE shows the typical error size, with larger mistakes counting more.",
      mae: "MAE shows the average absolute prediction error.",
      accuracy: "Accuracy is the share of predictions that were correct.",
      f1: "F1 balances precision and recall.",
      roc_auc: "ROC-AUC measures how well the model separates classes across thresholds.",
    },
    chart_explanations: {
      primary_chart:
        taskType === "regression"
          ? "This chart helps you see whether predictions track the real values."
          : "This chart helps you see which classes are predicted well and where mistakes cluster.",
      secondary_chart:
        taskType === "regression"
          ? "This chart helps you see whether the errors are mostly small or sometimes large."
          : "This chart helps you see whether performance is balanced across classes.",
    },
    beginner_notes: [
      "This run shows what happened on unseen data, not just on the training set.",
      "The strongest sign is when several metrics point in the same direction instead of only one looking good.",
      "The main thing to watch is whether the model stays stable across validation, test data, and loss curves.",
    ],
    learning_questions: [
      "Why do we evaluate on unseen data?",
      "What does a small train-test gap suggest?",
      "Why is a baseline useful?",
    ],
    source: "fallback",
    llm_used: false,
    model: "fallback",
    error: null,
  };
}

export function getUnavailableInsights(taskType: TaskType, errorMessage?: string | null): EvaluationInsightsResponse {
  const detail = errorMessage?.trim() || "Loading Responses";
  return {
    stage_summary: "Loading Responses",
    about_stage_text: "",
    performance_story: "Loading Responses",
    loss_explanation: "Loading Responses",
    generalization_explanation: "Loading Responses",
    cross_validation_explanation: "Loading Responses",
    baseline_explanation: "Loading Responses",
    deployment_reasoning: {
      recommendation: "review",
      confidence: "low",
      reason: "Loading Responses",
      risk_note: "Loading Responses",
      next_step: "Loading Responses",
    },
    metric_tooltips: {
      r2: "",
      rmse: "",
      mae: "",
      accuracy: "",
      f1: "",
      roc_auc: "",
    },
    chart_explanations: {
      primary_chart: "Loading Responses",
      secondary_chart: "Loading Responses",
    },
    beginner_notes: [
      "Loading Responses",
      "Loading Responses",
      "Loading Responses",
    ],
    learning_questions: [],
    source: "fallback",
    llm_used: false,
    model: "openrouter-required",
    error: detail,
  };
}

export function buildMetricHighlightCards(args: {
  taskType: TaskType;
  metrics: MetricsResponse | null;
  stageResult: EvaluationStageResultLike | null;
  insights: EvaluationInsightsResponse;
  targetColumn: string | null;
}): MetricHighlightCard[] {
  const { taskType, metrics, stageResult, insights, targetColumn } = args;
  const baseline = stageResult?.baseline_metrics ?? metrics?.baseline_metrics ?? null;

  if (taskType === "regression") {
    return [
      makeMetricCard("r2", "R2", metrics?.r2 ?? null, taskType, baseline, insights.metric_tooltips.r2, targetColumn),
      makeMetricCard("rmse", "RMSE", metrics?.rmse ?? null, taskType, baseline, insights.metric_tooltips.rmse, targetColumn),
      makeMetricCard("mae", "MAE", metrics?.mae ?? null, taskType, baseline, insights.metric_tooltips.mae, targetColumn),
    ];
  }

  return [
    makeMetricCard("accuracy", "Accuracy", metrics?.accuracy ?? null, taskType, baseline, insights.metric_tooltips.accuracy, targetColumn),
    makeMetricCard("f1", "F1", metrics?.f1 ?? null, taskType, baseline, insights.metric_tooltips.f1, targetColumn),
    makeMetricCard("roc_auc", "ROC-AUC", metrics?.roc_auc ?? null, taskType, baseline, insights.metric_tooltips.roc_auc, targetColumn),
  ];
}

export function buildGeneralizationSummary(metrics: MetricsResponse | null): {
  trainLabel: string;
  testLabel: string;
  gapLabel: string;
} {
  const train = metrics?.train_score ?? null;
  const test = metrics?.test_score ?? null;
  const gap = typeof train === "number" && typeof test === "number" ? Math.abs(train - test) : null;

  return {
    trainLabel: formatMetricValue(train, false),
    testLabel: formatMetricValue(test, false),
    gapLabel: typeof gap === "number" ? gap.toFixed(3) : "Pending",
  };
}

export function buildCrossValidationData(metrics: MetricsResponse | null): Array<{ fold: string; score: number }> {
  return (metrics?.cv_scores ?? []).map((score, index) => ({
    fold: `Fold ${index + 1}`,
    score,
  }));
}

export function buildBaselineSummary(args: {
  taskType: TaskType;
  metrics: MetricsResponse | null;
  stageResult: EvaluationStageResultLike | null;
}): {
  label: string;
  baselineValue: string;
  modelValue: string;
  improvement: string;
} {
  const { taskType, metrics, stageResult } = args;
  const baseline = stageResult?.baseline_metrics ?? metrics?.baseline_metrics ?? null;

  if (taskType === "regression") {
    const baselineRmse = baseline?.rmse ?? null;
    const modelRmse = metrics?.rmse ?? null;
    const improvement =
      typeof baselineRmse === "number" && typeof modelRmse === "number" && baselineRmse !== 0
        ? `${(((baselineRmse - modelRmse) / baselineRmse) * 100).toFixed(1)}% better`
        : "Pending";
    return {
      label: "RMSE vs mean-target baseline",
      baselineValue: formatMetricValue(baselineRmse, false),
      modelValue: formatMetricValue(modelRmse, false),
      improvement,
    };
  }

  const baselineAccuracy = baseline?.accuracy ?? null;
  const modelAccuracy = metrics?.accuracy ?? null;
  const improvement =
    typeof baselineAccuracy === "number" && typeof modelAccuracy === "number"
      ? `${((modelAccuracy - baselineAccuracy) * 100).toFixed(1)} pts`
      : "Pending";
  return {
    label: "Accuracy vs majority-class baseline",
    baselineValue: formatMetricValue(baselineAccuracy, true),
    modelValue: formatMetricValue(modelAccuracy, true),
    improvement,
  };
}

export function buildRegressionScatterData(stageResult: EvaluationStageResultLike | null, limit = 140): RegressionScatterPoint[] {
  const actual = readNumericArray(stageResult?.y_test);
  const predicted = readNumericArray(stageResult?.predictions);
  const points = actual
    .map((value, index) => ({
      actual: value,
      predicted: predicted[index],
      residual: value - predicted[index],
    }))
    .filter((point) => Number.isFinite(point.actual) && Number.isFinite(point.predicted));

  return sampleArray(points, limit);
}

export function buildResidualHistogram(stageResult: EvaluationStageResultLike | null, binCount = 8): HistogramBin[] {
  const residuals = readNumericArray(stageResult?.residuals ?? stageResult?.absolute_errors);
  return buildHistogram(residuals, binCount);
}

export function buildConfidenceHistogram(stageResult: EvaluationStageResultLike | null, binCount = 6): HistogramBin[] {
  const confidence = readNumericArray(stageResult?.prediction_confidence);
  return buildHistogram(confidence, binCount, 0, 1);
}

export function buildClassMetricBars(stageResult: EvaluationStageResultLike | null): ClassMetricBar[] {
  const report = stageResult?.classification_report;
  if (!report || typeof report !== "object") return [];

  return Object.entries(report)
    .filter(([label, value]) => {
      return !["accuracy", "macro avg", "weighted avg"].includes(label) && Boolean(value && typeof value === "object");
    })
    .map(([label, value]) => {
      const metrics = value as Record<string, number>;
      return {
        label,
        precision: Number(metrics.precision ?? 0),
        recall: Number(metrics.recall ?? 0),
        f1: Number(metrics["f1-score"] ?? 0),
      };
    })
    .slice(0, 8);
}

export function buildLossCurveData(lossStageResult: LossStageResultLike | null): LossCurvePoint[] {
  const trainLoss = readNumericArray(lossStageResult?.train_loss);
  const valLoss = readNumericArray(lossStageResult?.val_loss);
  const length = Math.min(trainLoss.length, valLoss.length);
  if (length === 0) return [];

  return Array.from({ length }, (_, index) => ({
    epoch: `Epoch ${index + 1}`,
    trainLoss: trainLoss[index],
    valLoss: valLoss[index],
  }));
}

export function buildLossSummary(lossStageResult: LossStageResultLike | null): {
  bestEpoch: string;
  finalTrainLoss: string;
  finalValLoss: string;
} {
  const trainLoss = readNumericArray(lossStageResult?.train_loss);
  const valLoss = readNumericArray(lossStageResult?.val_loss);
  const bestEpoch =
    typeof lossStageResult?.best_epoch === "number"
      ? String(lossStageResult.best_epoch + 1)
      : "Pending";

  return {
    bestEpoch,
    finalTrainLoss: trainLoss.length > 0 ? trainLoss[trainLoss.length - 1].toFixed(3) : "Pending",
    finalValLoss: valLoss.length > 0 ? valLoss[valLoss.length - 1].toFixed(3) : "Pending",
  };
}

export function getSimpleExplanation(taskType: TaskType): string {
  return taskType === "regression"
    ? "We let the model make predictions on new examples, then we measured how close those predicted numbers were to the true numbers."
    : "We let the model predict the class for new examples, then we measured how often it was right and what kinds of mistakes it made.";
}

export function getPseudocode(taskType: TaskType): string {
  if (taskType === "regression") {
    return `split data into train and test
train the model on the training set
predict numeric values for X_test
measure R2, MAE, MSE, and RMSE
compare train score, test score, CV, and baseline
decide whether the model is reliable enough to deploy`;
  }

  return `split data into train and test
train the model on the training set
predict classes for X_test
measure accuracy, precision, recall, F1, and ROC-AUC
inspect the confusion matrix, CV scores, and baseline
decide whether the model is reliable enough to deploy`;
}

export function getRealCode(taskType: TaskType): string {
  if (taskType === "regression") {
    return `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print({"r2": r2, "mae": mae, "mse": mse, "rmse": rmse})`;
  }

  return `from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
matrix = confusion_matrix(y_test, y_pred)

if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_score[:, 1])

print({"accuracy": accuracy, "f1": f1, "confusion_matrix": matrix})`;
}

export function formatMetricValue(value: number | null | undefined, asPercent: boolean): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "Pending";
  return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
}

function makeMetricCard(
  key: string,
  label: string,
  value: number | null,
  taskType: TaskType,
  baseline: BaselineMetrics | null | undefined,
  tooltip: string,
  targetColumn: string | null,
): MetricHighlightCard {
  const { badge, tone } = getMetricBadge(key, value, taskType, baseline);
  return {
    key,
    label,
    value: formatMetricValue(value, isPercentMetric(key)),
    rawValue: value,
    badge,
    tone,
    statement: buildMetricStatement(key, value, targetColumn),
    tooltip,
  };
}

function getMetricBadge(
  metricKey: string,
  value: number | null,
  taskType: TaskType,
  baseline: BaselineMetrics | null | undefined,
): { badge: string; tone: string } {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return {
      badge: "Pending",
      tone: "border-border/60 bg-background/40 text-muted-foreground",
    };
  }

  let score = value;
  if (taskType === "regression" && (metricKey === "rmse" || metricKey === "mae")) {
    const baselineValue = metricKey === "rmse" ? baseline?.rmse : baseline?.mae;
    if (typeof baselineValue === "number" && baselineValue > 0) {
      score = 1 - Math.min(value / baselineValue, 1.25);
    } else {
      score = 1 - Math.min(value, 1.25);
    }
  }

  const normalized =
    metricKey === "rmse" || metricKey === "mae"
      ? score
      : metricKey === "r2"
        ? value
        : value;

  if (normalized >= 0.9 || (metricKey === "rmse" || metricKey === "mae") && normalized >= 0.6) {
    return { badge: "Excellent", tone: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200" };
  }
  if (normalized >= 0.8 || (metricKey === "rmse" || metricKey === "mae") && normalized >= 0.35) {
    return { badge: "Strong", tone: "border-cyan-400/30 bg-cyan-400/10 text-cyan-100" };
  }
  if (normalized >= 0.65 || (metricKey === "rmse" || metricKey === "mae") && normalized >= 0.15) {
    return { badge: "Moderate", tone: "border-amber-300/30 bg-amber-300/10 text-amber-100" };
  }
  return { badge: "Needs improvement", tone: "border-rose-400/30 bg-rose-400/10 text-rose-200" };
}

function buildMetricStatement(metricKey: string, value: number | null, targetColumn: string | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "This value is not available for the current run yet.";
  const target = targetColumn || "the target";

  switch (metricKey) {
    case "r2":
      return `The model explains ${(value * 100).toFixed(1)}% of the variation in ${target}.`;
    case "rmse":
      return `Predictions are typically off by about ${value.toFixed(3)} ${target === "the target" ? "units" : ""}.`.trim();
    case "mae":
      return `The average absolute prediction error is about ${value.toFixed(3)}.`;
    case "accuracy":
      return `The model gets the correct class about ${(value * 100).toFixed(1)}% of the time.`;
    case "f1":
      return `This F1 score suggests how well the model balances precision and recall across classes.`;
    case "roc_auc":
      return `The model's class-separation ability is ${(value * 100).toFixed(1)}% on the ROC-AUC scale.`;
    default:
      return "This metric helps summarize how the model performed on unseen data.";
  }
}

function buildHistogram(values: number[], binCount: number, minOverride?: number, maxOverride?: number): HistogramBin[] {
  if (values.length === 0) return [];
  const min = typeof minOverride === "number" ? minOverride : Math.min(...values);
  const max = typeof maxOverride === "number" ? maxOverride : Math.max(...values);
  if (min === max) {
    return [{ label: min.toFixed(2), value: values.length }];
  }

  const step = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    label: `${(min + step * index).toFixed(2)}-${(min + step * (index + 1)).toFixed(2)}`,
    value: 0,
  }));

  values.forEach((value) => {
    const normalized = Math.min(Math.floor((value - min) / step), binCount - 1);
    bins[Math.max(normalized, 0)].value += 1;
  });

  return bins;
}

function readNumericArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => Number(entry))
    .filter((entry) => Number.isFinite(entry));
}

function sampleArray<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const stride = items.length / limit;
  const sampled: T[] = [];
  for (let index = 0; index < limit; index += 1) {
    sampled.push(items[Math.floor(index * stride)]);
  }
  return sampled;
}

function isPercentMetric(metricKey: string): boolean {
  return ["accuracy", "f1", "roc_auc"].includes(metricKey);
}
