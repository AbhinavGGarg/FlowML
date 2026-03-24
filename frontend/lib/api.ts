export type PipelineStageStatus = "waiting" | "running" | "completed" | "failed";
export type TaskType = "classification" | "regression";

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
}

export interface DatasetSummary {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  column_types: Record<string, string>;
  missing_values: Record<string, number>;
  numeric_summary?: Record<string, Record<string, number>> | null;
}

export interface DatasetColumn {
  name: string;
  dtype: string;
  is_numeric: boolean;
  missing_pct: number;
}

export interface DatasetPreviewResponse {
  rows: Array<Record<string, unknown>>;
  columns: string[];
}

export interface ChatMessagePayload {
  role: "user" | "assistant";
  content: string;
}

export interface ChatSelectionContext {
  text: string;
  source_label?: string | null;
  surrounding_text?: string | null;
}

export interface ChatResponse {
  answer: string;
  llm_used: boolean;
  response_mode?: "llm" | "unavailable" | "structured" | null;
  revision?: Record<string, unknown> | null;
}

export interface RevisionStateResponse {
  run_id: string | null;
  parent_run_id: string | null;
  dataset: Record<string, unknown>;
  target_column: string | null;
  task_type: TaskType;
  stage_configs: Record<string, Record<string, unknown>>;
  stage_outputs: Record<string, unknown>;
  metrics: Record<string, unknown>;
  revision_reason: string | null;
  changed_stages: string[];
  created_at: string | null;
}

export interface PipelineStatusResponse {
  pipeline_id: string | null;
  stages: Record<string, PipelineStageStatus>;
  target_column: string | null;
  dataset_loaded: boolean;
}

export interface PipelineLogsResponse {
  logs: Record<string, string[]>;
}

export interface PipelineConfig {
  task_type: TaskType;
  test_size: number;
  random_state: number;
}

export interface StageResultResponse {
  stage_id: string;
  status: PipelineStageStatus;
  result: Record<string, unknown> | null;
}

export interface BaselineMetrics {
  strategy?: string | null;
  label?: string | null;
  value?: number | null;
  accuracy?: number | null;
  f1?: number | null;
  roc_auc?: number | null;
  r2?: number | null;
  mae?: number | null;
  mse?: number | null;
  rmse?: number | null;
}

export interface MetricsResponse {
  task_type: TaskType;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number | null;
  r2?: number | null;
  mae?: number | null;
  mse?: number | null;
  rmse?: number | null;
  best_score: number;
  cv_scores?: number[];
  cv_std?: number | null;
  train_score?: number | null;
  test_score?: number | null;
  model_name?: string | null;
  deployment_decision?: string | null;
  performance_summary?: string | null;
  confusion_matrix: number[][];
  baseline_metrics?: BaselineMetrics | null;
}

export type ExplanationResponse = Record<string, unknown>;

export type DeploymentRecommendation = "deploy" | "review" | "do_not_deploy";
export type DeploymentConfidence = "high" | "medium" | "low";

export interface EvaluationInsightsResponse {
  stage_summary: string;
  about_stage_text: string;
  performance_story: string;
  loss_explanation: string;
  generalization_explanation: string;
  cross_validation_explanation: string;
  baseline_explanation: string;
  deployment_reasoning: {
    recommendation: DeploymentRecommendation;
    confidence: DeploymentConfidence;
    reason: string;
    risk_note: string;
    next_step: string;
  };
  metric_tooltips: {
    r2: string;
    rmse: string;
    mae: string;
    accuracy: string;
    f1: string;
    roc_auc: string;
  };
  chart_explanations: {
    primary_chart: string;
    secondary_chart: string;
  };
  beginner_notes: string[];
  learning_questions: string[];
  source: "openrouter" | "fallback";
  llm_used: boolean;
  model: string;
  error?: string | null;
}

export interface DataQualityProfile {
  total_missing_cells?: number | null;
  missing_rows_pct?: number | null;
  missing_columns_count?: number | null;
  high_missing_columns?: string[] | null;
  duplicate_rows?: number | null;
  duplicate_pct?: number | null;
  outlier_columns_count?: number | null;
  max_outlier_pct?: number | null;
  high_cardinality_columns?: string[] | null;
  high_cardinality_count?: number | null;
  placeholder_invalid_count?: number | null;
  placeholder_invalid_columns?: string[] | null;
  leakage_risk_columns?: string[] | null;
}

export interface QualityFlag {
  severity: "high" | "medium" | "low";
  message: string;
  field: string;
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://127.0.0.1:8000";
const CHAT_REQUEST_TIMEOUT_MS = 210000;
const CHAT_APPLY_TIMEOUT_MS = 300000;

class ApiTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiTimeoutError";
  }
}

async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  options?: {
    timeoutMs?: number;
  },
): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = options?.timeoutMs;
  const timeoutId =
    typeof timeoutMs === "number" && timeoutMs > 0
      ? window.setTimeout(() => controller.abort(), timeoutMs)
      : null;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiTimeoutError("The chat request took too long and timed out.");
    }
    throw error;
  }

  if (timeoutId !== null) {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.message || detail;
    } catch {
      // Ignore JSON parse errors and keep the fallback message.
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiRequest<DatasetUploadResponse>("/api/dataset/upload", {
    method: "POST",
    body: formData,
  });
}

export function getDatasetSummary(): Promise<DatasetSummary> {
  return apiRequest<DatasetSummary>("/api/dataset/summary");
}

export function getDatasetColumns(): Promise<{ columns: DatasetColumn[] }> {
  return apiRequest<{ columns: DatasetColumn[] }>("/api/dataset/columns");
}

export function getDatasetPreview(rows = 5): Promise<DatasetPreviewResponse> {
  return apiRequest<DatasetPreviewResponse>(`/api/dataset/preview?n=${rows}`);
}

export function setTargetColumn(targetColumn: string): Promise<{ target_column: string }> {
  return apiRequest<{ target_column: string }>("/api/dataset/target", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ target_column: targetColumn }),
  });
}

export function getPipelineStatus(): Promise<PipelineStatusResponse> {
  return apiRequest<PipelineStatusResponse>("/api/pipeline/status");
}

export function getPipelineLogs(): Promise<PipelineLogsResponse> {
  return apiRequest<PipelineLogsResponse>("/api/pipeline/logs");
}

export function runPipelineStage(stageId: string, config: PipelineConfig): Promise<StageResultResponse> {
  return apiRequest<StageResultResponse>(`/api/pipeline/stage/${stageId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });
}

export function getStageResult(stageId: string): Promise<StageResultResponse> {
  return apiRequest<StageResultResponse>(`/api/stages/${stageId}/results`);
}

export function getMetrics(): Promise<MetricsResponse> {
  return apiRequest<MetricsResponse>("/api/results/metrics");
}

export function getEvaluationInsights(): Promise<EvaluationInsightsResponse> {
  return apiRequest<EvaluationInsightsResponse>("/api/results/evaluation-insights");
}

export function getExplanation(): Promise<ExplanationResponse> {
  return apiRequest<ExplanationResponse>("/api/results/explanation");
}

export function queryChat(
  question: string,
  history: ChatMessagePayload[],
  selectionContext?: ChatSelectionContext | null,
  mode: "suggest" | "apply" = "suggest",
): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/api/chat/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      history,
      selection_context: selectionContext ?? null,
      mode,
    }),
  }, {
    timeoutMs: mode === "apply" ? CHAT_APPLY_TIMEOUT_MS : CHAT_REQUEST_TIMEOUT_MS,
  });
}

export function getCurrentRevisionState(): Promise<RevisionStateResponse> {
  return apiRequest<RevisionStateResponse>("/api/revisions/current");
}

export function getDownloadUrl(kind: "model" | "logs" | "deployment-package" | "report"): string {
  return `${API_BASE_URL}/api/results/download/${kind}`;
}
