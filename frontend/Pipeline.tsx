import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Brain, Database, Gauge, Play, Target } from "lucide-react";
import { stages, type PipelineStage, type StageStatus } from "@/data/pipelineStages";
import {
  executePipeline,
  getDatasetColumns,
  getDatasetPreview,
  getDatasetSummary,
  getExplanation,
  getMetrics,
  getPipelineLogs,
  getPipelineStatus,
  getStageResult,
  setTargetColumn,
  uploadDataset,
  type DatasetColumn,
  type DatasetPreviewResponse,
  type DatasetSummary,
  type MetricsResponse,
  type PipelineConfig,
  type PipelineStatusResponse,
  type TaskType,
} from "@/lib/api";
import ChatBot from "./ChatBot";
import DatasetUpload from "./DatasetUpload";
import NeuralBackground from "./NeuralBackground";
import PredictionTarget from "./PredictionTarget";
import ResultsPanel from "./ResultsPanel";
import StageDetailPanel from "./StageDetailPanel";
import WaterLogo from "./components/WaterLogo";

const HIDDEN_STAGE_IDS = new Set(["loss"]);
const VISIBLE_STAGES = stages.filter((stage) => !HIDDEN_STAGE_IDS.has(stage.id));
const VISIBLE_STAGE_ORDER = VISIBLE_STAGES.map((stage) => stage.id);
const EXECUTION_STAGE_ORDER = stages.map((stage) => stage.id);
const DEFAULT_CONFIG = {
  test_size: 0.2,
  random_state: 42,
} satisfies Omit<PipelineConfig, "task_type">;
const CANONICAL_TO_EXECUTION_STAGE: Record<string, string> = {
  analysis: "analysis",
  preprocessing: "preprocessing",
  feature_engineering: "features",
  training: "training",
  evaluation: "evaluation",
  explainability: "results",
};

const createInitialStatuses = (): Record<string, StageStatus> =>
  Object.fromEntries(EXECUTION_STAGE_ORDER.map((stageId) => [stageId, "waiting"])) as Record<string, StageStatus>;

const createEmptyLogs = (): Record<string, string[]> =>
  Object.fromEntries(EXECUTION_STAGE_ORDER.map((stageId) => [stageId, []]));

const Pipeline = () => {
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [datasetColumns, setDatasetColumns] = useState<DatasetColumn[]>([]);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewResponse | null>(null);
  const [isLoadingDatasetPreview, setIsLoadingDatasetPreview] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState<Record<string, StageStatus>>(createInitialStatuses);
  const [stageLogs, setStageLogs] = useState<Record<string, string[]>>(createEmptyLogs);
  const [stageResults, setStageResults] = useState<Record<string, Record<string, unknown>>>({});
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [explanation, setExplanation] = useState<Record<string, unknown> | null>(null);
  const [selectedColumn, setSelectedColumnState] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [isUploading, setIsUploading] = useState(false);
  const [isSavingTarget, setIsSavingTarget] = useState(false);
  const [isRunningPipeline, setIsRunningPipeline] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const syncStageResults = useCallback(async (statuses: Record<string, StageStatus>) => {
    const completedStages = EXECUTION_STAGE_ORDER.filter((stageId) => statuses[stageId] === "completed");
    if (completedStages.length === 0) {
      setStageResults({});
      return;
    }

    const responses = await Promise.all(
      completedStages.map(async (stageId) => {
        try {
          const response = await getStageResult(stageId);
          return [stageId, response.result || {}] as const;
        } catch {
          return [stageId, {}] as const;
        }
      }),
    );

    setStageResults(Object.fromEntries(responses));
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      const logs = await getPipelineLogs();
      setStageLogs({ ...createEmptyLogs(), ...logs.logs });
    } catch {
      setStageLogs(createEmptyLogs());
    }
  }, []);

  const refreshPipelineData = useCallback(async () => {
    let statusResponse: PipelineStatusResponse;

    try {
      statusResponse = await getPipelineStatus();
    } catch (err) {
      throw err instanceof Error ? err : new Error("Unable to reach the backend API.");
    }

    const normalizedStages = { ...createInitialStatuses(), ...statusResponse.stages };
    setPipelineStatus(normalizedStages);
    setSelectedColumnState(statusResponse.target_column);

    if (!statusResponse.dataset_loaded) {
      setDatasetSummary(null);
      setDatasetColumns([]);
      setDatasetPreview(null);
      setIsLoadingDatasetPreview(false);
      setStageLogs(createEmptyLogs());
      setStageResults({});
      setMetrics(null);
      setExplanation(null);
      return;
    }

    let latestColumns: DatasetColumn[] = [];
    setIsLoadingDatasetPreview(true);
    try {
      const [summary, columnResponse, previewResponse] = await Promise.all([
        getDatasetSummary(),
        getDatasetColumns(),
        getDatasetPreview(5).catch(() => null),
      ]);
      latestColumns = columnResponse.columns;
      setDatasetSummary(summary);
      setDatasetColumns(latestColumns);
      setDatasetPreview(previewResponse);
    } finally {
      setIsLoadingDatasetPreview(false);
    }

    if (statusResponse.target_column) {
      setTaskType(inferTaskType(latestColumns, statusResponse.target_column));
    }

    await refreshLogs();
    await syncStageResults(normalizedStages);

    if (normalizedStages.evaluation === "completed" || normalizedStages.results === "completed") {
      try {
        setMetrics(await getMetrics());
      } catch {
        setMetrics(null);
      }
    } else {
      setMetrics(null);
    }

    if (normalizedStages.results === "completed") {
      try {
        setExplanation(await getExplanation());
      } catch {
        // Keep any existing explanation; retry via effect.
      }
    } else {
      setExplanation(null);
    }
  }, [refreshLogs, syncStageResults]);

  const getRerunStages = useCallback((rerunFromStage: string) => {
    const concreteStage = CANONICAL_TO_EXECUTION_STAGE[rerunFromStage] || rerunFromStage;
    const startIndex = EXECUTION_STAGE_ORDER.indexOf(concreteStage);
    if (startIndex < 0) return [];
    return EXECUTION_STAGE_ORDER.slice(startIndex);
  }, []);

  const handleRevisionRerunStart = useCallback((rerunFromStage: string) => {
    const rerunStages = getRerunStages(rerunFromStage);
    if (rerunStages.length === 0) return;

    const [firstStage, ...downstreamStages] = rerunStages;
    const rerunStageSet = new Set(rerunStages);

    setIsRunningPipeline(true);
    setError(null);
    setPipelineStatus((current) => {
      const next = { ...current };
      next[firstStage] = "running";
      for (const stageId of downstreamStages) {
        next[stageId] = "waiting";
      }
      return next;
    });
    setStageResults((current) => {
      const next = { ...current };
      for (const stageId of rerunStages) {
        delete next[stageId];
      }
      return next;
    });
    setStageLogs((current) => {
      const next = { ...current };
      for (const stageId of rerunStages) {
        next[stageId] = [];
      }
      return next;
    });

    if (rerunStageSet.has("training") || rerunStageSet.has("evaluation") || rerunStageSet.has("results")) {
      setMetrics(null);
    }
    if (rerunStageSet.has("results")) {
      setExplanation(null);
    }
  }, [getRerunStages]);

  const handleRevisionRerunComplete = useCallback(async () => {
    await refreshPipelineData().catch(() => {
      // If the post-rerun refresh fails, keep the last optimistic state.
    });
    setIsRunningPipeline(false);
  }, [refreshPipelineData]);

  useEffect(() => {
    void refreshPipelineData().catch((err) => {
      const message = err instanceof Error ? err.message : "Unable to load pipeline state.";
      if (!message.includes("No dataset uploaded")) {
        setError(message);
      }
    });
  }, [refreshPipelineData]);

  useEffect(() => {
    if (pipelineStatus.results !== "completed" || explanation) return;
    let cancelled = false;
    const timeout = setTimeout(() => {
      void getExplanation()
        .then((data) => {
          if (!cancelled) setExplanation(data);
        })
        .catch(() => {
          // Silent retry; will attempt again on next refresh.
        });
    }, 800);
    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [explanation, pipelineStatus.results]);

  const evaluationResponsesReady = useMemo(() => {
    const evaluationResult = stageResults.evaluation as Record<string, unknown> | undefined;
    return Boolean(evaluationResult?.llm_insights);
  }, [stageResults.evaluation]);

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadDataset(file);
      setUploadedFile(file);
      setSelectedColumnState(null);
      setMetrics(null);
      setExplanation(null);
      setDatasetPreview(null);
      setStageResults({});
      setStageLogs(createEmptyLogs());
      setDatasetSummary({
        filename: response.filename,
        rows: response.rows,
        columns: response.columns,
        column_names: response.column_names,
        column_types: Object.fromEntries(response.column_names.map((name) => [name, "unknown"])),
        missing_values: {},
        numeric_summary: null,
      });
      setDatasetColumns(
        response.column_names.map((name) => ({
          name,
          dtype: "unknown",
          is_numeric: false,
          missing_pct: 0,
        })),
      );
      await refreshPipelineData().catch(() => {
        // Upload succeeded; keep local dataset metadata even if backend state polling fails.
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset upload failed.");
    } finally {
      setIsUploading(false);
    }
  }, [refreshPipelineData]);

  const handleTargetSelect = useCallback(async (column: string) => {
    setIsSavingTarget(true);
    setError(null);

    setSelectedColumnState(column);
    setTaskType(inferTaskType(datasetColumns, column));
    try {
      await setTargetColumn(column);
      await refreshPipelineData().catch(() => {
        // Keep local target selection if backend state polling fails.
      });
    } catch {
      // Keep local selection so users can still run the one-shot execute endpoint.
    } finally {
      setIsSavingTarget(false);
    }
  }, [datasetColumns, refreshPipelineData]);

  const handleRunPipeline = useCallback(async () => {
    if (!uploadedFile || !selectedColumn) {
      setError("Upload a dataset and select a target column before running the pipeline.");
      return;
    }

    const config: PipelineConfig = {
      task_type: taskType,
      ...DEFAULT_CONFIG,
    };

    setIsRunningPipeline(true);
    setError(null);
    const initialStatuses = createInitialStatuses();
    setPipelineStatus(initialStatuses);
    setStageResults({});
    setMetrics(null);
    setExplanation(null);
    setStageLogs(createEmptyLogs());
    setPipelineStatus((current) => ({ ...current, analysis: "running" }));

    try {
      const response = await executePipeline(uploadedFile, selectedColumn, config);
      setPipelineStatus({ ...createInitialStatuses(), ...response.stages });
      setStageResults(response.stage_results || {});
      setStageLogs({ ...createEmptyLogs(), ...(response.stage_logs || {}) });
      setMetrics(response.metrics || null);
      setExplanation(response.explanation || null);
      if (response.dataset) {
        setDatasetSummary((current) => ({
          filename: response.dataset.filename,
          rows: response.dataset.rows,
          columns: response.dataset.columns,
          column_names: response.dataset.column_names,
          column_types: current?.column_types || {},
          missing_values: current?.missing_values || {},
          numeric_summary: current?.numeric_summary || null,
        }));
      }
    } catch (err) {
      setPipelineStatus((current) => ({
        ...current,
        analysis: "failed",
      }));
      setError(err instanceof Error ? err.message : "Pipeline execution failed.");
    } finally {
      setIsRunningPipeline(false);
    }
  }, [uploadedFile, selectedColumn, taskType]);

  const getVisibleStageStatus = useCallback((stageId: string): StageStatus => {
    if (stageId === "evaluation") {
      if (pipelineStatus.evaluation === "running" || pipelineStatus.loss === "running") return "running";
      if (pipelineStatus.evaluation === "completed" && !evaluationResponsesReady) return "running";
      if (pipelineStatus.evaluation === "completed") return "completed";
      if (pipelineStatus.evaluation === "failed" || pipelineStatus.loss === "failed") return "failed";
      return "waiting";
    }
    if (stageId === "results") {
      if (!evaluationResponsesReady && pipelineStatus.evaluation === "completed") {
        return pipelineStatus.results === "failed" ? "failed" : "waiting";
      }
      return pipelineStatus[stageId] || "waiting";
    }
    return pipelineStatus[stageId] || "waiting";
  }, [evaluationResponsesReady, pipelineStatus]);

  const completedCount = VISIBLE_STAGE_ORDER.filter((stageId) => getVisibleStageStatus(stageId) === "completed").length;
  const progress = (completedCount / VISIBLE_STAGES.length) * 100;
  const activeStageId =
    pipelineStatus.evaluation === "completed" && !evaluationResponsesReady
      ? "evaluation"
      : EXECUTION_STAGE_ORDER.find((stageId) => pipelineStatus[stageId] === "running") || null;
  const isComplete = completedCount === VISIBLE_STAGES.length;
  const modelName =
    (metrics?.model_name as string | null | undefined) ||
    (stageResults.training?.model_name as string | undefined) ||
    null;
  const canRun = Boolean(datasetSummary && selectedColumn && !isUploading && !isSavingTarget);
  const focusedStageId =
    selectedStage?.id ||
    activeStageId ||
    VISIBLE_STAGE_ORDER.find((stageId) => getVisibleStageStatus(stageId) !== "completed") ||
    VISIBLE_STAGE_ORDER[0];
  const focusedStage = VISIBLE_STAGES.find((stage) => stage.id === focusedStageId) || VISIBLE_STAGES[0];
  const focusedLogs = (stageLogs[focusedStageId] || []).slice(-7).reverse();
  const missingHighlights = useMemo(
    () =>
      Object.entries(datasetSummary?.missing_values || {})
        .filter(([, value]) => Number(value) > 0)
        .sort(([, left], [, right]) => Number(right) - Number(left))
        .slice(0, 3),
    [datasetSummary],
  );

  return (
    <>
      <NeuralBackground />
      <div className="relative z-10 mx-auto w-full max-w-[1450px] space-y-5 px-4 pb-8 pt-4 sm:pt-6">
        <div className="relative z-20 flex items-center justify-between">
          <Link
            to="/"
            className="inline-flex items-center rounded-full border border-blue-400/40 bg-blue-500/10 px-3 py-1.5 text-xs font-semibold tracking-wide text-blue-100 transition hover:border-blue-300 hover:bg-blue-500/25"
          >
            ← Back to Home
          </Link>
          <div className="flex items-center gap-2 rounded-full border border-blue-500/35 bg-slate-950/55 px-3 py-1.5 text-xs text-blue-100/85">
            <WaterLogo className="h-5 w-5" />
            FlowML Studio
          </div>
        </div>

        <section className="glass-card border-blue-500/30 p-4 sm:p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.22em] text-blue-300/70">Agentic AutoML</p>
              <h1 className="mt-1 text-2xl font-semibold text-blue-100 sm:text-3xl">FlowML Lab Workspace</h1>
            </div>
            <button
              onClick={() => void handleRunPipeline()}
              disabled={!canRun || isRunningPipeline}
              className="inline-flex items-center gap-2 rounded-full border border-blue-300/45 bg-blue-500/20 px-5 py-2 text-sm font-semibold text-blue-50 transition hover:bg-blue-500/35 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {isRunningPipeline ? "Running..." : "Run Pipeline"}
            </button>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-blue-500/25 bg-blue-500/10 p-3">
              <p className="flex items-center gap-2 text-xs text-blue-200/80"><Database className="h-3.5 w-3.5" /> Dataset</p>
              <p className="mt-1 truncate text-sm font-semibold text-blue-100">{datasetSummary?.filename || "Awaiting upload"}</p>
            </div>
            <div className="rounded-xl border border-blue-500/25 bg-blue-500/10 p-3">
              <p className="flex items-center gap-2 text-xs text-blue-200/80"><Target className="h-3.5 w-3.5" /> Target</p>
              <p className="mt-1 truncate text-sm font-semibold text-blue-100">{selectedColumn || "Not selected"}</p>
            </div>
            <div className="rounded-xl border border-blue-500/25 bg-blue-500/10 p-3">
              <p className="flex items-center gap-2 text-xs text-blue-200/80"><Brain className="h-3.5 w-3.5" /> Model</p>
              <p className="mt-1 truncate text-sm font-semibold text-blue-100">{modelName || "Pending"}</p>
            </div>
            <div className="rounded-xl border border-blue-500/25 bg-blue-500/10 p-3">
              <p className="flex items-center gap-2 text-xs text-blue-200/80"><Gauge className="h-3.5 w-3.5" /> Progress</p>
              <p className="mt-1 text-sm font-semibold text-blue-100">{completedCount}/{VISIBLE_STAGES.length} complete</p>
            </div>
          </div>

          <div className="mt-4 h-2 rounded-full bg-slate-900/90">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-300"
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(progress, 2)}%` }}
              transition={{ duration: 0.45 }}
            />
          </div>
        </section>

        {error && (
          <motion.div
            className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
          </motion.div>
        )}

        <div className="grid gap-5 xl:grid-cols-[290px_minmax(0,1fr)_320px]">
          <aside className="glass-card border-blue-500/25 p-4">
            <p className="mb-3 text-xs uppercase tracking-[0.2em] text-blue-300/70">Workflow</p>
            <div className="space-y-3">
              {VISIBLE_STAGES.map((stage, index) => {
                const stageState = getVisibleStageStatus(stage.id);
                const badge =
                  stageState === "completed"
                    ? "Complete"
                    : stageState === "running"
                      ? "Running"
                      : stageState === "failed"
                        ? "Needs Fix"
                        : canRun
                          ? "Ready"
                          : "Locked";
                return (
                  <button
                    key={stage.id}
                    type="button"
                    onClick={() => setSelectedStage(stage)}
                    className={`relative w-full rounded-xl border p-3 text-left transition ${
                      focusedStageId === stage.id
                        ? "border-blue-300/70 bg-blue-500/16"
                        : "border-blue-500/25 bg-slate-950/45 hover:border-blue-400/45 hover:bg-blue-500/10"
                    }`}
                  >
                    {index < VISIBLE_STAGES.length - 1 && <span className="absolute -bottom-3 left-5 h-3 w-px bg-blue-500/45" />}
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="text-base">{stage.icon}</span>
                        <p className="text-sm font-semibold text-blue-50">{stage.label}</p>
                      </div>
                      <span className="rounded-full border border-blue-300/40 bg-blue-500/18 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-blue-100">
                        {badge}
                      </span>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-blue-100/70">{stage.description}</p>
                  </button>
                );
              })}
            </div>
          </aside>

          <section className="space-y-4">
            <div className="glass-card border-blue-500/25 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-blue-300/70">Current Step</p>
              <h2 className="mt-2 text-xl font-semibold text-blue-100">{focusedStage.label}</h2>
              <p className="mt-2 text-sm leading-relaxed text-blue-100/80">{focusedStage.details}</p>
            </div>

            {!datasetSummary ? (
              <div className="space-y-4">
                <DatasetUpload
                  fileName={null}
                  isUploading={isUploading}
                  error={null}
                  onUpload={handleUpload}
                />
                <div className="rounded-xl border border-blue-500/25 bg-blue-500/10 p-4">
                  <p className="text-sm font-semibold text-blue-100">Start here: upload your dataset</p>
                  <p className="mt-1 text-xs text-blue-100/75">Supported types: CSV, JSON, XLSX. Once uploaded, target selection and staged workflow unlock automatically.</p>
                </div>
              </div>
            ) : (
              <>
                <div className="grid gap-4 lg:grid-cols-2">
                  <DatasetUpload
                    fileName={datasetSummary.filename}
                    isUploading={isUploading}
                    error={null}
                    onUpload={handleUpload}
                  />
                  <PredictionTarget
                    datasetLoaded
                    columns={datasetColumns}
                    selectedColumn={selectedColumn}
                    isSaving={isSavingTarget}
                    onSelect={handleTargetSelect}
                  />
                </div>

                <div className="glass-card border-blue-500/25 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <p className="text-sm font-semibold text-blue-100">Dataset Preview</p>
                    <p className="text-xs text-blue-200/70">{datasetSummary.rows.toLocaleString()} rows • {datasetSummary.columns} cols</p>
                  </div>
                  {isLoadingDatasetPreview ? (
                    <p className="text-sm text-blue-100/70">Loading preview...</p>
                  ) : datasetPreview?.rows?.length ? (
                    <div className="overflow-x-auto rounded-xl border border-blue-500/20">
                      <table className="min-w-full text-left text-xs">
                        <thead className="bg-blue-500/10 text-blue-200/90">
                          <tr>
                            {datasetPreview.columns.map((column) => (
                              <th key={column} className="px-3 py-2 font-semibold">{column}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {datasetPreview.rows.slice(0, 5).map((row, index) => (
                            <tr key={`row-${index}`} className="border-t border-blue-500/15 text-blue-100/85">
                              {datasetPreview.columns.map((column) => (
                                <td key={`${index}-${column}`} className="max-w-[160px] truncate px-3 py-2">
                                  {String(row[column] ?? "—")}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-sm text-blue-100/70">Preview unavailable right now.</p>
                  )}
                </div>
              </>
            )}

            <ResultsPanel
              isComplete={isComplete}
              metrics={metrics}
              results={stageResults.results || null}
              explanation={explanation}
            />
          </section>

          <aside className="space-y-4">
            <div className="glass-card border-blue-500/25 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-blue-300/70">Run Status</p>
              <p className="mt-2 text-sm font-semibold text-blue-100">
                {isRunningPipeline ? "Pipeline running..." : isComplete ? "Pipeline complete" : "Ready"}
              </p>
              <p className="mt-1 text-xs text-blue-100/75">
                Active stage: {focusedStage.label}
              </p>
              <div className="mt-3 h-2 rounded-full bg-slate-900/90">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-300"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(progress, 2)}%` }}
                  transition={{ duration: 0.45 }}
                />
              </div>
            </div>

            <div className="glass-card border-blue-500/25 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-blue-300/70">Quick Stats</p>
              <ul className="mt-3 space-y-2 text-xs text-blue-100/80">
                <li>Rows: {datasetSummary?.rows?.toLocaleString() || "—"}</li>
                <li>Columns: {datasetSummary?.columns || "—"}</li>
                <li>Task: {taskType}</li>
                <li>Target: {selectedColumn || "Not selected"}</li>
                <li>Model: {modelName || "Pending"}</li>
              </ul>
              {missingHighlights.length > 0 && (
                <div className="mt-3 rounded-lg border border-blue-500/20 bg-blue-500/10 p-2">
                  <p className="text-[10px] uppercase tracking-[0.15em] text-blue-200/80">Missing highlights</p>
                  <ul className="mt-1 space-y-1 text-[11px] text-blue-100/75">
                    {missingHighlights.map(([column, value]) => (
                      <li key={column}>{column}: {(Number(value) * 100).toFixed(1)}%</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="glass-card border-blue-500/25 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-blue-300/70">Activity Feed</p>
              <div className="mt-3 space-y-2">
                {focusedLogs.length ? (
                  focusedLogs.map((entry, index) => (
                    <p key={`${entry}-${index}`} className="rounded-md border border-blue-500/15 bg-blue-500/10 px-2 py-1.5 text-[11px] text-blue-100/80">
                      {entry}
                    </p>
                  ))
                ) : (
                  <p className="text-xs text-blue-100/70">No activity logs yet.</p>
                )}
              </div>
            </div>
          </aside>
        </div>

        <StageDetailPanel
          stage={selectedStage}
          stageResult={selectedStage ? stageResults[selectedStage.id] || null : null}
          lossStageResult={stageResults.loss || null}
          datasetSummary={datasetSummary}
          datasetColumns={datasetColumns}
          datasetPreview={datasetPreview}
          isLoadingDatasetPreview={isLoadingDatasetPreview}
          metrics={metrics}
          stageLogs={selectedStage ? stageLogs[selectedStage.id] || [] : []}
          taskType={taskType}
          targetColumn={selectedColumn}
          explanation={explanation}
          onClose={() => setSelectedStage(null)}
        />

        <ChatBot
          datasetName={datasetSummary?.filename || null}
          targetColumn={selectedColumn}
          taskType={taskType}
          activeStageId={activeStageId}
          stageLogs={stageLogs}
          metrics={metrics}
          onRevisionRerunStart={handleRevisionRerunStart}
          onRevisionRerunComplete={handleRevisionRerunComplete}
          onPipelineRefresh={refreshPipelineData}
        />
      </div>
    </>
  );
};

const inferTaskType = (columns: DatasetColumn[], selectedColumn: string): TaskType => {
  const column = columns.find((entry) => entry.name === selectedColumn);
  if (!column) return "classification";
  if (!column.is_numeric) return "classification";
  if (typeof column.unique_values === "number" && column.unique_values <= 20) {
    return "classification";
  }
  return "regression";
};

export default Pipeline;
