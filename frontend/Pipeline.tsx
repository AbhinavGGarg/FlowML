import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { stages, type PipelineStage, type StageStatus } from "@/data/pipelineStages";
import {
  getDatasetColumns,
  getDatasetPreview,
  getDatasetSummary,
  getExplanation,
  getMetrics,
  getPipelineLogs,
  getPipelineStatus,
  getStageResult,
  runPipelineStage,
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
import PipelineConnector from "./PipelineConnector";
import PipelineHeader from "./PipelineHeader";
import PipelineNode from "./PipelineNode";
import PredictionTarget from "./PredictionTarget";
import ResultsPanel from "./ResultsPanel";
import StageDetailPanel from "./StageDetailPanel";
import TitleComponent from "./TitleComponent";
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
      await uploadDataset(file);
      setMetrics(null);
      setExplanation(null);
      setDatasetPreview(null);
      setStageResults({});
      setStageLogs(createEmptyLogs());
      await refreshPipelineData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset upload failed.");
    } finally {
      setIsUploading(false);
    }
  }, [refreshPipelineData]);

  const handleTargetSelect = useCallback(async (column: string) => {
    setIsSavingTarget(true);
    setError(null);

    try {
      await setTargetColumn(column);
      setSelectedColumnState(column);
      setTaskType(inferTaskType(datasetColumns, column));
      await refreshPipelineData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to set target column.");
    } finally {
      setIsSavingTarget(false);
    }
  }, [datasetColumns, refreshPipelineData]);

  const handleRunPipeline = useCallback(async () => {
    if (!datasetSummary || !selectedColumn) {
      setError("Upload a dataset and select a target column before running the pipeline.");
      return;
    }

    const config: PipelineConfig = {
      task_type: taskType,
      ...DEFAULT_CONFIG,
    };

    setIsRunningPipeline(true);
    setError(null);
    setPipelineStatus(createInitialStatuses());
    setStageResults({});
    setMetrics(null);
    setExplanation(null);
    setStageLogs(createEmptyLogs());

    for (let index = 0; index < EXECUTION_STAGE_ORDER.length; index += 1) {
      const stageId = EXECUTION_STAGE_ORDER[index];

      setPipelineStatus((current) => ({
        ...current,
        [stageId]: "running",
      }));

      try {
        const response = await runPipelineStage(stageId, config);
        const status = response.status as StageStatus;
        setPipelineStatus((current) => ({
          ...current,
          [stageId]: status,
        }));

        setStageResults((current) => ({
          ...current,
          [stageId]: response.result || {},
        }));

        await refreshLogs();

        if (stageId === "evaluation" || stageId === "results") {
          try {
            setMetrics(await getMetrics());
          } catch {
            setMetrics(null);
          }
        }
        if (stageId === "results") {
          try {
            setExplanation(await getExplanation());
          } catch {
            // Keep any existing explanation; allow retry later.
          }
        }
      } catch (err) {
        setPipelineStatus((current) => ({
          ...current,
          [stageId]: "failed",
        }));
        await refreshLogs();
        setError(err instanceof Error ? err.message : `The ${stageId} stage failed.`);
        break;
      }
    }

    await refreshPipelineData().catch(() => {
      // Keep the UI state we already have if the final refresh fails.
    });
    setIsRunningPipeline(false);
  }, [datasetSummary, refreshLogs, refreshPipelineData, selectedColumn, taskType]);

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

  return (
    <>
      <NeuralBackground />
      <div className="relative z-10 mx-auto w-full max-w-6xl space-y-5 px-4 pb-8 pt-4 sm:pt-6">
        <div className="relative z-20 mb-6 flex items-end gap-4 sm:mb-8 sm:gap-6">
          <WaterLogo className="h-auto w-24 sm:w-32 md:w-36 lg:w-40" />
          <div className="translate-y-1 sm:translate-y-2">
            <TitleComponent />
          </div>
        </div>

        <PipelineHeader
          completedCount={completedCount}
          totalStages={VISIBLE_STAGES.length}
          progress={progress}
          datasetName={datasetSummary?.filename || null}
          targetColumn={selectedColumn}
          modelName={modelName}
          metrics={metrics}
          canRun={canRun}
          isRunning={isRunningPipeline}
          onRun={handleRunPipeline}
        />

        {error && (
          <motion.div
            className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
          </motion.div>
        )}

        <DatasetUpload
          fileName={datasetSummary?.filename || null}
          isUploading={isUploading}
          error={null}
          onUpload={handleUpload}
        />

        <PredictionTarget
          datasetLoaded={Boolean(datasetSummary)}
          columns={datasetColumns}
          selectedColumn={selectedColumn}
          isSaving={isSavingTarget}
          onSelect={handleTargetSelect}
        />

        <div className="flex w-full items-start pb-6 pt-3">
          {VISIBLE_STAGES.map((stage, index) => (
            <div
              key={stage.id}
              className={index < VISIBLE_STAGES.length - 1 ? "flex min-w-0 flex-1 items-start" : "flex shrink-0 items-start"}
            >
              <PipelineNode
                stage={stage}
                status={getVisibleStageStatus(stage.id)}
                index={index}
                onClick={() => setSelectedStage(stage)}
              />
              {index < VISIBLE_STAGES.length - 1 && (
                <PipelineConnector
                  completed={getVisibleStageStatus(stage.id) === "completed"}
                  running={getVisibleStageStatus(VISIBLE_STAGES[index + 1].id) === "running"}
                />
              )}
            </div>
          ))}
        </div>

        {(completedCount > 0 || activeStageId) && (
          <motion.p
            className="text-center text-xs text-muted-foreground"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Hover for stage context. Click completed or failed stages for details.
          </motion.p>
        )}

        <ResultsPanel
          isComplete={isComplete}
          metrics={metrics}
          results={stageResults.results || null}
          explanation={explanation}
        />

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
  return column?.is_numeric ? "regression" : "classification";
};

export default Pipeline;
