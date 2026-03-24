import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, Terminal } from "lucide-react";
import type { MetricsResponse } from "@/lib/api";

interface LogEntry {
  stage: string;
  message: string;
}

interface PipelineLogsProps {
  activeStageId: string | null;
  completedStageIds: string[];
  stageLogs: Record<string, string[]>;
  stageResults: Record<string, Record<string, unknown>>;
  metrics: MetricsResponse | null;
}

const PipelineLogs = ({
  activeStageId,
  completedStageIds,
  stageLogs,
  stageResults,
  metrics,
}: PipelineLogsProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<"visual" | "debug">("debug");

  const logEntries = useMemo(() => {
    const entries: LogEntry[] = [];
    const stageOrder = [
      "analysis",
      "preprocessing",
      "features",
      "model_selection",
      "training",
      "loss",
      "evaluation",
      "results",
    ];

    for (const stageId of stageOrder) {
      const messages = stageLogs[stageId] || [];
      if (messages.length === 0) continue;

      const displayMessages =
        stageId === activeStageId ? messages.slice(Math.max(messages.length - 3, 0)) : messages;

      displayMessages.forEach((message) => {
        entries.push({ stage: stageId, message });
      });
    }

    return entries;
  }, [activeStageId, stageLogs]);

  const topFeatures = useMemo(() => {
    const featureScores = stageResults.features?.feature_scores as Record<string, number> | undefined;
    if (featureScores && Object.keys(featureScores).length > 0) {
      return Object.entries(featureScores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name, value]) => ({ name, value: Number(value) }));
    }

    const selectedFeatures = stageResults.features?.selected_features as string[] | undefined;
    return (selectedFeatures || []).slice(0, 5).map((name, index, all) => ({
      name,
      value: 1 - index / Math.max(all.length, 1),
    }));
  }, [stageResults.features]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logEntries.length]);

  const stageLabel = (id: string) => {
    const map: Record<string, string> = {
      analysis: "Analysis",
      preprocessing: "Preprocess",
      features: "Features",
      model_selection: "Model Select",
      training: "Training",
      loss: "Loss",
      evaluation: "Eval",
      results: "Results",
    };
    return map[id] || id;
  };

  const renderMetricCards = () => {
    if (!metrics) return null;

    if (metrics.task_type === "regression") {
      return [
        { label: "R2", value: formatMetric(metrics.r2, false) },
        { label: "RMSE", value: formatMetric(metrics.rmse, false) },
        { label: "MAE", value: formatMetric(metrics.mae, false) },
        { label: "CV", value: formatMetric(metrics.best_score, false) },
      ];
    }

    return [
      { label: "Accuracy", value: formatMetric(metrics.accuracy, true) },
      { label: "Precision", value: formatMetric(metrics.precision, true) },
      { label: "Recall", value: formatMetric(metrics.recall, true) },
      { label: "F1", value: formatMetric(metrics.f1, true) },
    ];
  };

  const metricCards = renderMetricCards();

  return (
    <div className="glass-card overflow-hidden" data-chat-context-label="Pipeline logs">
      <div className="flex items-center gap-0 border-b border-border/30">
        <button
          onClick={() => setActiveTab("visual")}
          className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
            activeTab === "visual" ? "border-b-2 border-accent text-accent" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <BarChart3 className="h-3.5 w-3.5" />
          Visual
        </button>
        <button
          onClick={() => setActiveTab("debug")}
          className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
            activeTab === "debug" ? "border-b-2 border-accent text-accent" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Terminal className="h-3.5 w-3.5" />
          Debug
        </button>
        <div className="ml-auto flex gap-1 pr-3">
          <div className="h-2 w-2 rounded-full bg-destructive/60" />
          <div className="h-2 w-2 rounded-full bg-primary/40" />
          <div className="h-2 w-2 rounded-full bg-accent/60" />
        </div>
      </div>

      {activeTab === "visual" ? (
        <div className="space-y-4 p-4">
          {completedStageIds.length === 0 && !activeStageId ? (
            <p className="py-6 text-center text-sm text-muted-foreground">Upload a dataset, choose a target, and run the pipeline to see live outputs.</p>
          ) : (
            <>
              {metricCards && (
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {metricCards.map((metric) => (
                    <div key={metric.label} className="rounded-lg bg-secondary/50 p-2 text-center">
                      <div className="gradient-text text-sm font-bold">{metric.value}</div>
                      <div className="text-[10px] text-muted-foreground">{metric.label}</div>
                    </div>
                  ))}
                </div>
              )}

              {topFeatures.length > 0 ? (
                <div className="space-y-1">
                  <p className="text-[11px] font-medium text-muted-foreground">Feature importance</p>
                  {topFeatures.map((feature) => (
                    <div key={feature.name} className="flex items-center gap-2">
                      <span className="w-20 text-right font-mono text-[10px] text-muted-foreground">{feature.name}</span>
                      <div className="h-3 flex-1 overflow-hidden rounded-sm bg-secondary">
                        <div
                          className="h-full rounded-sm"
                          style={{
                            width: `${Math.max(feature.value, 0.05) * 100}%`,
                            background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))",
                          }}
                        />
                      </div>
                      <span className="w-10 font-mono text-[10px] text-foreground/60">
                        {(feature.value * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="py-4 text-center text-sm text-muted-foreground">Feature insights will appear after the feature engineering stage completes.</p>
              )}
            </>
          )}
        </div>
      ) : (
        <div ref={scrollRef} className="scrollbar-thin h-48 space-y-0.5 overflow-y-auto p-3 font-mono text-[11px]">
          <AnimatePresence>
            {logEntries.map((entry, index) => (
              <motion.div
                key={`${entry.stage}-${index}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.02 }}
                className="flex gap-2"
              >
                <span className="shrink-0 text-accent">[{stageLabel(entry.stage)}]</span>
                <span
                  className={`whitespace-pre-wrap leading-relaxed ${
                    entry.message.includes(" summary:")
                      ? "text-accent"
                      : entry.message.includes(" overall:")
                        ? "text-primary"
                        : "text-foreground/70"
                  }`}
                >
                  {entry.message}
                </span>
              </motion.div>
            ))}
          </AnimatePresence>

          {logEntries.length === 0 && (
            <p className="text-muted-foreground">No pipeline logs yet.</p>
          )}

          {activeStageId && (
            <motion.div
              className="text-primary"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              _
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
};

const formatMetric = (value: number | null | undefined, asPercent = true) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "Pending";
  return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
};

export default PipelineLogs;
