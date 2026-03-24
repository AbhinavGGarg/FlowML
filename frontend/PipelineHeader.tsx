import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { Activity, Brain, Database, Play, Target } from "lucide-react";
import type { MetricsResponse } from "@/lib/api";

interface PipelineHeaderProps {
  completedCount: number;
  totalStages: number;
  progress: number;
  datasetName: string | null;
  targetColumn: string | null;
  modelName: string | null;
  metrics: MetricsResponse | null;
  canRun: boolean;
  isRunning: boolean;
  onRun: () => Promise<void> | void;
}

const PipelineHeader = ({
  completedCount,
  totalStages,
  progress,
  datasetName,
  targetColumn,
  modelName,
  metrics,
  canRun,
  isRunning,
  onRun,
}: PipelineHeaderProps) => {
  const isComplete = completedCount === totalStages;
  const showMetrics = Boolean(metrics && (metrics.accuracy || metrics.r2 !== null));

  return (
    <div className="glass-card space-y-9 p-10" data-chat-context-label="Pipeline header">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-center gap-3">
          <div>
            <h1 className="gradient-text font-mono text-2xl font-bold md:text-3xl">Agentic AutoML</h1>
            <div className="mt-0.5 flex items-center gap-2">
              <motion.div
                className={`h-2 w-2 rounded-full ${isComplete ? "bg-accent" : isRunning ? "bg-primary" : "bg-muted-foreground"}`}
                animate={isRunning ? { scale: [1, 1.3, 1], opacity: [1, 0.5, 1] } : {}}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-sm font-mono text-muted-foreground">
                {isComplete ? "Pipeline complete" : isRunning ? "Agents running..." : "Ready"}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center">
          <button
            onClick={() => void onRun()}
            disabled={!canRun || isRunning}
            className="flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-base font-semibold text-accent-foreground transition-transform hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {isRunning ? "Running pipeline..." : "Run pipeline"}
          </button>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="h-2 overflow-hidden rounded-full bg-secondary">
          <motion.div
            className="h-full rounded-full"
            style={{ background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))" }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
          <span>Progress</span>
          <span>
            {completedCount}/{totalStages} stages
          </span>
        </div>
      </div>

      <div className="flex flex-wrap gap-4">
        <InfoChip icon={<Database className="h-3 w-3" />} label="Dataset" value={datasetName || "Awaiting upload"} />
        <InfoChip icon={<Target className="h-3 w-3" />} label="Target" value={targetColumn || "Select a column"} />
        <InfoChip icon={<Brain className="h-3 w-3" />} label="Model" value={modelName || "Pending selection"} />
        {showMetrics && (
          <InfoChip
            icon={<Activity className="h-3 w-3" />}
            label={metrics?.task_type === "regression" ? "R2" : "Accuracy"}
            value={
              metrics?.task_type === "regression"
                ? formatMetric(metrics?.r2)
                : formatMetric(metrics?.accuracy, true)
            }
            highlight
          />
        )}
      </div>
    </div>
  );
};

const formatMetric = (value: number | null | undefined, asPercent = false) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "Pending";
  return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
};

const InfoChip = ({
  icon,
  label,
  value,
  highlight,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  highlight?: boolean;
}) => (
  <div className="flex items-center gap-1.5 font-mono text-[11px]">
    <span className="text-muted-foreground">{icon}</span>
    <span className="text-muted-foreground">{label}:</span>
    <span className={highlight ? "font-semibold text-accent" : "text-foreground"}>{value}</span>
  </div>
);

export default PipelineHeader;
