import type { ReactNode } from "react";
import type { PipelineStage } from "@/data/pipelineStages";
import type { DatasetSummary, MetricsResponse } from "@/lib/api";

interface StageVisualizationProps {
  stage: PipelineStage;
  stageResult: Record<string, unknown> | null;
  datasetSummary: DatasetSummary | null;
  metrics: MetricsResponse | null;
}

const EmptyState = ({ message }: { message: string }) => (
  <div className="rounded-2xl border border-dashed border-border/60 p-6 text-center text-sm text-muted-foreground">
    {message}
  </div>
);

const Heatmap = ({ stageResult }: { stageResult: Record<string, unknown> | null }) => {
  const correlations = (stageResult?.correlations as Record<string, Record<string, number>> | undefined) || {};
  const labels = Object.keys(correlations).slice(0, 5);

  if (labels.length === 0) {
    return <EmptyState message="Correlation insights will appear after analysis completes." />;
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <div className="w-12" />
        {labels.map((label) => (
          <div key={label} className="w-12 truncate text-center text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            {label}
          </div>
        ))}
      </div>
      {labels.map((rowLabel) => (
        <div key={rowLabel} className="flex items-center gap-2">
          <div className="w-12 truncate pr-1 text-right text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{rowLabel}</div>
          {labels.map((columnLabel) => {
            const raw = correlations[rowLabel]?.[columnLabel] ?? 0;
            const value = Number(raw);
            const hue = value >= 0 ? 265 : 145;
            return (
              <div
                key={`${rowLabel}-${columnLabel}`}
                className="flex h-12 w-14 items-center justify-center rounded-xl border border-border/50 font-mono text-[11px] text-foreground/80"
                style={{ backgroundColor: `hsl(${hue} 80% 60% / ${Math.abs(value) * 0.6})` }}
              >
                {value.toFixed(1)}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

const CvBars = ({ stageResult }: { stageResult: Record<string, unknown> | null }) => {
  const cvScores = (stageResult?.cv_scores as number[] | undefined) || [];
  if (cvScores.length === 0) {
    return <EmptyState message="Cross-validation scores will appear after training completes." />;
  }
  const maxScore = Math.max(...cvScores, 1);
  return (
    <div className="space-y-2">
      {cvScores.map((score, idx) => (
        <div key={idx} className="flex items-center gap-2">
          <span className="w-16 text-right font-mono text-[11px] text-muted-foreground">Fold {idx + 1}</span>
          <div className="h-5 flex-1 overflow-hidden rounded-sm bg-secondary">
            <div
              className="h-full rounded-sm"
              style={{
                width: `${(score / maxScore) * 100}%`,
                background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))",
              }}
            />
          </div>
          <span className="w-12 font-mono text-[11px] text-foreground/70">{score.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
};

const BarChart = ({ stageResult }: { stageResult: Record<string, unknown> | null }) => {
  const featureScores = (stageResult?.feature_scores as Record<string, number> | undefined) || {};
  const features = Object.entries(featureScores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, value]) => ({ name, value: Number(value) }));

  if (features.length === 0) {
    const selectedFeatures = (stageResult?.selected_features as string[] | undefined) || [];
    if (selectedFeatures.length === 0) {
      return <EmptyState message="Feature rankings will appear after feature engineering completes." />;
    }
    return (
      <div className="space-y-2">
        {selectedFeatures.slice(0, 6).map((feature, index) => (
          <div key={feature} className="flex items-center gap-2">
            <span className="w-20 text-right font-mono text-[11px] text-muted-foreground">{feature}</span>
            <div className="h-5 flex-1 overflow-hidden rounded-sm bg-secondary">
              <div
                className="h-full rounded-sm"
                style={{
                  width: `${Math.max(100 - index * 12, 25)}%`,
                  background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))",
                }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {features.map((feature) => (
        <div key={feature.name} className="flex items-center gap-2">
          <span className="w-20 text-right font-mono text-[11px] text-muted-foreground">{feature.name}</span>
          <div className="h-5 flex-1 overflow-hidden rounded-sm bg-secondary">
            <div
              className="h-full rounded-sm"
              style={{
                width: `${Math.max(feature.value, 0.05) * 100}%`,
                background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))",
              }}
            />
          </div>
          <span className="w-10 font-mono text-[11px] text-foreground/70">{(feature.value * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
};

const DataTable = ({ datasetSummary }: { datasetSummary: DatasetSummary | null }) => {
  if (!datasetSummary) {
    return <EmptyState message="Upload a dataset to inspect preprocessing inputs." />;
  }

  const rows = datasetSummary.column_names.slice(0, 8).map((columnName) => [
    columnName,
    `${((datasetSummary.missing_values[columnName] || 0) * 100).toFixed(1)}%`,
    datasetSummary.column_types[columnName] || "unknown",
  ]);

  return (
    <div className="overflow-x-auto rounded-2xl border border-border/60 bg-background/35">
      <table className="w-full font-mono text-[11px]">
        <thead>
          <tr className="border-b border-border/60 bg-background/40">
            {["feature", "missing", "type"].map((column) => (
              <th key={column} className="px-3 py-2 text-left font-medium uppercase tracking-[0.18em] text-muted-foreground">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row[0]} className="border-b border-border/30 last:border-b-0">
              {row.map((cell) => (
                <td key={cell} className="px-3 py-2 text-foreground/85">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const ConfusionMatrix = ({ metrics, stageResult }: { metrics: MetricsResponse | null; stageResult: Record<string, unknown> | null }) => {
  const matrix = (metrics?.confusion_matrix || (stageResult?.confusion_matrix as number[][] | undefined) || []).slice(0, 2);
  const labels = ["Pred 0", "Pred 1"];

  if (matrix.length === 0) {
    return <EmptyState message="Evaluation outputs will appear after the evaluation stage completes." />;
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex gap-1">
        <div className="w-14" />
        {labels.map((label) => (
          <div key={label} className="w-20 text-center text-[10px] text-muted-foreground">
            {label}
          </div>
        ))}
      </div>
      {matrix.map((row, index) => (
        <div key={`row-${index}`} className="flex items-center gap-1">
          <div className="w-14 text-right text-[10px] text-muted-foreground">Actual {index}</div>
          {row.map((value, columnIndex) => (
            <div
              key={`cell-${index}-${columnIndex}`}
              className={`flex h-14 w-20 items-center justify-center rounded-md font-mono text-sm font-semibold ${
                index === columnIndex ? "bg-primary/20 text-primary" : "bg-destructive/10 text-destructive"
              }`}
            >
              {value}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

const MetricsCard = ({ metrics }: { metrics: MetricsResponse | null }) => {
  if (!metrics) {
    return <EmptyState message="Metrics will appear here after evaluation and deployment." />;
  }

  const cards =
    metrics.task_type === "regression"
      ? [
          { label: "R2", value: formatMetric(metrics.r2, false) },
          { label: "RMSE", value: formatMetric(metrics.rmse, false) },
          { label: "MAE", value: formatMetric(metrics.mae, false) },
          { label: "CV", value: formatMetric(metrics.best_score, false) },
        ]
      : [
          { label: "Accuracy", value: formatMetric(metrics.accuracy, true) },
          { label: "Precision", value: formatMetric(metrics.precision, true) },
          { label: "Recall", value: formatMetric(metrics.recall, true) },
          { label: "F1 Score", value: formatMetric(metrics.f1, true) },
        ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {cards.map((metric) => (
        <div key={metric.label} className="glass-card p-3 text-center">
          <div className="gradient-text text-lg font-bold">{metric.value}</div>
          <div className="mt-1 text-[10px] text-muted-foreground">{metric.label}</div>
        </div>
      ))}
    </div>
  );
};

const ModelSelectionViz = ({ stageResult }: { stageResult: Record<string, unknown> | null }) => {
  if (!stageResult) {
    return <EmptyState message="Model selection details will appear after the selection stage completes." />;
  }

  const topCandidates = ((stageResult.top_candidates as Array<Record<string, unknown>> | undefined) || []).slice(0, 3);
  const selectedModel = String(topCandidates[0]?.model_name ?? "Pending");
  const candidates = topCandidates.map((candidate) => String(candidate.model_name ?? "")).filter(Boolean);
  const llmReturned = stageResult.llm_returned as boolean | undefined;
  const reasoning = String(stageResult.selection_reasoning ?? "");

  return (
    <div className="space-y-3 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground">Selected:</span>
        <span className="rounded-md bg-accent/10 px-2 py-1 font-mono text-[11px] text-accent">{selectedModel}</span>
        {typeof llmReturned === "boolean" && (
          <span className="rounded-md bg-secondary px-2 py-1 text-[11px] text-muted-foreground">
            LLM {llmReturned ? "returned" : "fallback"}
          </span>
        )}
      </div>
      {candidates.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {candidates.map((candidate) => (
            <span
              key={candidate}
              className="rounded-md bg-secondary px-2 py-1 font-mono text-[10px] text-muted-foreground"
            >
              {candidate}
            </span>
          ))}
        </div>
      ) : (
        <p className="text-muted-foreground">Candidate models will appear here after selection runs.</p>
      )}
      {reasoning && <p className="text-[11px] text-muted-foreground">{reasoning}</p>}
    </div>
  );
};

export const StageVisualization = ({
  stage,
  stageResult,
  datasetSummary,
  metrics,
}: StageVisualizationProps) => {
  const vizMap: Record<string, ReactNode> = {
    heatmap: <Heatmap stageResult={stageResult} />,
    lossCurve: <CvBars stageResult={stageResult} />,
    barChart: <BarChart stageResult={stageResult} />,
    table: <DataTable datasetSummary={datasetSummary} />,
    confusionMatrix: <ConfusionMatrix metrics={metrics} stageResult={stageResult} />,
    metrics: <MetricsCard metrics={metrics} />,
    modelSelection: <ModelSelectionViz stageResult={stageResult} />,
  };

  return <>{vizMap[stage.vizType] || <EmptyState message="No visualization available for this stage yet." />}</>;
};

const formatMetric = (value: number | null | undefined, asPercent = true) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "Pending";
  return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
};
