import { Fragment, type ReactNode, useMemo } from "react";
import {
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Info,
  LineChart,
  Radar,
  Target,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart as RechartsLineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PipelineStage } from "@/data/pipelineStages";
import {
  type MetricsResponse,
  type TaskType,
} from "@/lib/api";
import {
  buildBaselineSummary,
  buildClassMetricBars,
  buildConfidenceHistogram,
  buildCrossValidationData,
  buildGeneralizationSummary,
  buildLossCurveData,
  buildLossSummary,
  buildMetricHighlightCards,
  buildRegressionScatterData,
  buildResidualHistogram,
  formatMetricValue,
  getEvaluationSubtitle,
  getUnavailableInsights,
  resolveTaskType,
  type EvaluationStageResultLike,
  type LossStageResultLike,
  type MetricHighlightCard,
} from "@/lib/evaluationStage";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface EvaluationDashboardProps {
  stage: PipelineStage;
  stageResult: EvaluationStageResultLike | null;
  lossStageResult: LossStageResultLike | null;
  metrics: MetricsResponse | null;
  taskType: TaskType;
  targetColumn: string | null;
}

const EvaluationDashboard = ({
  stage,
  stageResult,
  lossStageResult,
  metrics,
  taskType,
  targetColumn,
}: EvaluationDashboardProps) => {
  const resolvedTaskType = resolveTaskType(stageResult, metrics, taskType);
  const insights = stageResult?.llm_insights ?? null;
  const displayInsights = insights ?? getUnavailableInsights(resolvedTaskType);

  const highlightCards = useMemo(
    () =>
      buildMetricHighlightCards({
        taskType: resolvedTaskType,
        metrics,
        stageResult,
        insights: displayInsights,
        targetColumn,
      }),
    [displayInsights, metrics, resolvedTaskType, stageResult, targetColumn],
  );
  const generalization = useMemo(() => buildGeneralizationSummary(metrics), [metrics]);
  const cvData = useMemo(() => buildCrossValidationData(metrics), [metrics]);
  const baseline = useMemo(
    () => buildBaselineSummary({ taskType: resolvedTaskType, metrics, stageResult }),
    [metrics, resolvedTaskType, stageResult],
  );
  const scatterData = useMemo(() => buildRegressionScatterData(stageResult), [stageResult]);
  const residualHistogram = useMemo(() => buildResidualHistogram(stageResult), [stageResult]);
  const confidenceHistogram = useMemo(() => buildConfidenceHistogram(stageResult), [stageResult]);
  const classMetricBars = useMemo(() => buildClassMetricBars(stageResult), [stageResult]);
  const lossSource = lossStageResult?.loss_source as string | undefined;
  const treeMetrics = lossStageResult?.tree_metrics as
    | { num_trees?: number | null; train_scores?: number[]; val_scores?: number[]; score_name?: string }
    | undefined;
  const showTreeMetrics = Boolean(treeMetrics);
  const showLoss = !showTreeMetrics && lossSource === "real";
  const showTrainingBehaviorSection = showTreeMetrics || showLoss;
  const lossCurveData = useMemo(
    () => (showLoss ? buildLossCurveData(lossStageResult) : []),
    [lossStageResult, showLoss],
  );
  const lossSummary = useMemo(
    () =>
      showLoss
        ? buildLossSummary(lossStageResult)
        : { bestEpoch: "N/A", finalTrainLoss: "N/A", finalValLoss: "N/A" },
    [lossStageResult, showLoss],
  );
  const modelName = metrics?.model_name ?? "Current model";
  const subtitle = getEvaluationSubtitle(resolvedTaskType);

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <section className="glass-card relative overflow-hidden border-primary/20 p-6">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(129,140,248,0.18),transparent_34%),radial-gradient(circle_at_bottom_left,rgba(34,211,238,0.14),transparent_30%),radial-gradient(circle_at_center,rgba(16,185,129,0.10),transparent_42%)]" />
          <div className="relative flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <Pill className="border-primary/35 bg-primary/10 text-primary">{stage.label}</Pill>
                <Pill className="border-accent/30 bg-accent/10 text-accent">
                  {resolvedTaskType === "regression" ? "Regression tutoring view" : "Classification tutoring view"}
                </Pill>
                {targetColumn && (
                  <Pill className="border-border/70 bg-background/35 text-muted-foreground">Target: {targetColumn}</Pill>
                )}
                <Pill className="border-border/70 bg-background/35 text-muted-foreground">{modelName}</Pill>
              </div>

              <div>
                <h2 className="text-3xl font-semibold tracking-tight text-foreground">Evaluation</h2>
                <p className="mt-2 max-w-3xl text-sm leading-7 text-secondary-foreground">{subtitle}</p>
              </div>

              <div className="glass-card max-w-3xl border-border/60 bg-background/35 p-4">
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">
                  Run Summary
                </div>
                <p className="mt-3 text-sm leading-7 text-foreground/90">
                  {displayInsights.stage_summary}
                </p>
              </div>
            </div>

            <div className="glass-card min-w-[280px] border-border/60 bg-background/35 p-5 xl:max-w-sm">
              <div className="flex items-center gap-2">
                <Radar className="h-4 w-4 text-accent" />
                <p className="text-sm font-semibold text-foreground">What happened in this run?</p>
              </div>
              <p className="mt-3 text-sm leading-7 text-secondary-foreground">{displayInsights.performance_story}</p>
            </div>
          </div>
        </section>

        <section className="glass-card border-border/60 p-5">
          <SectionHeading
            title="Notes"
            icon={BrainCircuit}
            description="Beginner-friendly teaching notes for the evaluation stage."
          />
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            {displayInsights.beginner_notes.map((note, index) => (
              <NoteCard
                key={`${note}-${index}`}
                title={index === 0 ? "What happened" : index === 1 ? "What's good" : "What to improve"}
                body={note}
              />
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <SectionHeading title="Stage highlights" icon={Target} description="Key evaluation signals, paired with interpretation instead of raw numbers alone." />
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {highlightCards.map((card) => (
              <HighlightCard key={card.key} card={card} />
            ))}
          </div>
        </section>

        <section className="glass-card border-border/60 p-5">
          <SectionHeading
            title="Model Performance Story"
            icon={Radar}
          />
          <p className="mt-4 text-sm leading-7 text-secondary-foreground">{displayInsights.performance_story}</p>
        </section>

        {showTrainingBehaviorSection && (showTreeMetrics ? (
          <section className="glass-card border-border/60 p-5">
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <StatTile label="Number of trees" value={String(treeMetrics?.num_trees ?? "Pending")} />
              <StatTile label="Final train score" value={formatMetricValue(treeMetrics?.train_scores?.slice(-1)[0], false)} />
              <StatTile label="Final validation score" value={formatMetricValue(treeMetrics?.val_scores?.slice(-1)[0], false)} />
            </div>
            <div className="mt-4 h-64">
              {(treeMetrics?.train_scores?.length ?? 0) > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsLineChart
                    data={(treeMetrics?.train_scores || []).map((value, index) => ({
                      step: `Step ${index + 1}`,
                      trainScore: value,
                      valScore: treeMetrics?.val_scores?.[index],
                    }))}
                  >
                    <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
                    <XAxis dataKey="step" tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <RechartsTooltip
                      contentStyle={{
                        background: "rgba(15,23,42,0.96)",
                        border: "1px solid rgba(148,163,184,0.24)",
                        borderRadius: 14,
                      }}
                      formatter={(value: number, name: string) => [Number(value).toFixed(3), name]}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey="trainScore" name="Train score" stroke="rgba(129,140,248,0.95)" strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="valScore" name="Validation score" stroke="rgba(16,185,129,0.95)" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
                  </RechartsLineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="Score progression will appear here when the model reports it." />
              )}
            </div>
            <p className="mt-4 text-sm leading-7 text-secondary-foreground">
              Scores show how accuracy (or R2) changes as more trees are added.
            </p>
          </section>
        ) : (
          <section className="glass-card border-border/60 p-5">
            <SectionHeading
              title="Training Behavior / Loss Review"
              icon={LineChart}
              description="The old Loss step is folded in here so you can judge whether training stayed stable or started to drift."
            />
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <StatTile label="Best epoch" value={lossSummary.bestEpoch} />
              <StatTile label="Final train loss" value={lossSummary.finalTrainLoss} />
              <StatTile label="Final val loss" value={lossSummary.finalValLoss} />
            </div>
            <div className="mt-4 h-64">
              {lossCurveData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsLineChart data={lossCurveData}>
                    <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
                    <XAxis dataKey="epoch" tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <RechartsTooltip
                      contentStyle={{
                        background: "rgba(15,23,42,0.96)",
                        border: "1px solid rgba(148,163,184,0.24)",
                        borderRadius: 14,
                      }}
                      formatter={(value: number, name: string) => [Number(value).toFixed(3), name]}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey="trainLoss" name="Train loss" stroke="rgba(129,140,248,0.95)" strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="valLoss" name="Validation loss" stroke="rgba(16,185,129,0.95)" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
                  </RechartsLineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart
                  message={
                    showLoss
                      ? "Loss curves will appear here when the training stage reports them."
                      : "Loss curves are only shown when the model reports real training history."
                  }
                />
              )}
            </div>
            <p className="mt-4 text-sm leading-7 text-secondary-foreground">{displayInsights.loss_explanation}</p>
          </section>
        ))}

        <section className="space-y-3">
          {resolvedTaskType === "regression" ? (
            <div className="grid gap-4 xl:grid-cols-2">
              <ChartCard
                title="Predicted vs Actual"
                description="Points near the diagonal mean the predictions stayed close to the true values."
                explanation={displayInsights.chart_explanations.primary_chart}
              >
                <div className="h-72">
                  {scatterData.length > 0 ? (
                    <RegressionScatterChart data={scatterData} />
                  ) : (
                    <EmptyChart message="Predicted and actual values will appear here after evaluation." />
                  )}
                </div>
              </ChartCard>
              <ChartCard
                title="Residual / Error Distribution"
                description="This helps you see whether the model usually makes small errors or sometimes very large ones."
                explanation={displayInsights.chart_explanations.secondary_chart}
              >
                <div className="h-72">
                  {residualHistogram.length > 0 ? (
                    <HistogramChart data={residualHistogram} color="rgba(129,140,248,0.82)" />
                  ) : (
                    <EmptyChart message="Residuals will appear here after the regression evaluation completes." />
                  )}
                </div>
              </ChartCard>
            </div>
          ) : (
            <div className="grid gap-4 xl:grid-cols-2">
              <ChartCard
                title="Confusion Matrix"
                description="This shows where the classifier is correct and which classes it tends to confuse."
                explanation={displayInsights.chart_explanations.primary_chart}
              >
                <div className="min-h-[18rem]">
                  <ConfusionMatrixCard stageResult={stageResult} metrics={metrics} />
                </div>
              </ChartCard>
              <ChartCard
                title={classMetricBars.length > 0 ? "Class-wise Metrics" : "Prediction Confidence"}
                description={
                  classMetricBars.length > 0
                    ? "Compare precision, recall, and F1 across the classes that matter."
                    : "Confidence scores can reveal whether the model is certain or hesitant."
                }
                explanation={displayInsights.chart_explanations.secondary_chart}
              >
                <div className="h-72">
                  {classMetricBars.length > 0 ? (
                    <ClassMetricsChart data={classMetricBars} />
                  ) : confidenceHistogram.length > 0 ? (
                    <HistogramChart data={confidenceHistogram} color="rgba(16,185,129,0.82)" />
                  ) : (
                    <EmptyChart message="Class metrics or confidence scores will appear here after classification evaluation." />
                  )}
                </div>
              </ChartCard>
            </div>
          )}
        </section>

        <section className="grid gap-4 xl:grid-cols-2">
          <div className="glass-card border-border/60 p-5">
            <SectionHeading
              title="Generalization / Overfitting Check"
              icon={LineChart}
              description="Train and test scores help show whether the model learned a reusable pattern or memorized the training set."
            />
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <StatTile label="Train score" value={generalization.trainLabel} />
              <StatTile label="Test score" value={generalization.testLabel} />
              <StatTile label="Gap" value={generalization.gapLabel} />
            </div>
            <p className="mt-4 text-sm leading-7 text-secondary-foreground">{displayInsights.generalization_explanation}</p>
          </div>

          <div className="glass-card border-border/60 p-5">
            <SectionHeading
              title="Cross-Validation Stability"
              icon={BarChart3}
              description="This shows whether the model stays steady across several validation folds, like an AutoML reliability check."
            />
            <div className="mt-4 h-48">
              {cvData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={cvData}>
                    <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
                    <XAxis dataKey="fold" tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 1]} tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <RechartsTooltip
                      contentStyle={{
                        background: "rgba(15,23,42,0.96)",
                        border: "1px solid rgba(148,163,184,0.24)",
                        borderRadius: 14,
                      }}
                      formatter={(value: number) => formatMetricValue(value, false)}
                    />
                    <Bar dataKey="score" radius={[10, 10, 4, 4]}>
                      {cvData.map((entry) => (
                        <Cell key={entry.fold} fill="rgba(34,211,238,0.85)" />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="Cross-validation fold scores will appear here after training returns them." />
              )}
            </div>
            <div className="mt-4 flex flex-wrap gap-3">
              <InlineStat label="CV mean" value={formatMetricValue(metrics?.best_score ?? null, false)} />
              <InlineStat label="CV std" value={formatMetricValue(metrics?.cv_std ?? null, false)} />
            </div>
            <p className="mt-4 text-sm leading-7 text-secondary-foreground">{displayInsights.cross_validation_explanation}</p>
          </div>
        </section>

        <section className="glass-card border-border/60 p-5">
          <SectionHeading
            title="Baseline Comparison"
            icon={CheckCircle2}
            description="A good model should beat a simple starting point, not just produce a decent-looking score."
          />
          <div className="mt-4 rounded-2xl border border-border/60 bg-background/35 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">{baseline.label}</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <StatTile label="Baseline" value={baseline.baselineValue} />
              <StatTile label="Model" value={baseline.modelValue} />
              <StatTile label="Improvement" value={baseline.improvement} accent />
            </div>
          </div>
          <p className="mt-4 text-sm leading-7 text-secondary-foreground">{displayInsights.baseline_explanation}</p>
        </section>
      </div>
    </TooltipProvider>
  );
};

const SectionHeading = ({
  title,
  description,
  icon: Icon,
}: {
  title: string;
  description: string;
  icon: typeof BrainCircuit;
}) => (
  <div className="flex items-start gap-3">
    <div className="rounded-2xl border border-border/60 bg-background/40 p-2.5">
      <Icon className="h-4 w-4 text-accent" />
    </div>
    <div>
      <p className="text-sm font-semibold text-foreground">{title}</p>
      <p className="mt-1 text-sm text-secondary-foreground">{description}</p>
    </div>
  </div>
);

const HighlightCard = ({ card }: { card: MetricHighlightCard }) => (
  <div className="glass-card border-border/60 bg-background/35 p-4">
    <div className="flex items-start justify-between gap-3">
      <MetricTooltipLabel label={card.label} tooltip={card.tooltip} />
      <span className={`rounded-full border px-2.5 py-1 text-[11px] ${card.tone}`}>{card.badge}</span>
    </div>
    <p className="mt-4 text-2xl font-semibold text-foreground">{card.value}</p>
    <p className="mt-3 text-sm leading-6 text-secondary-foreground">{card.statement}</p>
  </div>
);

const MetricTooltipLabel = ({ label, tooltip }: { label: string; tooltip: string }) => (
  <div className="flex items-center gap-2">
    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">{label}</p>
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="rounded-full text-muted-foreground transition hover:text-foreground">
          <Info className="h-3.5 w-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs text-[11px] leading-5">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  </div>
);

const StatTile = ({ label, value, accent }: { label: string; value: string; accent?: boolean }) => (
  <div className="rounded-2xl border border-border/60 bg-background/35 p-4">
    <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{label}</p>
    <p className={`mt-2 text-xl font-semibold ${accent ? "text-accent" : "text-foreground"}`}>{value}</p>
  </div>
);

const InlineStat = ({ label, value }: { label: string; value: string }) => (
  <div className="rounded-full border border-border/60 bg-background/35 px-3 py-1.5 text-xs text-secondary-foreground">
    <span className="mr-2 text-muted-foreground">{label}</span>
    <span className="font-mono text-foreground">{value}</span>
  </div>
);

const ChartCard = ({
  title,
  description,
  explanation,
  children,
}: {
  title: string;
  description: string;
  explanation: string;
  children: ReactNode;
}) => (
  <div className="glass-card border-border/60 p-5">
    <p className="text-sm font-semibold text-foreground">{title}</p>
    <p className="mt-1 text-sm text-secondary-foreground">{description}</p>
    <div className="mt-4">{children}</div>
    <div className="mt-4 rounded-2xl border border-border/60 bg-background/35 p-4 text-sm leading-7 text-secondary-foreground">
      {explanation}
    </div>
  </div>
);

const RegressionScatterChart = ({
  data,
}: {
  data: Array<{ actual: number; predicted: number; residual: number }>;
}) => {
  const minValue = Math.min(...data.map((point) => Math.min(point.actual, point.predicted)));
  const maxValue = Math.max(...data.map((point) => Math.max(point.actual, point.predicted)));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart>
        <CartesianGrid stroke="rgba(148,163,184,0.16)" />
        <XAxis
          type="number"
          dataKey="actual"
          name="Actual"
          tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="number"
          dataKey="predicted"
          name="Predicted"
          tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <RechartsTooltip
          contentStyle={{
            background: "rgba(15,23,42,0.96)",
            border: "1px solid rgba(148,163,184,0.24)",
            borderRadius: 14,
          }}
          formatter={(value: number, name: string) => [value.toFixed(3), name]}
        />
        <ReferenceLine segment={[{ x: minValue, y: minValue }, { x: maxValue, y: maxValue }]} stroke="rgba(16,185,129,0.9)" strokeDasharray="4 4" />
        <Scatter data={data} fill="rgba(129,140,248,0.85)" />
      </ScatterChart>
    </ResponsiveContainer>
  );
};

const HistogramChart = ({ data, color }: { data: Array<{ label: string; value: number }>; color: string }) => (
  <ResponsiveContainer width="100%" height="100%">
    <BarChart data={data}>
      <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
      <XAxis dataKey="label" tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 10 }} axisLine={false} tickLine={false} interval={0} angle={-20} textAnchor="end" height={56} />
      <YAxis tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
      <RechartsTooltip
        contentStyle={{
          background: "rgba(15,23,42,0.96)",
          border: "1px solid rgba(148,163,184,0.24)",
          borderRadius: 14,
        }}
      />
      <Bar dataKey="value" fill={color} radius={[10, 10, 4, 4]} />
    </BarChart>
  </ResponsiveContainer>
);

const ClassMetricsChart = ({
  data,
}: {
  data: Array<{ label: string; precision: number; recall: number; f1: number }>;
}) => (
  <ResponsiveContainer width="100%" height="100%">
    <BarChart data={data}>
      <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
      <XAxis dataKey="label" tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
      <YAxis domain={[0, 1]} tick={{ fill: "rgba(203,213,225,0.72)", fontSize: 11 }} axisLine={false} tickLine={false} />
      <RechartsTooltip
        contentStyle={{
          background: "rgba(15,23,42,0.96)",
          border: "1px solid rgba(148,163,184,0.24)",
          borderRadius: 14,
        }}
        formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
      />
      <Legend wrapperStyle={{ fontSize: 11 }} />
      <Bar dataKey="precision" fill="rgba(129,140,248,0.85)" radius={[6, 6, 0, 0]} />
      <Bar dataKey="recall" fill="rgba(34,211,238,0.85)" radius={[6, 6, 0, 0]} />
      <Bar dataKey="f1" fill="rgba(16,185,129,0.85)" radius={[6, 6, 0, 0]} />
    </BarChart>
  </ResponsiveContainer>
);

const ConfusionMatrixCard = ({
  stageResult,
  metrics,
}: {
  stageResult: EvaluationStageResultLike | null;
  metrics: MetricsResponse | null;
}) => {
  const matrix = (stageResult?.confusion_matrix ?? metrics?.confusion_matrix ?? []).filter(Array.isArray);
  const labels =
    Array.isArray(stageResult?.class_labels) && stageResult.class_labels.length === matrix.length
      ? stageResult.class_labels.map((label) => String(label))
      : matrix.map((_, index) => `Class ${index}`);
  const maxValue = Math.max(...matrix.flatMap((row) => row.map((value) => Number(value))), 1);

  if (matrix.length === 0) {
    return <EmptyChart message="Confusion counts will appear here after classification evaluation." />;
  }

  return (
    <div className="overflow-x-auto">
      <div className="grid min-w-[20rem] gap-2" style={{ gridTemplateColumns: `80px repeat(${matrix.length}, minmax(70px, 1fr))` }}>
        <div />
        {labels.map((label) => (
          <div key={`pred-${label}`} className="px-2 text-center text-[11px] text-muted-foreground">
            Pred {label}
          </div>
        ))}
        {matrix.map((row, rowIndex) => (
          <Fragment key={`row-${labels[rowIndex]}`}>
            <div key={`actual-${labels[rowIndex]}`} className="flex items-center justify-end px-2 text-[11px] text-muted-foreground">
              Actual {labels[rowIndex]}
            </div>
            {row.map((value, columnIndex) => {
              const alpha = Math.max(Number(value) / maxValue, 0.14);
              const diagonal = rowIndex === columnIndex;
              return (
                <div
                  key={`cell-${rowIndex}-${columnIndex}`}
                  className="flex h-20 items-center justify-center rounded-2xl border font-mono text-sm font-semibold"
                  style={{
                    borderColor: diagonal ? "rgba(16,185,129,0.32)" : "rgba(244,114,182,0.22)",
                    background: diagonal
                      ? `rgba(16,185,129,${alpha})`
                      : `rgba(244,114,182,${alpha * 0.72})`,
                    color: diagonal ? "rgb(220 252 231)" : "rgb(255 228 230)",
                  }}
                >
                  {Number(value)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
};

const NoteCard = ({ title, body }: { title: string; body: string }) => (
  <div className="rounded-2xl border border-border/60 bg-background/35 p-4">
    <p className="text-sm font-semibold text-foreground">{title}</p>
    <p className="mt-3 text-sm leading-7 text-secondary-foreground">{body}</p>
  </div>
);

const EmptyChart = ({ message }: { message: string }) => (
  <div className="flex h-full min-h-[12rem] items-center justify-center rounded-2xl border border-dashed border-border/60 bg-background/20 px-6 text-center text-sm text-muted-foreground">
    {message}
  </div>
);

const Pill = ({ className, children }: { className: string; children: ReactNode }) => (
  <span className={`rounded-full border px-3 py-1 text-xs ${className}`}>{children}</span>
);

export default EvaluationDashboard;
