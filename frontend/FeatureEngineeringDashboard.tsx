import {
  BrainCircuit,
  Filter,
  GraduationCap,
  Radar,
  Sparkles,
} from "lucide-react";
import type { TaskType, MetricsResponse } from "@/lib/api";
import type { PipelineStage } from "@/data/pipelineStages";
import {
  buildFeatureStageViewModel,
  formatFeatureLabel,
  type FeatureInsightCard,
  type FeatureStageResultLike,
} from "@/lib/featureStage";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface FeatureEngineeringDashboardProps {
  stage: PipelineStage;
  stageResult: FeatureStageResultLike | null;
  metrics: MetricsResponse | null;
  stageLogs: string[];
  taskType: TaskType;
  targetColumn: string | null;
}

const FeatureEngineeringDashboard = ({
  stage,
  stageResult,
  metrics,
  stageLogs,
  taskType,
  targetColumn,
}: FeatureEngineeringDashboardProps) => {
  const viewModel = buildFeatureStageViewModel({
    stageResult,
    stageLogs,
    metrics,
    taskType,
    targetColumn,
  });

  return (
    <TooltipProvider>
      <div className="space-y-6" data-chat-context-label="Feature engineering dashboard">
        <section className="glass-card relative overflow-hidden border-primary/20 p-6" data-chat-context-label="Feature engineering overview">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(128,90,213,0.18),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(34,197,94,0.16),transparent_30%)]" />
          <div className="relative space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-primary/40 bg-primary/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-primary">
                    {stage.label}
                  </span>
                  <span className="rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent">
                    {viewModel.taskType === "classification" ? "Classification scoring" : "Regression scoring"}
                  </span>
                  {viewModel.targetColumn && (
                    <span className="rounded-full border border-border/70 bg-background/40 px-3 py-1 text-xs text-muted-foreground">
                      Target: {formatFeatureLabel(viewModel.targetColumn)}
                    </span>
                  )}
                </div>
                <div>
                  <h2 className="text-3xl font-semibold tracking-tight text-foreground">Features</h2>
                  <p className="mt-2 max-w-3xl text-sm leading-7 text-secondary-foreground">
                    This stage selects, transforms, and creates features that help the model learn patterns instead of memorizing noise.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <FeatureNotes viewModel={viewModel} />

        <section className="w-full" data-chat-context-label="Feature engineering insights">
          <div className="w-full space-y-4">
            <SelectedFeaturesPanel features={readSelectedFeatures(stageResult)} />
            <TopFeaturesSection features={viewModel.topFeatures} />
            <DroppedFeaturesSection viewModel={viewModel} />
            <BeforeAfterPanel viewModel={viewModel} />
          </div>
        </section>
      </div>
    </TooltipProvider>
  );
};

const FeatureNotes = ({
  viewModel,
}: {
  viewModel: ReturnType<typeof buildFeatureStageViewModel>;
}) => (
  <section className="glass-card border-border/60 p-5">
    <div>
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Learning View</p>
        <p className="mt-1 text-sm text-secondary-foreground">
          Beginner-friendly notes that explain what changed during feature engineering.
        </p>
      </div>
    </div>

    <div className="mt-5 grid gap-3 lg:grid-cols-3">
      <NoteCard
        title="What happened"
        icon={BrainCircuit}
        body={viewModel.notes.whatHappened}
        accent="border-primary/20 bg-primary/10"
      />
      <NoteCard
        title="Why it mattered"
        icon={Radar}
        body={viewModel.notes.whyItMattered}
        accent="border-accent/20 bg-accent/10"
      />
      <NoteCard
        title="Key takeaway"
        icon={GraduationCap}
        body={viewModel.notes.keyTakeaway}
        accent="border-sky-400/20 bg-sky-400/10"
      />
    </div>
  </section>
);

const SelectedFeaturesPanel = ({ features }: { features: string[] }) => (
  <section className="glass-card border-border/60 p-5">
    <div className="flex items-center justify-between gap-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Final Selected Features</p>
        <p className="mt-1 text-sm text-secondary-foreground">
          These are the features that made it into the final training set after engineering and filtering.
        </p>
      </div>
      <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs text-primary">
        {features.length} kept
      </span>
    </div>

    {features.length > 0 ? (
      <div className="mt-4 flex flex-wrap gap-2">
        {features.map((feature) => (
          <Tooltip key={feature}>
            <TooltipTrigger asChild>
              <div className="rounded-full border border-border/60 bg-background/35 px-3 py-2 text-sm text-foreground transition hover:border-primary/25 hover:bg-background/45">
                {formatFeatureLabel(feature)}
              </div>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-xs">
              <p className="font-mono text-[11px]">{feature}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    ) : (
      <div className="mt-4 rounded-2xl border border-dashed border-border/60 p-5 text-sm text-muted-foreground">
        Final selected features will appear here after the feature engineering stage completes.
      </div>
    )}
  </section>
);

const NoteCard = ({
  title,
  body,
  icon: Icon,
  accent,
}: {
  title: string;
  body: string;
  icon: typeof BrainCircuit;
  accent: string;
}) => (
  <div className={`rounded-2xl border p-4 ${accent}`}>
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4 text-foreground/80" />
      <p className="text-sm font-semibold text-foreground">{title}</p>
    </div>
    <p className="mt-3 text-sm leading-7 text-secondary-foreground">{body}</p>
  </div>
);

const TopFeaturesSection = ({ features }: { features: FeatureInsightCard[] }) => (
  <section className="glass-card border-border/60 p-5" data-chat-context-label="Top features">
    <div className="flex items-center justify-between gap-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Top Features</p>
        <p className="mt-1 text-sm text-secondary-foreground">
          The strongest features from this run, shown with plain-English labels and importance scores.
        </p>
      </div>
      <span className="rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent">
        Importance view
      </span>
    </div>

    {features.length > 0 ? (
      <div className="mt-5 space-y-3">
        {features.map((feature, index) => {
          const strongest = features[0]?.score ?? 1;
          const width = strongest > 0 ? Math.max((feature.score / strongest) * 100, 12) : 12;
          return (
            <div
              key={feature.rawName}
              className="rounded-2xl border border-border/60 bg-background/30 p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <FriendlyFeatureLabel rawName={feature.rawName} />
                  <p className="mt-2 text-sm leading-6 text-secondary-foreground">{feature.meaning}</p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="font-mono text-sm text-foreground/80">{feature.score.toFixed(3)}</p>
                  <p className="mt-1 text-[11px] text-muted-foreground">{feature.type}</p>
                </div>
              </div>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-secondary/80">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-primary via-sky-400 to-accent"
                  style={{ width: `${width}%` }}
                />
              </div>
              <div className="mt-3 flex items-center justify-between gap-3">
                <span className="rounded-full border border-border/60 px-2.5 py-1 text-[11px] text-muted-foreground">
                  {feature.badge}
                </span>
                <span className="text-[11px] text-muted-foreground">Rank #{index + 1}</span>
              </div>
            </div>
          );
        })}
      </div>
    ) : (
      <div className="mt-5 rounded-2xl border border-dashed border-border/60 p-6 text-sm text-muted-foreground">
        Feature importance will appear once the feature engineering stage returns scores.
      </div>
    )}
  </section>
);

const BeforeAfterPanel = ({ viewModel }: { viewModel: ReturnType<typeof buildFeatureStageViewModel> }) => (
  <section className="glass-card border-border/60 p-5">
    <div className="flex items-center gap-2">
      <Sparkles className="h-4 w-4 text-primary" />
      <div>
        <p className="text-sm font-semibold text-foreground">Before vs After</p>
        <p className="text-sm text-secondary-foreground">
          A quick visual summary of how the feature space changed during engineering.
        </p>
      </div>
    </div>

    <div className="mt-5 grid gap-3 sm:grid-cols-4">
      {[
        { label: "Before", value: viewModel.originalCount },
        { label: "Generated", value: viewModel.generatedCount },
        { label: "Removed", value: viewModel.droppedCount },
        { label: "After", value: viewModel.selectedCount },
      ].map((item) => (
        <div key={item.label} className="rounded-2xl border border-border/60 bg-background/35 p-4">
          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{item.label}</p>
          <p className="mt-2 text-2xl font-semibold text-foreground">{item.value}</p>
        </div>
      ))}
    </div>

    <div className="mt-4 rounded-2xl border border-border/60 bg-background/35 p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Example transformations</p>
      <div className="mt-3 grid gap-2">
        {viewModel.exampleTransformations.map((example) => (
          <div key={example} className="rounded-xl border border-border/50 bg-background/30 px-3 py-2 text-sm text-secondary-foreground">
            {example}
          </div>
        ))}
      </div>
    </div>
  </section>
);

const DroppedFeaturesSection = ({ viewModel }: { viewModel: ReturnType<typeof buildFeatureStageViewModel> }) => (
  <section className="glass-card border-border/60 p-5" data-chat-context-label="Dropped features">
    <div className="flex items-center gap-2">
      <Filter className="h-4 w-4 text-amber-300" />
      <div>
        <p className="text-sm font-semibold text-foreground">Why features were removed</p>
        <p className="text-sm text-secondary-foreground">
          Removed features are grouped by the reason they left the final feature set.
        </p>
      </div>
    </div>

    {viewModel.droppedGroups.length > 0 ? (
      <div className="mt-5 space-y-4">
        {viewModel.droppedGroups.map((group) => (
          <div key={group.id} className="rounded-2xl border border-border/60 bg-background/35 p-4">
            <div className="flex items-center justify-between gap-3">
              <p className={`text-sm font-semibold ${group.tone}`}>{group.label}</p>
              <span className="rounded-full border border-border/60 px-2.5 py-1 text-[11px] text-muted-foreground">
                {group.features.length} features
              </span>
            </div>
            <div className="mt-4 grid gap-3">
              {group.features.map((feature) => (
                <div key={feature.rawName} className="rounded-xl border border-border/50 bg-background/30 p-3">
                  <FriendlyFeatureLabel rawName={feature.rawName} />
                  <p className="mt-2 text-sm leading-6 text-secondary-foreground">{feature.reason}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    ) : (
      <div className="mt-5 rounded-2xl border border-dashed border-border/60 p-6 text-sm text-muted-foreground">
        No dropped features were reported for this run.
      </div>
    )}
  </section>
);

const FriendlyFeatureLabel = ({ rawName }: { rawName: string }) => (
  <Tooltip>
    <TooltipTrigger asChild>
      <div className="group min-w-0">
        <p className="line-clamp-1 text-sm font-semibold text-foreground">{formatFeatureLabel(rawName)}</p>
        <p className="mt-1 line-clamp-1 font-mono text-[11px] text-muted-foreground/80 group-hover:text-muted-foreground">
          {rawName}
        </p>
      </div>
    </TooltipTrigger>
    <TooltipContent side="top" className="max-w-xs">
      <p className="font-mono text-[11px]">{rawName}</p>
    </TooltipContent>
  </Tooltip>
);

const readSelectedFeatures = (stageResult: FeatureStageResultLike | null): string[] =>
  Array.isArray(stageResult?.selected_features)
    ? stageResult.selected_features.filter((feature): feature is string => typeof feature === "string")
    : [];

export default FeatureEngineeringDashboard;
