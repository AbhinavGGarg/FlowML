import type { MetricsResponse, TaskType } from "@/lib/api";

export interface FeatureStageNarrative {
  stageSummary?: string;
  whatHappened?: string;
  whyItMattered?: string;
  keyTakeaway?: string;
  featureExplanations?: Record<string, string>;
  droppedFeatureExplanations?: Record<string, string>;
  llmUsed?: boolean;
}

export interface FeatureStageResultLike extends Record<string, unknown> {
  original_feature_count?: number;
  post_engineering_feature_count?: number;
  final_feature_count?: number;
  selected_features?: string[];
  generated_features?: string[];
  dropped_columns?: string[];
  dropped_features?: string[];
  feature_scores?: Record<string, number>;
  applied_transformations?: Array<Record<string, unknown>>;
  numeric_features?: string[];
  categorical_features?: string[];
  pca_result?: Record<string, unknown> | null;
  feature_engineering_config?: Record<string, unknown>;
  llm_explanations?: FeatureStageNarrative;
}

export interface FeatureInsightCard {
  rawName: string;
  label: string;
  score: number;
  type: string;
  badge: string;
  meaning: string;
}

export interface DroppedFeatureGroup {
  id: string;
  label: string;
  tone: string;
  features: Array<{
    rawName: string;
    label: string;
    reason: string;
  }>;
}

export interface FeatureStageViewModel {
  summary: string;
  summarySource: "llm" | "fallback";
  originalCount: number;
  generatedCount: number;
  droppedCount: number;
  selectedCount: number;
  topFeatures: FeatureInsightCard[];
  droppedGroups: DroppedFeatureGroup[];
  notes: {
    whatHappened: string;
    whyItMattered: string;
    keyTakeaway: string;
  };
  exampleTransformations: string[];
  technicalLogs: string[];
  taskType: TaskType;
  targetColumn: string | null;
}

const FRIENDLY_LABELS: Record<string, string> = {
  age: "Age",
  carat: "Carat",
  cut: "Cut Quality",
  color: "Color Grade",
  clarity: "Clarity Grade",
  depth: "Depth",
  table: "Table Width",
  price: "Price",
  x: "Length",
  y: "Width",
  z: "Depth",
  "educational-num": "Education Level",
  education_num: "Education Level",
  "hours-per-week": "Hours Per Week",
  hours_per_week: "Hours Per Week",
  "capital-gain": "Capital Gain",
  capital_gain: "Capital Gain",
  "capital-loss": "Capital Loss",
  capital_loss: "Capital Loss",
};

const reasonLabels: Record<string, { label: string; tone: string }> = {
  feature_importance_filter: { label: "Low importance", tone: "text-amber-300" },
  correlation_filter: { label: "High correlation", tone: "text-sky-300" },
  selection_cap: { label: "Selection cap", tone: "text-violet-300" },
  drop_index_like_columns: { label: "Other filtering", tone: "text-zinc-300" },
};

export function buildFeatureStageViewModel(args: {
  stageResult: FeatureStageResultLike | null;
  stageLogs: string[];
  metrics: MetricsResponse | null;
  taskType: TaskType;
  targetColumn: string | null;
}): FeatureStageViewModel {
  const { stageResult, stageLogs, metrics, taskType, targetColumn } = args;
  const result = stageResult ?? {};
  const config = (result.feature_engineering_config as Record<string, unknown> | undefined) ?? {};
  const generatedFeatures = readStringArray(result.generated_features);
  const selectedFeatures = readStringArray(result.selected_features);
  const droppedFeatures = readStringArray(result.dropped_columns ?? result.dropped_features);
  const numericFeatures = new Set(readStringArray(result.numeric_features));
  const categoricalFeatures = new Set(readStringArray(result.categorical_features));
  const featureScores = sortFeatureScores(result.feature_scores as Record<string, number> | undefined);
  const transformations = Array.isArray(result.applied_transformations)
    ? (result.applied_transformations as Array<Record<string, unknown>>)
    : [];
  const llm = (result.llm_explanations as FeatureStageNarrative | undefined) ?? {};

  const originalCount = readNumber(result.original_feature_count)
    ?? readNumber(config.original_feature_count)
    ?? Math.max(selectedFeatures.length + droppedFeatures.length, 0);
  const selectedCount = readNumber(result.final_feature_count) ?? selectedFeatures.length;
  const droppedCount = droppedFeatures.length;

  const topFeatures = featureScores
    .slice(0, 8)
    .map(([rawName, score]) =>
      buildFeatureInsight({
        rawName,
        score,
        generatedFeatures,
        numericFeatures,
        categoricalFeatures,
        llm,
      }),
    );

  const droppedGroups = groupDroppedFeatures({
    droppedFeatures,
    transformations,
    generatedFeatures,
    taskType,
    llm,
  });

  const summary =
    llm.stageSummary ??
    `We started with ${originalCount} original features, explored ${generatedFeatures.length} engineered signals, removed ${droppedCount} weaker or redundant features, and kept ${selectedCount} features that looked most useful for this ${taskType} task.`;

  const notes = {
    whatHappened:
      llm.whatHappened ??
      `We started with ${originalCount} original features, created ${generatedFeatures.length} extra feature signals, then filtered the set down to ${selectedCount} final features that looked most useful for prediction.`,
    whyItMattered:
      llm.whyItMattered ??
      buildWhyItMattered({
        taskType,
        generatedCount: generatedFeatures.length,
        droppedCount,
      }),
    keyTakeaway:
      llm.keyTakeaway ??
      "Feature engineering turns raw columns into a cleaner learning space so the model sees stronger patterns and less noise.",
  };

  return {
    summary,
    summarySource: llm.llmUsed ? "llm" : "fallback",
    originalCount,
    generatedCount: generatedFeatures.length,
    droppedCount,
    selectedCount,
    topFeatures,
    droppedGroups,
    notes,
    exampleTransformations: buildExampleTransformations(transformations, generatedFeatures, selectedFeatures),
    technicalLogs: stageLogs,
    taskType: metrics?.task_type ?? taskType,
    targetColumn,
  };
}

export function formatFeatureLabel(rawName: string): string {
  if (rawName.includes("__mul__")) {
    return rawName
      .split("__mul__")
      .map((part) => formatFeatureLabel(part))
      .join(" x ");
  }
  if (rawName.includes("__div__")) {
    return rawName
      .split("__div__")
      .map((part) => formatFeatureLabel(part))
      .join(" / ");
  }
  if (rawName.startsWith("PC")) {
    return rawName.replace("PC", "Principal Component ");
  }

  if (FRIENDLY_LABELS[rawName]) {
    return FRIENDLY_LABELS[rawName];
  }

  return rawName
    .replace(/[_-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => FRIENDLY_LABELS[word] ?? `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function buildFeatureInsight(args: {
  rawName: string;
  score: number;
  generatedFeatures: string[];
  numericFeatures: Set<string>;
  categoricalFeatures: Set<string>;
  llm: FeatureStageNarrative;
}): FeatureInsightCard {
  const { rawName, score, generatedFeatures, numericFeatures, categoricalFeatures, llm } = args;
  const isEngineered = generatedFeatures.includes(rawName) || rawName.includes("__");
  const type = isEngineered
    ? "Engineered"
    : numericFeatures.has(rawName)
      ? "Numeric"
      : categoricalFeatures.has(rawName)
        ? "Categorical"
        : rawName.startsWith("PC")
          ? "PCA"
          : "Feature";

  return {
    rawName,
    label: formatFeatureLabel(rawName),
    score,
    type,
    badge: isEngineered ? "Engineered feature" : score >= 0.15 ? "Strong predictor" : "Useful signal",
    meaning: llm.featureExplanations?.[rawName] ?? defaultFeatureMeaning(rawName),
  };
}

function defaultFeatureMeaning(rawName: string): string {
  if (rawName.includes("__mul__")) {
    const [left, right] = rawName.split("__mul__");
    return `${formatFeatureLabel(left)} and ${formatFeatureLabel(right)} are multiplied so the model can notice when both rise together.`;
  }
  if (rawName.includes("__div__")) {
    const [left, right] = rawName.split("__div__");
    return `${formatFeatureLabel(left)} is divided by ${formatFeatureLabel(right)} so the model can compare one measurement relative to another.`;
  }
  if (rawName.startsWith("PC")) {
    return `${formatFeatureLabel(rawName)} is a compressed summary of several numeric features.`;
  }
  return `${formatFeatureLabel(rawName)} is one of the original inputs from the dataset.`;
}

function groupDroppedFeatures(args: {
  droppedFeatures: string[];
  transformations: Array<Record<string, unknown>>;
  generatedFeatures: string[];
  taskType: TaskType;
  llm: FeatureStageNarrative;
}): DroppedFeatureGroup[] {
  const { droppedFeatures, transformations, generatedFeatures, taskType, llm } = args;
  const grouped = new Map<string, DroppedFeatureGroup>();

  const droppedReasonLookup = new Map<string, string>();
  transformations.forEach((transformation) => {
    const type = String(transformation.type ?? "other");
    const columns = readStringArray(transformation.columns);
    columns.forEach((column) => droppedReasonLookup.set(column, type));
  });

  droppedFeatures.forEach((rawName) => {
    const reasonKey = droppedReasonLookup.get(rawName) ?? "other";
    const labelConfig = reasonLabels[reasonKey] ?? { label: "Other filtering", tone: "text-zinc-300" };
    if (!grouped.has(reasonKey)) {
      grouped.set(reasonKey, {
        id: reasonKey,
        label: labelConfig.label,
        tone: labelConfig.tone,
        features: [],
      });
    }

    grouped.get(reasonKey)?.features.push({
      rawName,
      label: formatFeatureLabel(rawName),
      reason:
        llm.droppedFeatureExplanations?.[rawName]
        ?? defaultDroppedReason(rawName, reasonKey, generatedFeatures.includes(rawName), taskType),
    });
  });

  return Array.from(grouped.values()).filter((group) => group.features.length > 0);
}

function defaultDroppedReason(
  rawName: string,
  reasonKey: string,
  isGenerated: boolean,
  taskType: TaskType,
): string {
  if (reasonKey === "feature_importance_filter") {
    return `${formatFeatureLabel(rawName)} added very little extra signal for this ${taskType} problem, so it was removed to keep the model simpler.`;
  }
  if (reasonKey === "correlation_filter") {
    return `${formatFeatureLabel(rawName)} told a very similar story to another feature, so it was removed to reduce redundancy.`;
  }
  if (reasonKey === "selection_cap") {
    return `${formatFeatureLabel(rawName)} was decent, but it did not make the final cut after ranking the strongest signals.`;
  }
  if (reasonKey === "drop_index_like_columns") {
    return `${formatFeatureLabel(rawName)} looked more like an ID or row label than a real learning signal.`;
  }
  return isGenerated
    ? `${formatFeatureLabel(rawName)} was a generated experiment that did not help enough to justify the extra complexity.`
    : `${formatFeatureLabel(rawName)} was filtered out during cleanup.`;
}

function buildWhyItMattered(args: {
  taskType: TaskType;
  generatedCount: number;
  droppedCount: number;
}): string {
  const { taskType, generatedCount, droppedCount } = args;
  const taskPhrase = taskType === "classification"
    ? "separating the target classes"
    : "predicting the target value";

  const interactionNote = generatedCount > 0
    ? "The engineered interactions gave the model a chance to learn feature relationships, not just single-column effects."
    : "This run leaned more on the original features than on new interactions.";

  return `Good feature engineering makes ${taskPhrase} easier. ${interactionNote} Dropping ${droppedCount} weaker or repetitive features also helps the model focus on the strongest clues.`;
}

function buildExampleTransformations(
  transformations: Array<Record<string, unknown>>,
  generatedFeatures: string[],
  selectedFeatures: string[],
): string[] {
  const examples: string[] = [];

  transformations.forEach((transformation) => {
    const type = String(transformation.type ?? "");
    if (type === "numeric_imputation") {
      examples.push("Filled missing numeric values with a median so gaps would not confuse the model.");
    }
    if (type === "categorical_imputation") {
      examples.push("Filled missing categories with the most common value to keep those rows usable.");
    }
    if (type === "log1p_transform") {
      examples.push("Compressed skewed numeric features so extreme values had less outsized influence.");
    }
    if (type === "correlation_filter") {
      examples.push("Removed highly similar numeric features so the final feature set stayed cleaner.");
    }
  });

  if (generatedFeatures.length > 0) {
    examples.push(`Generated interaction examples like ${formatFeatureLabel(generatedFeatures[0])}.`);
  }
  if (selectedFeatures.length > 0) {
    examples.push(`The final set kept signals like ${formatFeatureLabel(selectedFeatures[0])}.`);
  }

  return examples.slice(0, 4);
}

function sortFeatureScores(featureScores?: Record<string, number>): Array<[string, number]> {
  if (!featureScores) return [];
  return Object.entries(featureScores)
    .map(([name, value]) => [name, Number(value)] as [string, number])
    .filter(([, value]) => !Number.isNaN(value))
    .sort((a, b) => b[1] - a[1]);
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && !Number.isNaN(value) ? value : null;
}
