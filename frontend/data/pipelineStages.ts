export type StageStatus = "waiting" | "running" | "completed" | "failed";

export interface PipelineStage {
  id: string;
  label: string;
  icon: string;
  iconPath: string;
  description: string;
  details: string;
  tooltipPoints: string[];
  vizType: "heatmap" | "table" | "barChart" | "lossCurve" | "confusionMatrix" | "metrics" | "modelSelection";
  codeSnippet: string;
}

export const stages: PipelineStage[] = [
  {
    id: "analysis",
    label: "Analysis",
    icon: "🔎",
    iconPath: "/stage-icons/analyze.png",
    description: "Inspect dataset structure, quality, and feature relationships.",
    details:
      "The analysis stage profiles the uploaded dataset, scans for missing values and outliers, and highlights correlations that matter for modeling decisions.",
    tooltipPoints: [
      "Profiles numerical and categorical columns",
      "Flags missing values and outliers",
      "Builds an initial correlation heatmap",
    ],
    vizType: "heatmap",
    codeSnippet: `import pandas as pd

df = pd.read_csv("housing_prices.csv")
summary = df.describe(include="all")
missing = df.isna().mean().sort_values(ascending=False)
correlation = df.corr(numeric_only=True)
print(summary)
print(missing.head())`,
  },
  {
    id: "preprocessing",
    label: "Preprocess",
    icon: "🧼",
    iconPath: "/stage-icons/preprocess.png",
    description: "Clean raw data and prepare a model-ready feature matrix.",
    details:
      "This stage imputes missing values, encodes categorical columns, scales numeric features, and splits the dataset into training and evaluation sets.",
    tooltipPoints: [
      "Imputes sparse missing values",
      "Encodes categorical features",
      "Scales and splits the dataset",
    ],
    vizType: "table",
    codeSnippet: `from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])`,
  },
  {
    id: "features",
    label: "Features",
    icon: "🧠",
    iconPath: "/stage-icons/features.png",
    description: "Select and engineer the features that best explain the target.",
    details:
      "Feature engineering cleans raw columns, creates a small set of smarter interactions, removes redundant signals, and ranks the final features before model training.",
    tooltipPoints: [
      "Cleans and reshapes the feature space",
      "Creates bounded interaction features",
      "Keeps the strongest final predictors",
    ],
    vizType: "barChart",
    codeSnippet: `numeric_columns = X.select_dtypes(include="number").columns
top_numeric = X[numeric_columns].var().sort_values(ascending=False).head(5).index

for left, right in combinations(top_numeric, 2):
    X[f"{left}__mul__{right}"] = X[left] * X[right]
    X[f"{left}__div__{right}"] = (X[left] / X[right].replace(0, np.nan)).fillna(0)

encoded_X = encode_categorical_columns(X)
forest = RandomForestRegressor(n_estimators=50, random_state=42)
forest.fit(encoded_X, y)

feature_scores = dict(zip(encoded_X.columns, forest.feature_importances_))
selected_features = [name for name, score in feature_scores.items() if score >= 0.01]`,
  },
  {
    id: "model_selection",
    label: "Model Select",
    icon: "🧭",
    iconPath: "/stage-icons/modelselect.png",
    description: "Choose the best model family and sensible hyperparameters.",
    details:
      "Model selection balances dataset size, feature mix, and risk signals to propose a reliable estimator before training begins.",
    tooltipPoints: [
      "Evaluates candidate model families",
      "Selects the most reliable baseline",
      "Tunes safe starter hyperparameters",
    ],
    vizType: "modelSelection",
    codeSnippet: `candidates = ["RandomForest", "GradientBoosting", "LogisticRegression"]
selected = candidates[0]
hyperparameters = {"n_estimators": 100, "max_depth": 10}
print("Selected model:", selected)
print("Hyperparameters:", hyperparameters)`,
  },
  {
    id: "training",
    label: "Model Training",
    icon: "🤖",
    iconPath: "/stage-icons/training.png",
    description: "Train the best-performing model on the prepared data.",
    details:
      "In this step, the system trains a machine learning model using the prepared dataset. The model learns patterns that help predict the target variable. Different models may be evaluated, and the one that performs best is selected automatically.",
    tooltipPoints: [
      "Evaluates multiple models",
      "Trains on the prepared data",
      "Reports cross-validation performance",
    ],
    vizType: "lossCurve",
    codeSnippet: `from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
)
model.fit(X_train, y_train)`,
  },
  {
    id: "loss",
    label: "Loss",
    icon: "📉",
    iconPath: "/stage-icons/loss.png",
    description: "Compare training and validation loss to spot drift or overfitting.",
    details:
      "Loss monitoring gives quick feedback on whether the model is learning generalizable structure or memorizing the training set.",
    tooltipPoints: [
      "Plots train and validation curves",
      "Highlights divergence between the curves",
      "Helps identify overfitting early",
    ],
    vizType: "lossCurve",
    codeSnippet: `history = {
    "train_loss": [0.90, 0.65, 0.45, 0.32, 0.22, 0.16, 0.12, 0.09],
    "val_loss": [0.92, 0.70, 0.52, 0.40, 0.33, 0.29, 0.27, 0.26],
}

best_epoch = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
print("best_epoch", best_epoch)`,
  },
  {
    id: "evaluation",
    label: "Evaluation",
    icon: "🎯",
    iconPath: "/stage-icons/evaluation.png",
    description: "Measure model quality, reliability, and deployment readiness on held-out data.",
    details:
      "Evaluation checks how well the model performs on unseen examples, compares it with simple baselines, and helps you judge whether the model generalizes well enough to trust.",
    tooltipPoints: [
      "Runs predictions on the test split",
      "Compares train, test, and cross-validation behavior",
      "Summarizes whether the model looks deployment-ready",
    ],
    vizType: "confusionMatrix",
    codeSnippet: `# The Evaluation panel now shows task-aware code automatically.
# Classification runs show accuracy, F1, ROC-AUC, and confusion matrix examples.
# Regression runs show R2, MAE, MSE, and RMSE examples.`,
  },
  {
    id: "results",
    label: "Results",
    icon: "📦",
    iconPath: "/stage-icons/results.png",
    description: "Package the model artifacts and summary metrics.",
    details:
      "The final stage bundles the trained model, configuration, and evaluation outputs into assets that can be downloaded or reused later.",
    tooltipPoints: [
      "Persists the trained model",
      "Exports a metrics report",
      "Captures the final pipeline configuration",
    ],
    vizType: "metrics",
    codeSnippet: `import joblib
from pathlib import Path

output_dir = Path("artifacts")
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "model.pkl")
(output_dir / "metrics.txt").write_text(report)`,
  },
];
