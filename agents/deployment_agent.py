"""Deployment Agent for AutoML Pipeline.

This agent handles model deployment including:
- Saving model to disk
- Assembling a complete, self-contained deployment package (zip) containing:
    app.py            FastAPI predict API with schema-aware request validation
    schema.json       Input schema (columns, types, ranges, categories, preprocessing stats)
    model.pkl         The trained model artifact
    requirements.txt  Pinned serving dependencies
    Dockerfile        Reproducible container build
    docker-compose.yml  Local one-command testing
    README.md         LLM-generated (with deterministic fallback) model card
"""

import json
import zipfile
from importlib.metadata import version as _pkg_version
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from agents.report_generator import ReportGenerator
from core.exceptions import AgentExecutionError

# Ensure outputs directory exists
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Packages we want to pin in the generated requirements.txt
_SERVING_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "joblib",
    "python-multipart",
]


class DeploymentAgent(BaseAgent):
    """Agent for deploying trained models.

    Produces two outputs:
    1. model.pkl + metadata.json in /outputs (existing behaviour)
    2. A self-contained deployment zip package in /outputs containing
       everything needed to run the model as a production REST API.
    """

    def __init__(self) -> None:
        """Initialize the DeploymentAgent."""
        super().__init__("Deployment")

    # ── Public entry point ──────────────────────────────────────────────────

    async def execute(
        self,
        training_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        pipeline_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        analysis_result: Optional[dict[str, Any]] = None,
        preprocessing_result: Optional[dict[str, Any]] = None,
        features_result: Optional[dict[str, Any]] = None,
        model_selection_result: Optional[dict[str, Any]] = None,
        explanation_result: Optional[dict[str, Any]] = None,
        raw_dataset: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ) -> dict[str, Any]:
        """Deploy the trained model and build the deployment package.

        Args:
            training_result: Output from TrainingAgent.
            evaluation_result: Output from EvaluationAgent.
            pipeline_id: Unique identifier for this pipeline run.
            analysis_result: Output from DataAnalyzerAgent (optional).
            preprocessing_result: Output from PreprocessorAgent (optional).
            features_result: Output from FeatureEngineeringAgent (optional).
            explanation_result: Output from ExplanationGeneratorAgent (optional).
            raw_dataset: The original uploaded DataFrame used to compute
                preprocessing statistics for schema.json.
            target_column: Name of the target column to exclude from schema.

        Returns:
            dict with model_path, metadata_path, deployment_code, package_path,
            package_ready, and pipeline_id.
        """
        try:
            try:
                import joblib
            except ModuleNotFoundError as exc:
                raise AgentExecutionError(
                    "joblib is required to export trained models. "
                    "Install backend dependencies with 'pip install -r requirements.txt'.",
                    agent_name=self.name,
                ) from exc

            model = training_result.get("model")
            model_name = training_result.get("model_name", "model")
            task_type = evaluation_result.get("task_type", "classification")
            pipeline_id = pipeline_id or "pipeline"

            if model is None:
                raise AgentExecutionError(
                    "No model available for deployment",
                    agent_name=self.name,
                )

            # ── 1. Save model artifact ──────────────────────────────────────
            model_path = OUTPUTS_DIR / f"{pipeline_id}_model.pkl"
            joblib.dump(model, model_path)

            # ── 2. Save metadata ────────────────────────────────────────────
            metadata = {
                "pipeline_id": pipeline_id,
                "model_name": model_name,
                "task_type": task_type,
                "accuracy": evaluation_result.get("accuracy"),
                "f1": evaluation_result.get("f1"),
                "r2": evaluation_result.get("r2"),
                "deployment_decision": evaluation_result.get("deployment_decision"),
            }
            metadata_path = OUTPUTS_DIR / f"{pipeline_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # ── 3. Build input schema ───────────────────────────────────────
            input_schema = self._build_input_schema(
                analysis_result=analysis_result or {},
                preprocessing_result=preprocessing_result or {},
                features_result=features_result or {},
                training_result=training_result,
                evaluation_result=evaluation_result,
                raw_dataset=raw_dataset,
                target_column=target_column,
            )

            # ── 4. Assemble package zip ─────────────────────────────────────
            report_generator = ReportGenerator()
            report_assets = report_generator.generate_assets(
                pipeline_id=pipeline_id,
                dataset_name=dataset_name,
                target_column=target_column,
                analysis_result=analysis_result or {},
                preprocessing_result=preprocessing_result or {},
                features_result=features_result or {},
                model_selection_result=model_selection_result or {},
                training_result=training_result,
                evaluation_result=evaluation_result,
                evaluation_insights=evaluation_result.get("llm_insights") if isinstance(evaluation_result, dict) else None,
                explanation_result=explanation_result or {},
            )

            report_html_path = OUTPUTS_DIR / f"{pipeline_id}_report.html"
            report_html_path.write_text(str(report_assets.get("html", "")), encoding="utf-8")

            package_path = self._build_package_zip(
                pipeline_id=pipeline_id,
                model_path=model_path,
                input_schema=input_schema,
                model_name=model_name,
                task_type=task_type,
                evaluation_result=evaluation_result,
                explanation_result=explanation_result or {},
                training_result=training_result,
                report_html=str(report_assets.get("html") or ""),
            )

            # ── 5. Legacy deployment_code for API compatibility ─────────────
            deployment_code = self._generate_deployment_code(model_name, task_type)

            return {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "deployment_code": deployment_code,
                "pipeline_id": pipeline_id,
                "deployment_success": True,
                "package_path": str(package_path),
                "package_ready": True,
                "report_path": str(report_html_path),
                "report_ready": True,
            }

        except Exception as e:
            raise AgentExecutionError(
                f"Deployment failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    # ── Schema builder ──────────────────────────────────────────────────────

    def _build_input_schema(
        self,
        analysis_result: dict[str, Any],
        preprocessing_result: dict[str, Any],
        features_result: dict[str, Any],
        training_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        raw_dataset: Optional[pd.DataFrame],
        target_column: Optional[str],
    ) -> dict[str, Any]:
        """Build a normalised input schema consumed by the generated app.py."""
        numeric_cols: list[str] = preprocessing_result.get("numeric_columns", [])
        categorical_cols: list[str] = preprocessing_result.get("categorical_columns", [])
        encoding_mapping: dict[str, list] = preprocessing_result.get("encoding_mapping", {})

        # Fallback to analysis columns if preprocessing didn't run
        if not numeric_cols and not categorical_cols and analysis_result:
            numeric_cols = analysis_result.get("numeric_columns", [])
            categorical_cols = analysis_result.get("categorical_columns", [])

        # Exclude target from inputs
        if target_column:
            numeric_cols = [c for c in numeric_cols if c != target_column]
            categorical_cols = [c for c in categorical_cols if c != target_column]

        required_columns = numeric_cols + categorical_cols
        column_types = (
            {c: "numeric" for c in numeric_cols}
            | {c: "categorical" for c in categorical_cols}
        )

        train_medians: dict[str, float] = {}
        train_means: dict[str, float] = {}
        train_stds: dict[str, float] = {}
        column_ranges: dict[str, dict[str, float]] = {}
        allowed_categories: dict[str, list[str]] = {}

        if raw_dataset is not None and not raw_dataset.empty:
            df_feat = (
                raw_dataset.drop(columns=[target_column], errors="ignore")
                if target_column else raw_dataset
            )
            for col in numeric_cols:
                if col in df_feat.columns:
                    series = pd.to_numeric(df_feat[col], errors="coerce").dropna()
                    if len(series):
                        train_medians[col] = float(series.median())
                        train_means[col] = float(series.mean())
                        train_stds[col] = float(series.std()) if series.std() > 0 else 1.0
                        column_ranges[col] = {
                            "min": float(series.min()),
                            "max": float(series.max()),
                        }
            for col in categorical_cols:
                if col in df_feat.columns:
                    cats = df_feat[col].dropna().astype(str).unique().tolist()
                    allowed_categories[col] = cats
        else:
            # Fallback: derive allowed cats from encoding_mapping
            for col, cats in encoding_mapping.items():
                if col in categorical_cols:
                    allowed_categories[col] = [str(c) for c in cats]

        enc_map_serialisable: dict[str, list[str]] = {
            col: [str(c) for c in cats]
            for col, cats in encoding_mapping.items()
        }

        # Feature order the trained model expects
        feature_order: list[str] = training_result.get("feature_names", [])
        trained_model = training_result.get("model")
        if not feature_order and trained_model is not None:
            # sklearn estimators
            model_feature_names = getattr(trained_model, "feature_names_in_", None)
            if model_feature_names is not None:
                feature_order = [str(name) for name in model_feature_names]
            else:
                # LightGBM sklearn wrapper often exposes `feature_name_`
                lgbm_feature_names = getattr(trained_model, "feature_name_", None)
                if isinstance(lgbm_feature_names, list):
                    feature_order = [str(name) for name in lgbm_feature_names]

        if not feature_order:
            feature_order = list(numeric_cols)
            for col in categorical_cols:
                cats = enc_map_serialisable.get(col, allowed_categories.get(col, []))
                feature_order.extend([f"{col}_{c}" for c in cats])

        return {
            "required_columns": required_columns,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "column_types": column_types,
            "allowed_categories": allowed_categories,
            "column_ranges": column_ranges,
            "train_medians": train_medians,
            "train_means": train_means,
            "train_stds": train_stds,
            "encoding_mapping": enc_map_serialisable,
            "feature_order": feature_order,
            "task_type": evaluation_result.get("task_type", "classification"),
            "target_column": target_column,
            "model_name": training_result.get("model_name", "model"),
        }

    # ── Package assembly ────────────────────────────────────────────────────

    def _build_package_zip(
        self,
        pipeline_id: str,
        model_path: Path,
        input_schema: dict[str, Any],
        model_name: str,
        task_type: str,
        evaluation_result: dict[str, Any],
        explanation_result: dict[str, Any],
        training_result: dict[str, Any],
        report_html: str,
    ) -> Path:
        """Assemble all package files into a zip and return its path."""
        zip_path = OUTPUTS_DIR / f"{pipeline_id}_deployment_package.zip"

        readme_text = self._generate_package_readme(
            model_name=model_name,
            task_type=task_type,
            input_schema=input_schema,
            evaluation_result=evaluation_result,
            explanation_result=explanation_result,
            pipeline_id=pipeline_id,
        )

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, arcname="model.pkl")
            zf.writestr("schema.json", json.dumps(input_schema, indent=2, default=str))
            zf.writestr("app.py", self._render_app_py(model_name, task_type))
            zf.writestr("requirements.txt", self._render_requirements_txt())
            zf.writestr("Dockerfile", self._render_dockerfile())
            zf.writestr("docker-compose.yml", self._render_docker_compose(pipeline_id))
            zf.writestr("README.md", readme_text)
            zf.writestr("report.html", report_html)

        return zip_path

    # ── File renderers ─────────────────────────────────────────────────────

    def _render_app_py(self, model_name: str, task_type: str) -> str:
        """Return the generated FastAPI prediction server source."""
        return dedent(f'''\
            """Prediction API for {model_name}.

            Auto-generated by FlowML AutoML.
            Task type : {task_type}

            Quick start
            -----------
            # Run locally
            uvicorn app:app --reload --port 8000

            # Or via Docker Compose (see docker-compose.yml)
            docker compose up --build
            """

            import json
            from pathlib import Path
            from typing import Any

            import joblib
            import numpy as np
            import pandas as pd
            from fastapi import FastAPI, HTTPException

            # ── Load artifacts ────────────────────────────────────────────────────────
            BASE_DIR = Path(__file__).parent
            model = joblib.load(BASE_DIR / "model.pkl")
            with open(BASE_DIR / "schema.json") as _f:
                SCHEMA: dict = json.load(_f)

            REQUIRED_COLUMNS:    list[str]        = SCHEMA["required_columns"]
            NUMERIC_COLUMNS:     list[str]        = SCHEMA["numeric_columns"]
            CATEGORICAL_COLUMNS: list[str]        = SCHEMA["categorical_columns"]
            COLUMN_TYPES:        dict[str, str]   = SCHEMA["column_types"]
            ALLOWED_CATEGORIES:  dict[str, list]  = SCHEMA.get("allowed_categories", {{}})
            COLUMN_RANGES:       dict[str, dict]  = SCHEMA.get("column_ranges", {{}})
            TRAIN_MEDIANS:       dict[str, float] = SCHEMA.get("train_medians", {{}})
            TRAIN_MEANS:         dict[str, float] = SCHEMA.get("train_means", {{}})
            TRAIN_STDS:          dict[str, float] = SCHEMA.get("train_stds", {{}})
            ENCODING_MAPPING:    dict[str, list]  = SCHEMA.get("encoding_mapping", {{}})
            FEATURE_ORDER:       list[str]        = SCHEMA.get("feature_order", [])
            TASK_TYPE:           str              = SCHEMA.get("task_type", "classification")

            app = FastAPI(
                title="{model_name} Prediction API",
                description="Auto-generated inference API",
                version="1.0.0",
            )

            # ── Validation ────────────────────────────────────────────────────────────

            def validate_input(data: dict[str, Any]) -> list[str]:
                """Return a list of validation error messages (empty means valid)."""
                errors: list[str] = []
                for col in REQUIRED_COLUMNS:
                    if col not in data:
                        errors.append(f"Missing required field: {{col!r}}")
                for col, value in data.items():
                    if value is None:
                        continue
                    col_type = COLUMN_TYPES.get(col)
                    if col_type == "numeric":
                        try:
                            fval = float(value)
                        except (TypeError, ValueError):
                            errors.append(
                                f"Field {{col!r}} expects a numeric value, got {{type(value).__name__!r}}"
                            )
                            continue
                        rng = COLUMN_RANGES.get(col)
                        if rng:
                            lo, hi = rng.get("min"), rng.get("max")
                            if lo is not None and fval < lo:
                                errors.append(
                                    f"Field {{col!r}} value {{fval}} is below the expected minimum of {{lo}}"
                                )
                            if hi is not None and fval > hi:
                                errors.append(
                                    f"Field {{col!r}} value {{fval}} is above the expected maximum of {{hi}}"
                                )
                    elif col_type == "categorical":
                        allowed = ALLOWED_CATEGORIES.get(col)
                        if allowed and str(value) not in [str(a) for a in allowed]:
                            sample = [str(a) for a in allowed[:8]]
                            suffix = "..." if len(allowed) > 8 else ""
                            errors.append(
                                f"Field {{col!r}} received unexpected category {{value!r}}. "
                                f"Expected one of: {{sample}}{{suffix}}"
                            )
                return errors

            # ── Preprocessing ─────────────────────────────────────────────────────────

            def preprocess(data: dict[str, Any]) -> pd.DataFrame:
                """Apply train-time preprocessing and return a model-ready DataFrame.

                Notes:
                - Preserves raw categorical columns when the trained model expects them.
                - Adds one-hot columns only when FEATURE_ORDER indicates they are needed.
                - Recomputes simple engineered numeric interactions (e.g. a__div__b).
                """
                row: dict[str, Any] = {{}}
                for col in NUMERIC_COLUMNS:
                    val = data.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        num = float(TRAIN_MEDIANS.get(col, 0.0))
                    else:
                        num = float(val)
                    row[col] = num
                for col in CATEGORICAL_COLUMNS:
                    val_str = str(data.get(col, "")) if data.get(col) is not None else ""

                    # Keep raw categorical value when model expects raw feature names.
                    if not FEATURE_ORDER or col in FEATURE_ORDER:
                        row[col] = val_str

                    # Add one-hot columns only when FEATURE_ORDER explicitly expects them.
                    expects_one_hot = any(feat.startswith(f"{{col}}_") for feat in FEATURE_ORDER)
                    if expects_one_hot:
                        cats = ENCODING_MAPPING.get(col, ALLOWED_CATEGORIES.get(col, []))
                        for cat in cats:
                            row[f"{{col}}_{{cat}}"] = 1 if val_str == str(cat) else 0

                def _num(name: str) -> float | None:
                    value = row.get(name)
                    if value is None:
                        return None
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None

                # Rebuild simple engineered interaction features when expected.
                for feat in FEATURE_ORDER:
                    if feat in row:
                        continue
                    if "__div__" in feat:
                        left, right = feat.split("__div__", 1)
                        lv, rv = _num(left), _num(right)
                        if lv is not None and rv is not None:
                            row[feat] = 0.0 if abs(rv) < 1e-12 else lv / rv
                    elif "__mul__" in feat:
                        left, right = feat.split("__mul__", 1)
                        lv, rv = _num(left), _num(right)
                        if lv is not None and rv is not None:
                            row[feat] = lv * rv
                    elif "__add__" in feat:
                        left, right = feat.split("__add__", 1)
                        lv, rv = _num(left), _num(right)
                        if lv is not None and rv is not None:
                            row[feat] = lv + rv
                    elif "__sub__" in feat:
                        left, right = feat.split("__sub__", 1)
                        lv, rv = _num(left), _num(right)
                        if lv is not None and rv is not None:
                            row[feat] = lv - rv

                aligned = {{feat: row.get(feat, 0.0) for feat in FEATURE_ORDER}}
                if not FEATURE_ORDER:
                    aligned = row  # type: ignore[assignment]
                return pd.DataFrame([aligned])

            # ── Endpoints ──────────────────────────────────────────────────────────────

            @app.get("/")
            def root():
                return {{
                    "model": "{model_name}",
                    "task_type": TASK_TYPE,
                    "required_columns": REQUIRED_COLUMNS,
                    "status": "ready",
                }}

            @app.get("/schema")
            def get_schema():
                """Return the full input schema with column types and ranges."""
                return SCHEMA

            @app.post("/predict")
            def predict(payload: dict[str, Any]):
                """Predict for a single record.

                Returns HTTP 422 with {{validation_errors: [...]}} for invalid inputs.
                """
                errors = validate_input(payload)
                if errors:
                    raise HTTPException(status_code=422, detail={{"validation_errors": errors}})
                try:
                    X = preprocess(payload)
                    pred = model.predict(X).tolist()[0]
                    result: dict[str, Any] = {{"prediction": pred}}
                    if TASK_TYPE == "classification" and hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X).tolist()[0]
                        classes = (
                            model.classes_.tolist()
                            if hasattr(model, "classes_")
                            else list(range(len(proba)))
                        )
                        result["probabilities"] = dict(zip([str(c) for c in classes], proba))
                    return result
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Prediction error: {{exc}}") from exc

            @app.post("/predict/batch")
            def predict_batch(records: list[dict[str, Any]]):
                """Predict for a list of records in one request."""
                validation_errors: dict[int, list[str]] = {{}}
                for idx, rec in enumerate(records):
                    errs = validate_input(rec)
                    if errs:
                        validation_errors[idx] = errs
                if validation_errors:
                    raise HTTPException(
                        status_code=422,
                        detail={{"validation_errors": validation_errors}},
                    )
                results = []
                for rec in records:
                    X = preprocess(rec)
                    pred = model.predict(X).tolist()[0]
                    entry: dict[str, Any] = {{"prediction": pred}}
                    if TASK_TYPE == "classification" and hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X).tolist()[0]
                        classes = (
                            model.classes_.tolist()
                            if hasattr(model, "classes_")
                            else list(range(len(proba)))
                        )
                        entry["probabilities"] = dict(zip([str(c) for c in classes], proba))
                    results.append(entry)
                return {{"predictions": results}}
        ''')

    def _render_requirements_txt(self) -> str:
        """Generate a requirements.txt with pinned versions from the current runtime."""
        lines: list[str] = []
        for pkg in _SERVING_PACKAGES:
            try:
                ver = _pkg_version(pkg)
                lines.append(f"{pkg}=={ver}")
            except Exception:
                lines.append(pkg)
        return "\n".join(lines) + "\n"

    def _render_dockerfile(self) -> str:
        """Generate a minimal production Dockerfile."""
        return dedent("""\
            FROM python:3.11-slim

            RUN apt-get update \
                && apt-get install -y --no-install-recommends libgomp1 curl \
                && rm -rf /var/lib/apt/lists/*

            WORKDIR /app

            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt

            COPY . .

            EXPOSE 8000

            CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        """)

    def _render_docker_compose(self, pipeline_id: str) -> str:
        """Generate a docker-compose.yml for one-command local testing."""
        short_id = pipeline_id[:8] if len(pipeline_id) >= 8 else pipeline_id
        return dedent(f"""\
            services:
              model-api:
                build: .
                image: flowml-model-{short_id}
                ports:
                  - "8000:8000"
                environment:
                  - LOG_LEVEL=info
                restart: unless-stopped
                healthcheck:
                  test: ["CMD", "curl", "-f", "http://localhost:8000/"]
                  interval: 30s
                  timeout: 5s
                  retries: 3
        """)

    # ── README generation ──────────────────────────────────────────────────

    def _generate_package_readme(
        self,
        model_name: str,
        task_type: str,
        input_schema: dict[str, Any],
        evaluation_result: dict[str, Any],
        explanation_result: dict[str, Any],
        pipeline_id: str,
    ) -> str:
        """Generate README via LLM; fall back to a deterministic template."""
        numeric_cols     = input_schema.get("numeric_columns", [])
        categorical_cols = input_schema.get("categorical_columns", [])
        target_col       = input_schema.get("target_column", "target")
        required_cols    = input_schema.get("required_columns", [])

        # Build an example payload snippet
        example_parts: list[str] = []
        for col in required_cols[:6]:
            if col in numeric_cols:
                lo = (input_schema.get("column_ranges") or {}).get(col, {}).get("min", 0.0)
                hi = (input_schema.get("column_ranges") or {}).get(col, {}).get("max", 1.0)
                mid = round((lo + hi) / 2, 4) if lo is not None and hi is not None else 0.0
                example_parts.append(f'  "{col}": {mid}')
            else:
                cats = (input_schema.get("allowed_categories") or {}).get(col, ["A"])
                example_parts.append(f'  "{col}": "{cats[0] if cats else "A"}"')
        suffix = ",\n  ..." if len(required_cols) > 6 else ""
        example_payload = "{\n" + ",\n".join(example_parts) + suffix + "\n}"

        # Metrics summary
        if task_type == "regression":
            metrics_text = (
                f"R² = {evaluation_result.get('r2', 'N/A')} | "
                f"RMSE = {evaluation_result.get('rmse', 'N/A')} | "
                f"MAE = {evaluation_result.get('mae', 'N/A')}"
            )
        else:
            metrics_text = (
                f"Accuracy = {evaluation_result.get('accuracy', 'N/A')} | "
                f"F1 = {evaluation_result.get('f1', 'N/A')}"
            )

        deploy_decision     = evaluation_result.get("deployment_decision", "deploy")
        perf_summary        = evaluation_result.get("performance_summary", "")
        explanation_summary = (
            explanation_result.get("summary")
            or explanation_result.get("llm_summary")
            or ""
        )
        top_features = explanation_result.get("top_features", [])
        top_feat_text = (
            ", ".join(
                str(f.get("feature", f)) if isinstance(f, dict) else str(f)
                for f in top_features[:5]
            )
            if top_features else "See /schema endpoint for full feature list"
        )

        # ── Try LLM first ──────────────────────────────────────────────────
        system_prompt = (
            "You are a technical writer generating a README.md for a machine learning model "
            "deployment package. The README should be clear, accurate, and developer-friendly. "
            "Include: what the model predicts, how to run it (Docker and manual), how to call "
            "the /predict endpoint with a valid example, what the output means, and known "
            "caveats. Use Markdown with headers. Be concise — aim for ~400 words."
        )
        user_prompt = (
            f"Model: {model_name}\n"
            f"Task: {task_type}\n"
            f"Target column: {target_col}\n"
            f"Performance: {metrics_text}\n"
            f"Deployment decision: {deploy_decision}\n"
            f"Performance summary: {perf_summary}\n"
            f"Explanation: {explanation_summary}\n"
            f"Top predictive features: {top_feat_text}\n"
            f"Input columns ({len(required_cols)} total): {', '.join(required_cols[:10])}"
            f"{'...' if len(required_cols) > 10 else ''}\n"
            f"Example request payload:\n{example_payload}\n\n"
            f"Pipeline ID: {pipeline_id}\n\n"
            "Write a complete README.md for this deployment package."
        )

        llm_text = self._generate_llm_text(
            system_prompt,
            user_prompt,
            temperature=0.4,
            max_tokens=1500,
        )
        if llm_text and len(llm_text.strip()) > 100:
            return self._ensure_compose_instructions(llm_text)

        # ── Deterministic fallback ─────────────────────────────────────────
        numeric_rows = "".join(f"| `{c}` | numeric |\n" for c in numeric_cols[:8])
        categorical_rows = "".join(f"| `{c}` | categorical |\n" for c in categorical_cols[:8])
        return dedent(f"""\
            # {model_name} – Deployment Package

            Auto-generated by **FlowML AutoML** | Pipeline `{pipeline_id[:8]}`

            ## What this model does

            This is a **{task_type}** model trained to predict **{target_col}**.

            **Evaluation metrics:** {metrics_text}
            **Deployment decision:** {deploy_decision}

            {perf_summary}

            ## Quick start

            ### Docker Compose (recommended)
            ```bash
            docker compose up --build
            ```
            The API will be available at http://localhost:8000.

            ### Manual
            ```bash
            pip install -r requirements.txt
            uvicorn app:app --reload --port 8000
            ```

            ## API Usage

            ### GET /schema
            Returns the full expected input schema.

            ### POST /predict
            ```bash
            curl -X POST http://localhost:8000/predict \\
              -H "Content-Type: application/json" \\
              -d '{example_payload}'
            ```

            **Response (classification)**
            ```json
            {{
              "prediction": "class_label",
              "probabilities": {{"class_a": 0.85, "class_b": 0.15}}
            }}
            ```

            **Response (regression)**
            ```json
            {{"prediction": 42.7}}
            ```

            ### POST /predict/batch
            Send a JSON array of records for bulk inference.

            ## Input schema

            | Column | Type |
            |--------|------|
            {numeric_rows}{categorical_rows}
            See `schema.json` for complete ranges and allowed category values.

            ## Top features

            {top_feat_text}

            ## Error handling

            The `/predict` endpoint returns **HTTP 422** with a `validation_errors` list for:
            - Missing required columns
            - Wrong value types
            - Numeric values outside the training range
            - Unknown categorical values

            ## Caveats

            - This model was trained on a specific dataset; distribution shift in inputs may reduce accuracy.
            - The preprocessing logic in `app.py` must match the original training pipeline — verify with `schema.json`.
        """)

    def _ensure_compose_instructions(self, readme_text: str) -> str:
        """Ensure generated README includes Docker Compose startup commands."""
        normalized = readme_text.lower()
        if "docker compose up" in normalized or "docker-compose up" in normalized:
            return readme_text

        compose_block = dedent("""

            ## Quick start (Docker)

            Run the API with Docker Compose:

            ```bash
            docker compose up --build
            ```

            If Compose tries to pull a non-existent prebuilt image first, use:

            ```bash
            docker compose up --build --pull never
            ```
        """)
        return readme_text.rstrip() + compose_block

    # ── Legacy helper (kept for API backward compatibility) ─────────────────

    def _generate_deployment_code(self, model_name: str, task_type: str) -> str:
        """Generate a simple deployment code snippet (legacy output field)."""
        return dedent(f'''\
            """Usage snippet for {model_name} ({task_type})."""
            import joblib, pandas as pd

            model = joblib.load("outputs/{{}}_model.pkl".format(pipeline_id))

            def predict(data: pd.DataFrame):
                return model.predict(data)
        ''')
