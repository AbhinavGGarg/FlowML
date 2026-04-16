# FlowML

> Convert a CSV into a deployable ML service, with transparent reasoning at every stage.

Most teams have useful data but never ship predictive features because the setup cost is high: data cleaning, model choice, tuning, validation, packaging, and documentation. That process can take weeks before anyone sees value.

**FlowML compresses that process into one guided pipeline.**

Upload a dataset, choose a target column, and let a coordinated set of AI agents execute analysis, preprocessing, feature work, training, evaluation, and deployment prep. Every decision is logged so you can inspect exactly what happened and why.

FlowML also includes a conversational control layer. You can ask the system to revise a stage, compare model runs, explain feature choices, or roll back and rerun. It tracks changes and preserves experiment history so results remain reproducible.

No specialized ML handoff required. No opaque automation. No guesswork about how the model was produced.

---

## What it does

1. **Upload data**: import a CSV in the web UI
2. **Select target**: choose the column to predict
3. **Execute pipeline**: eight agents run in order with live status updates
4. **Audit decisions**: inspect stage outputs and rationale
5. **Refine by chat**: request changes in natural language and rerun only impacted stages
6. **Deliver artifact**: export a complete deployment package with API and documentation

---

## Pipeline stages

| Agent | Responsibility |
|---|---|
| `DataAnalyzerAgent` | Profiles dataset quality, missingness, outliers, and relationships |
| `PreprocessorAgent` | Handles imputation, encoding, scaling, and train/test splitting |
| `FeatureEngineeringAgent` | Builds interaction and polynomial features, removes weak/redundant features |
| `ModelSelectionAgent` | Chooses up to three candidate algorithms and tuning spaces |
| `TrainingAgent` | Trains candidates, runs Optuna tuning, compares CV performance |
| `EvaluationAgent` | Computes final metrics, checks overfit risk, decides readiness |
| `DeploymentAgent` | Saves model assets and assembles deployable bundle |
| `ExplanationGeneratorAgent` | Produces feature-importance and SHAP-style interpretation output |

---

## Conversational pipeline agent

The chat panel is not a simple Q&A layer. It acts as an orchestration interface that can update configuration, trigger scoped reruns, and explain pipeline behavior with full context from prior runs.

---

## Deployment package

A successful run exports a ready-to-ship archive:

```text
deployment/
  app.py              ← FastAPI serving endpoint with schema checks
  requirements.txt    ← dependency versions aligned to training
  Dockerfile
  docker-compose.yml
  schema.json         ← expected columns, dtypes, and value boundaries
  README.md           ← model behavior, caveats, I/O contract
  model.pkl           ← trained model artifact
  report.html         ← formatted metrics + explanation report
```

Run `docker-compose up` and the service is online. The generated deployment README captures key operational context so deployment knowledge is not trapped with one person.

---

## Tech stack

### Backend
- FastAPI
- pandas / NumPy
- scikit-learn
- XGBoost / LightGBM
- Optuna
- SHAP

### Frontend
- React 18 + TypeScript + Vite
- Tailwind + Radix UI
- Recharts
- Framer Motion

---

## Project structure

```text
agents/        AutoML agents (analysis, prep, training, eval, deploy)
api/           FastAPI routes and stage execution endpoints
core/          Orchestration, comparison, HPO, and ML utilities
frontend/      React interface and dashboards
utils/         Shared helpers (logging, OpenRouter client, insights)
tests/         Pytest suite
outputs/       Artifacts (models, reports, deployment bundles)
uploads/       Uploaded CSV files (cleaned up after completion)
```

---

## Quick start

### 1. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.template` to `.env` and set the required key:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for LLM calls |
| `MODEL_NAME` | `arcee-ai/trinity-large-preview:free` | LLM selection |
| `ENABLE_MULTI_MODEL` | `true` | Train all candidates or single model |
| `ENABLE_HPO` | `true` | Enable Optuna search |
| `N_HPO_TRIALS` | `20` | Optuna budget |
| `ENABLE_ENSEMBLE` | `false` | Enable stacking/voting after training |

### 3. Start backend

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend points to `http://127.0.0.1:8000` by default.

---

## Testing

### Backend

```bash
pytest
pytest tests/test_agents.py -v
```

### Frontend

```bash
cd frontend
npm test
npx tsc --noEmit
```

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/dataset/upload` | Upload CSV |
| `GET` | `/api/dataset/summary` | Dataset summary |
| `GET` | `/api/dataset/columns` | Available columns |
| `POST` | `/api/dataset/target` | Set prediction target |
| `POST` | `/api/pipeline/stage/{stage_id}` | Execute a stage |
| `GET` | `/api/pipeline/status` | Pipeline status |
| `GET` | `/api/pipeline/logs` | Live execution logs |
| `GET` | `/api/stages/{stage_id}/results` | Stage output details |
| `GET` | `/api/results/metrics` | Final model metrics |
| `GET` | `/api/results/explanation` | Importance and explanation output |
| `GET` | `/api/results/evaluation-insights` | Generalization and overfit analysis |
| `GET` | `/api/results/download/model` | Download model file |
| `GET` | `/api/results/download/logs` | Download run logs |
| `GET` | `/api/results/download/deployment-package` | Download deployment archive |
| `GET` | `/api/results/download/report` | Download HTML report |
