# FlowML

> Turn any CSV into a production-ready model and know exactly how and why, every step of the way.

Most dev teams sit on data worth predicting from and never act on it. Not because the idea isn't there, but because building a reliable ML pipeline is a project in itself: weeks of preprocessing decisions, model selection guesswork, hyperparameter tuning, and evaluation work, all before you've written a single line of serving code.

**FlowML removes all of that.**

Upload a dataset. Pick a target column. A team of AI agents handles the rest (analysis, feature engineering, model selection, training, evaluation, and deployment) while logging every decision it makes so you can inspect, trust, and ship the result.

And when the pipeline finishes, you're not done interacting with it. A conversational chatbot agent sits on top of the entire system, letting you modify, rerun, and interrogate the pipeline through natural language. Ask why a feature was selected. Undo a change. Compare two models. It plans the revision, executes the affected stages, and tracks the full history, so every experiment is reproducible and nothing is lost.

No ML background required. No black box you can't interrogate. No hand-off to a specialist.

---

## What it does

1. **Upload a dataset**: drag and drop a CSV into the browser
2. **Pick a target column**: the column you want to predict
3. **Run the pipeline**: eight agents execute in sequence, visualised in real time
4. **Inspect every decision**: click any stage to see what the agent did and why
5. **Refine conversationally**: tell the chatbot what you want to change; it plans the revision, reruns only the affected stages, and stores the diff
6. **Ship the result**: download a deployment package with a working REST API, Docker config, and a plain-English model report ready for stakeholders

---

## Pipeline stages

| Agent | What it does |
|---|---|
| `DataAnalyzerAgent` | Profiles columns, flags missing values and outliers, builds correlation view |
| `PreprocessorAgent` | Imputes, encodes categoricals, scales numerics, splits train/test |
| `FeatureEngineeringAgent` | Creates polynomial and interaction features, prunes redundant ones |
| `ModelSelectionAgent` | Selects up to 3 candidate algorithms with fixed params and search spaces |
| `TrainingAgent` | Trains all candidates, runs Optuna HPO per candidate, compares by CV score |
| `EvaluationAgent` | Computes metrics, detects overfitting, decides deploy or reject |
| `DeploymentAgent` | Saves model and metadata, generates deployment package |
| `ExplanationGeneratorAgent` | Generates feature importance and SHAP-style explanations |

---

## Conversational pipeline agent
A floating chat interface gives you full control over the pipeline through natural language. This isn't just a Q&A wrapper-- it's an orchestration agent that can modify, rerun, and audit the pipeline safely.

---

## Deployment package

A successful pipeline produces a ready-to-ship zip containing:

```
deployment/
  app.py              ← FastAPI serving endpoint, schema-validated
  requirements.txt    ← pinned to training versions
  Dockerfile
  docker-compose.yml
  schema.json         ← column names, types, expected ranges
  README.md           ← What the model does, caveats, input/output spec
  model.pkl           ← trained model artifact 
  report.html         ← styled pipeline report with metrics and explanations
```

`docker-compose up` and your model is serving. The generated `README.md` transfers knowledge automatically, removing human dependency.

---

## Tech stack

### Backend
- FastAPI
- pandas / NumPy
- scikit-learn
- XGBoost / LightGBM
- Optuna (HPO)
- SHAP

### Frontend
- React 18 + TypeScript + Vite
- Tailwind + Radix UI
- Recharts
- Framer Motion

---

## Project structure

```
agents/        Specialized AutoML agents (analysis, preprocessing, training, etc.)
api/           FastAPI endpoints and pipeline stage execution
core/          Orchestration and ML utilities (comparison, HPO, ensembling)
frontend/      React UI and dashboards
utils/         Shared helpers (logging, OpenRouter client, insights)
tests/         Pytest test suite
outputs/       Saved artifacts (models, metadata, reports, deployment zips)
uploads/       Uploaded datasets (deleted after pipeline completes)
```

---

## Quick start

### 1. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.template` to `.env` and set OPENROUTER_API_KEY:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for all LLM calls |
| `MODEL_NAME` | `arcee-ai/trinity-large-preview:free` | LLM model |
| `ENABLE_MULTI_MODEL` | `true` | Train all candidates vs. just one |
| `ENABLE_HPO` | `true` | Run Optuna HPO |
| `N_HPO_TRIALS` | `20` | Optuna trial budget |
| `ENABLE_ENSEMBLE` | `false` | Stack/vote ensemble after training |

### 3. Run the backend

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend targets `http://127.0.0.1:8000` by default.

---

## Testing

### Backend

```bash
pytest
pytest tests/test_agents.py -v   # specific file
```

### Frontend

```bash
cd frontend
npm test
npx tsc --noEmit   # type check
```

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/dataset/upload` | Upload CSV |
| `GET` | `/api/dataset/summary` | Dataset profile |
| `GET` | `/api/dataset/columns` | Column list |
| `POST` | `/api/dataset/target` | Set target column |
| `POST` | `/api/pipeline/stage/{stage_id}` | Run a pipeline stage |
| `GET` | `/api/pipeline/status` | Current pipeline state |
| `GET` | `/api/pipeline/logs` | Live logs |
| `GET` | `/api/stages/{stage_id}/results` | Stage result detail |
| `GET` | `/api/results/metrics` | Final evaluation metrics |
| `GET` | `/api/results/explanation` | Feature importance / SHAP |
| `GET` | `/api/results/evaluation-insights` | Overfitting analysis |
| `GET` | `/api/results/download/model` | Download model artifact |
| `GET` | `/api/results/download/logs` | Download run logs |
| `GET` | `/api/results/download/deployment-package` | Download deployment zip |
| `GET` | `/api/results/download/report` | Download HTML report |
