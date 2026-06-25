# AI-Powered Customer Retention Copilot

End-to-end customer retention platform that goes beyond "will the customer churn?" to answer **why**, **what to do**, **what's the ROI**, and **who to prioritize** — wired together as a multi-agent system on top of explainable churn models, RAG, and a FastAPI service.

## Architecture

```
Customer Data
     |
     v
Feature Engineering
     |
     v
Churn Prediction Model (XGBoost / LightGBM)
     |
     +------> SHAP Explainability
     |
     +------> Customer Segmentation (KMeans + RFM)
     |
     +------> LLM Recommendation Engine (Claude)
     |
     +------> Multi-Agent Retention System (LangGraph)
     |             Profile -> Risk -> Explanation -> Retention -> ROI
     |
     +------> RAG (FAISS) over retention playbooks
     |
     v
FastAPI Backend  +  Streamlit Dashboard
     |
     v
Docker / MLflow / AWS (S3, SageMaker, Bedrock, ECS, CloudWatch)
```

## Phases

| Phase | Component                             | Path                                           |
| ----- | ------------------------------------- | ---------------------------------------------- |
| 1     | Data + synthetic enhancement          | `src/data/`                                    |
| 2     | Churn models (LR/RF/XGB/LightGBM)     | `src/models/churn_predictor.py`                |
| 3     | Segmentation (KMeans + RFM)           | `src/models/segmentation.py`                   |
| 4     | SHAP explainability                   | `src/explainability/`                          |
| 5     | GenAI recommendation engine           | `src/llm/`                                     |
| 6     | LangGraph multi-agent                 | `src/agents/`                                  |
| 7     | RAG with FAISS                        | `src/rag/`, `knowledge_base/`                  |
| 8     | ROI prediction                        | `src/models/roi_estimator.py`                  |
| 9     | MLOps (MLflow)                        | `src/mlops/`                                   |
| 10    | FastAPI backend                       | `src/api/`                                     |
| 11    | Docker + AWS deployment               | `deployment/`                                  |

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate synthetic-enhanced dataset (no Kaggle creds required)
python scripts/generate_data.py

# 3. Train all models + log to MLflow
python scripts/train_model.py

# 4. Ingest knowledge base into FAISS
python scripts/ingest_knowledge.py

# 5. Run the multi-agent pipeline end-to-end on a sample customer
python scripts/run_pipeline.py --customer-id 7590-VHVEG

# 6. Serve the API
uvicorn src.api.main:app --reload --port 8000

# 7. Open the dashboard
streamlit run dashboard/streamlit_app.py
```

## Endpoints

| Method | Path                          | Purpose                                         |
| ------ | ----------------------------- | ----------------------------------------------- |
| POST   | `/predict_churn`              | Churn probability + segment for a customer      |
| POST   | `/customer_analysis`          | SHAP-driven explanation of churn drivers        |
| POST   | `/generate_strategy`          | LLM + RAG retention recommendation              |
| POST   | `/customer_roi`               | Expected revenue saved & ROI of recommended action |
| POST   | `/copilot/run`                | Full multi-agent pipeline (Profile→ROI)         |
| GET    | `/health`                     | Liveness                                        |

## Models & metrics

Target: **ROC-AUC > 0.85** on Telco Customer Churn.

| Model               | Use         |
| ------------------- | ----------- |
| Logistic Regression | Baseline    |
| Random Forest       | Baseline    |
| XGBoost             | Production  |
| LightGBM            | Production (fallback / ensemble) |

All runs tracked in MLflow (`mlruns/`). The best model is promoted to the `Production` stage in the registry and loaded by the API.

## Multi-agent flow (LangGraph)

```
START
  |
  v
[Profile Agent]  --- builds customer 360 from raw + synthetic features
  |
  v
[Risk Agent]     --- loads production churn model, returns prob + segment
  |
  v
[Explanation Agent] --- SHAP top-k drivers, plain-English rationale
  |
  v
[Retention Agent]   --- LLM + RAG over playbooks, structured action plan
  |
  v
[ROI Agent]         --- expected revenue saved, offer cost, payback
  |
  v
END  ->  consolidated CopilotReport
```

## Project layout

```
churn-copilot/
├── data/                     # raw, processed, synthetic
├── knowledge_base/           # markdown policies, playbooks, campaigns
├── notebooks/                # EDA, modeling, SHAP analysis
├── src/
│   ├── data/                 # loader, synthetic generator, preprocessor
│   ├── features/             # feature engineering
│   ├── models/               # churn, segmentation, ROI
│   ├── explainability/       # SHAP
│   ├── llm/                  # Claude client + prompt templates
│   ├── agents/               # LangGraph agents + orchestrator
│   ├── rag/                  # FAISS index + retriever
│   ├── api/                  # FastAPI app
│   ├── mlops/                # MLflow tracking + registry helpers
│   └── utils/                # logging, config
├── tests/
├── scripts/                  # one-shot pipelines
├── deployment/
│   ├── docker/
│   └── aws/                  # SageMaker, ECS, CloudFormation
└── dashboard/                # Streamlit UI
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
MLFLOW_TRACKING_URI=./mlruns
FAISS_INDEX_DIR=./artifacts/faiss
MODEL_DIR=./artifacts/models
```

## Resume bullets

- Developed an end-to-end customer retention platform using XGBoost, LangGraph, Claude (Anthropic), and FastAPI to predict churn and automate retention decision-making.
- Built a multi-agent architecture (Profile → Risk → Explanation → Retention → ROI) on LangGraph for end-to-end retention orchestration.
- Implemented RAG-based retrieval using FAISS over historical retention campaigns and customer success playbooks.
- Achieved ROC-AUC > 0.85 on churn prediction and improved recommendation relevance through retrieval-augmented generation.
- Containerized services using Docker and tracked experiments with MLflow; deployed to AWS (S3 / SageMaker / Bedrock / ECS / CloudWatch).
