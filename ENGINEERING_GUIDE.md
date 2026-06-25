# Churn-Copilot — The Complete AI Engineering Guide

> A walkthrough of **every** layer of this project, written for someone who wants to *understand the
> decisions*, not just run the code. For each component we cover: **what it does**, **how it works**
> (with the actual code), **why this choice and not the alternatives**, and **what changes in
> production** (scaling, guardrails, monitoring, data).
>
> Use this as a reference for *any* applied-ML / GenAI system, not just churn.

---

## Table of contents

1. [The problem & the mental model](#1-the-problem--the-mental-model)
2. [What "churn" actually means — and how we detect it](#2-what-churn-actually-means--and-how-we-detect-it)
3. [The data: what it looks like & how it's made](#3-the-data-what-it-looks-like--how-its-made)
4. [Preprocessing & feature engineering](#4-preprocessing--feature-engineering)
5. [The churn model: 4 algorithms, 1 winner](#5-the-churn-model-4-algorithms-1-winner)
6. [Evaluation: which metric, and why not accuracy](#6-evaluation-which-metric-and-why-not-accuracy)
7. [Segmentation: KMeans + RFM + business quadrants](#7-segmentation-kmeans--rfm--business-quadrants)
8. [Explainability: SHAP](#8-explainability-shap)
9. [The GenAI layer: LLM client & prompts](#9-the-genai-layer-llm-client--prompts)
10. [RAG: retrieval over playbooks](#10-rag-retrieval-over-playbooks)
11. [The multi-agent system (LangGraph)](#11-the-multi-agent-system-langgraph)
12. [ROI estimation: turning a probability into a dollar decision](#12-roi-estimation-turning-a-probability-into-a-dollar-decision)
13. [Serving: the FastAPI layer](#13-serving-the-fastapi-layer)
14. [MLOps: MLflow tracking & the model registry](#14-mlops-mlflow-tracking--the-model-registry)
15. [Deployment: Docker & AWS](#15-deployment-docker--aws)
16. [Scaling to production](#16-scaling-to-production)
17. [Guardrails](#17-guardrails)
18. [Performance tracking & monitoring](#18-performance-tracking--monitoring)
19. [The "perfect score" trap — a critical lesson](#19-the-perfect-score-trap--a-critical-lesson)
20. [Interview / design-review cheat sheet](#20-interview--design-review-cheat-sheet)

---

## 1. The problem & the mental model

Most churn projects stop at *"will this customer leave?"* — a single probability. That number alone
is **not actionable**. A Customer Success Manager (CSM) reading `churn_probability = 0.78` still has
to ask four more questions before doing anything:

| Question | Answered by | Component |
| --- | --- | --- |
| **Will** they churn? | Probability | Churn model (XGBoost/LightGBM) |
| **Why**? | Feature attributions | SHAP |
| **What** should we do? | A retrieved + reasoned action plan | RAG + LLM |
| **Is it worth it** ($)? | Expected revenue saved vs. cost | ROI estimator |
| **Who** do I call first? | Value × Risk segment | Segmentation |

This project wires all five into one pipeline. The **core engineering insight** that recurs at every
layer: *a prediction is the beginning of a decision, not the end of one.* Everything downstream
(SHAP, RAG, ROI) exists to convert a number into a defensible human action.

**Architecture at a glance:**

```
Customer record
   │
   ├─► Preprocess (scale + one-hot)
   │
   ├─► Churn model ──► probability
   │        │
   │        ├─► SHAP ──────────► top drivers ("why")
   │        ├─► Segmentation ──► value × risk quadrant + KMeans persona ("who")
   │        └─► RAG + LLM ─────► retention plan ("what")
   │                  │
   │                  └─► ROI estimator ──► $ saved, cost, payback ("worth it?")
   │
   └─► FastAPI / Streamlit, tracked by MLflow, deployable to AWS
```

The five-step reasoning is implemented as a **LangGraph multi-agent pipeline**:
`Profile → Risk → Explanation → Retention → ROI`.

---

## 2. What "churn" actually means — and how we detect it

This is the single most under-discussed decision in any churn project, so we start here.

**"Churn" is a label you define, not a fact you observe.** Before any model, you must answer:

- **Voluntary vs. involuntary?** A customer who cancels is different from one whose card expired.
- **What window?** "Churned" usually means *no activity / cancelled within the next N days* (30/60/90).
  This horizon must match how fast you can intervene — predicting churn 3 days out is useless if your
  retention play takes 2 weeks.
- **Hard vs. soft churn?** Full cancellation vs. downgrade / dormancy.

In this project the label is the Telco-style binary `Churn ∈ {Yes, No}`, converted to `1/0` in
[`src/data/preprocessor.py`](src/data/preprocessor.py):

```python
df[settings.target_col] = (df[settings.target_col] == "Yes").astype(int)
```

### The "basic function" that detects churn in the synthetic data

Because there's no live Telco feed, the project *manufactures* a realistic churn signal with a
**weighted logistic score** in [`src/data/synthetic_generator.py`](src/data/synthetic_generator.py).
This is worth reading closely, because it encodes **domain knowledge about why people churn**:

```python
score = (
    (contract == "Month-to-month") * 1.2     # biggest risk: no lock-in
    + (tenure < 6) * 1.0                      # brand-new customers bail
    + (monthly > 80) * 0.5                    # price sensitivity
    + (payment == "Electronic check") * 0.6   # friction / lower commitment
    + (internet == "Fiber optic") * 0.4       # higher expectations / outages
    + (online_security == "No") * 0.3
    + (tech_support == "No") * 0.3
    + (senior == 1) * 0.3
    - (contract == "Two year") * 1.5           # strong loyalty anchor
    - (tenure > 48) * 0.8                       # long-tenure = sticky
    + rng.normal(0, 0.6, n)                     # noise so it's learnable, not trivial
)
prob = 1 / (1 + np.exp(-(score - 0.5)))         # logistic sigmoid → probability
churn = np.where(rng.random(n) < prob, "Yes", "No")
```

**Why model it this way?** Three reasons every synthetic-data exercise should follow:

1. **Signal must be real but not perfect.** The coefficients are domain-plausible (month-to-month is
   the #1 churn driver in real telco data); the Gaussian noise term prevents a trivially separable
   dataset (well — *mostly*; see §19).
2. **It's a generative story, not random labels.** Random labels would make the whole ML pipeline a
   no-op. A *structured* generator lets SHAP later "rediscover" the very drivers we baked in — a great
   sanity check.
3. **Reproducible** via `random_seed=42`.

> **Production reality:** you will *not* hand-write this. Your label comes from a `SELECT` against
> event logs / billing — e.g. `churned = 1 if days_since_last_activity > 60 or subscription_status =
> 'cancelled'`. The hard part is **label leakage** (don't include the cancellation event itself as a
> feature) and **survivorship/censoring** (customers who *haven't yet* had time to churn are not
> negatives — they're censored, which is why survival models exist; see §5).

---

## 3. The data: what it looks like & how it's made

### Base schema (Kaggle Telco-compatible, 21 columns)

`customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn`

### + 7 synthetic *behavioural* fields (the project's value-add)

Real churn rarely lives in billing columns alone — it lives in **engagement signals**. The generator
adds these, deliberately drawing churners and stayers from *different distributions* so the signal is
learnable:

| Field | Churner distribution | Loyal distribution | Why it matters |
| --- | --- | --- | --- |
| `last_login_days` | `randint(7,90)` | `randint(0,14)` | Recency = #1 engagement signal |
| `support_ticket_count` | `poisson(1)+poisson(2)` | `poisson(1)` | Friction |
| `avg_response_time` (hrs) | `gamma(2.5, 6.0)` | `gamma(2.0, 2.0)` | Service quality |
| `nps_score` (0–10) | `randint(0,6)` (detractors) | `randint(7,11)` (promoters) | Stated satisfaction |
| `feature_usage_score` (0–1) | `beta(2,6)` (low) | `beta(5,2)` (high) | Product stickiness |
| `sentiment` | derived from NPS+tickets | — | Qualitative tag |
| `tenure_bucket` | binned `tenure` | — | Cohort analysis |

**Why different statistical distributions per field?** Each one mirrors the real-world shape of that
metric: ticket counts are **Poisson** (counts of rare events), response times are **Gamma**
(positive, right-skewed), usage is **Beta** (bounded 0–1). Drawing churners and stayers from shifted
versions of the *correct* distribution is what makes a synthetic dataset behave like real one.

### Monetary fields are *derived*, not random

```python
base_monthly = where(internet=="Fiber optic", 75, where(internet=="DSL", 45, 20))
addon_boost  = (security + backup + protection + techsupport)*5 + (tv + movies)*10
monthly      = (base_monthly + addon_boost + noise).clip(18.25, 119.0)
total        = (monthly * tenure + noise).clip(min=0)
```

This keeps internal consistency (`TotalCharges ≈ MonthlyCharges × tenure`) — important because a model
will otherwise learn the *artifacts* of bad synthetic data rather than the intended signal.

The loader ([`src/data/loader.py`](src/data/loader.py)) is **zero-dependency**: if the real Kaggle CSV
isn't present it builds the synthetic set on demand, so the repo runs anywhere with no credentials.

---

## 4. Preprocessing & feature engineering

Two **separate** transformation paths exist, and the separation is intentional:

### (a) Model preprocessor — [`src/data/preprocessor.py`](src/data/preprocessor.py)

A scikit-learn `ColumnTransformer`:

```python
ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_COLS),       # 9 cols
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),  # 17 cols
], remainder="drop")
```

Key decisions and **why**:

- **`StandardScaler` on numerics** — mandatory for Logistic Regression (gradient-based, scale-sensitive)
  and harmless for trees. Scaling *everything* lets you swap models without re-thinking preprocessing.
- **`OneHotEncoder(handle_unknown="ignore")`** — at inference a category never seen in training
  (e.g. a new payment method) encodes to all-zeros instead of crashing. This is a **production
  guardrail**, not a convenience.
- **`customerID` is dropped** — it's an identifier. Leaving an ID in is a classic **leakage** bug: the
  model memorises individuals instead of learning patterns.
- **`train_test_split(..., stratify=y, random_state=42)`** — stratification preserves the churn rate
  in both splits (critical when classes are imbalanced); the seed makes runs reproducible.
- **Encoders are persisted to `joblib`** — *the exact same fitted transformer must be used at
  inference.* Re-fitting at predict time would silently produce different encodings → "training/serving
  skew", the most common production-ML bug. `transform_one(record, pre)` applies the saved encoder to a
  single live record.

**Why one-hot and not target/ordinal encoding?** One-hot is safe with low-cardinality categoricals
(all of these have <5 values) and introduces no leakage. Target encoding leaks the label unless you do
careful out-of-fold encoding; ordinal encoding invents a false ordering. With low cardinality, one-hot
wins on simplicity and safety.

### (b) Business feature engineering — [`src/features/engineering.py`](src/features/engineering.py)

These produce **human-readable** features for segmentation and dashboards (kept out of the model
preprocessor so reports stay interpretable):

- **CLV**: use `TotalCharges` if > 0, else `MonthlyCharges × tenure`.
- **RFM scores**: Recency = `-last_login_days`, Frequency = `feature_usage_score`,
  Monetary = `MonthlyCharges`, each binned into quintiles 1–5 via `pd.qcut`, summed to `rfm_score` (3–15).
- **Value tier** (High/Mid/Low) from annualised revenue and tenure thresholds.
- **Risk tier** (High ≥ 0.6, Medium ≥ 0.3, else Low) from churn probability.
- **Segment** = the Value × Risk matrix (e.g. *"High Value + High Risk"* = drop-everything-and-call).

---

## 5. The churn model: 4 algorithms, 1 winner

[`src/models/churn_predictor.py`](src/models/churn_predictor.py) trains **four** models, then picks the
best. This is the heart of "why this model and not that one."

### The four contenders

| Model | Role | Key hyperparameters | Why it's here |
| --- | --- | --- | --- |
| **Logistic Regression** | Baseline | `C=1.0, max_iter=1000` | Fast, fully interpretable, sets the floor |
| **Random Forest** | Baseline | `400 trees, max_depth=12, class_weight="balanced"` | Bagging baseline, robust, low-tuning |
| **XGBoost** | Production | `600 trees, depth=6, lr=0.05, subsample=0.9, scale_pos_weight=neg/pos, tree_method="hist"` | Usually the SOTA on tabular |
| **LightGBM** | Production / fallback | `700 trees, num_leaves=63, lr=0.05, class_weight="balanced"` | Faster on big/wide data, leaf-wise growth |

```python
# XGBoost — note the imbalance handling and AUC early-stopping metric
pos = max(int(y_train.sum()), 1); neg = len(y_train) - pos
XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
              subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
              scale_pos_weight=neg/pos,           # ← counteracts class imbalance
              eval_metric="auc", tree_method="hist")
```

### Why always start with a baseline (Logistic Regression)?

A baseline answers *"is the complexity worth it?"*. If a 50-tree gradient booster beats logistic
regression by 0.3% AUC, you ship the linear model — it's faster, debuggable, and stakeholders trust it.
**You never know if a fancy model is good until you know what a simple one scores.**

### Why gradient boosting (XGBoost/LightGBM) for production?

On **tabular, heterogeneous** data (mix of categorical + numeric, non-linear interactions), gradient-
boosted trees are the empirical winners. They:

- handle non-linearities and feature interactions automatically (no manual interaction terms),
- are insensitive to feature scaling and monotonic transforms,
- handle mixed types gracefully,
- expose feature importances and play perfectly with **SHAP TreeExplainer** (exact, fast).

### Why these were considered but **not** chosen as the primary model

| Alternative | Why not (here) |
| --- | --- |
| **Deep learning (MLP / TabNet / FT-Transformer)** | On ~7k rows of tabular data, neural nets *underperform* GBMs and need far more tuning/compute. Deep learning wins on images/text/audio and very large tabular sets — not this. |
| **Plain Logistic Regression as production** | Misses non-linear interactions (e.g. "month-to-month *and* fiber *and* new"). Kept only as baseline. |
| **SVM** | Doesn't scale, no native probability, no clean SHAP story. |
| **Naive Bayes** | Strong independence assumption violated by correlated features (charges ↔ add-ons). |
| **kNN** | Slow at inference, suffers in high-dimensional one-hot space, no explainability. |
| **Survival models (Cox PH, Kaplan-Meier)** | *The technically "more correct" framing* — churn is time-to-event with censoring. Not used here because the Telco label is a fixed binary snapshot, but in production with timestamps you should seriously consider survival analysis (predicts *when*, handles censoring properly). |

### XGBoost vs. LightGBM — when each wins

- **XGBoost**: level-wise tree growth, extremely well-tested, great default. Slightly slower.
- **LightGBM**: leaf-wise growth + histogram binning → faster on **wide / large** datasets, but can
  overfit small data (hence `num_leaves` control). Kept as the **fallback / ensemble** member.

### Class imbalance — handled three ways

Real churn is ~15–25% positives. Three levers used here:
1. `class_weight="balanced"` (RF, LightGBM) — reweights the loss inversely to class frequency.
2. `scale_pos_weight = neg/pos` (XGBoost) — same idea, XGBoost's native knob.
3. **Evaluating on PR-AUC**, not accuracy (see §6).

*Not used but valid:* **SMOTE / oversampling** (`imbalanced-learn` is in `requirements.txt`). Reweighting
is generally preferred over synthetic oversampling because it doesn't fabricate data points; SMOTE is a
fallback when reweighting underperforms.

---

## 6. Evaluation: which metric, and why not accuracy

Every model is scored with six metrics; the winner is chosen by **ROC-AUC**:

```python
metrics = {
  "roc_auc":   roc_auc_score(y, proba),            # ranking quality, threshold-free
  "pr_auc":    average_precision_score(y, proba),  # precision-recall AUC (imbalance-robust)
  "accuracy":  accuracy_score(y, pred),
  "precision": precision_score(y, pred),           # of those we flagged, how many really churn
  "recall":    recall_score(y, pred),              # of all true churners, how many we caught
  "f1":        f1_score(y, pred),
}
best = max(results, key=lambda r: r.metrics["roc_auc"])
```

**Why not accuracy?** With 20% churn, a model that predicts *"nobody churns"* scores **80% accuracy**
and is completely useless. Accuracy is a trap on imbalanced data.

**Why ROC-AUC to select?** It's **threshold-independent** — it measures how well the model *ranks* a
random churner above a random non-churner. You want a good ranker because the business sets the
intervention threshold *later* based on budget/capacity.

**Why also track PR-AUC?** ROC-AUC can look optimistic under heavy imbalance; PR-AUC focuses on the
positive (churn) class and is the more honest headline metric when positives are rare.

**The precision/recall trade-off is a business decision, not a math one:**
- High **recall** → catch every churner, but waste retention budget on false alarms.
- High **precision** → only act on sure things, but miss savable customers.
- You pick the **operating threshold** (default 0.5 here) to match retention-team capacity and offer
  cost. In production this threshold is tuned on the PR curve, *not* left at 0.5.

---

## 7. Segmentation: KMeans + RFM + business quadrants

[`src/models/segmentation.py`](src/models/segmentation.py) runs **two complementary** segmentations —
and using both is itself a design lesson.

### (a) Rule-based business quadrant (Value × Risk)

Deterministic, explainable to executives: *High Value + High Risk*, *High Value + Low Risk*, etc. This
drives **prioritisation** ("call the high-value/high-risk people first").

### (b) Unsupervised KMeans persona clustering

```python
KMeans(n_clusters=4, n_init=10, random_state=42).fit(StandardScaler().fit_transform(X))
```
on 7 behavioural columns (`tenure, MonthlyCharges, last_login_days, support_ticket_count,
avg_response_time, nps_score, feature_usage_score`). Clusters are auto-named from their **centroids**:

```python
if nps >= 8 and tickets <= 2:        "Loyal Promoter"
elif tickets >= 3 and nps <= 6:      "Frustrated"
elif tenure <= 12 and monthly >= 70: "New & Expensive"
elif feature_usage <= 0.3:           "Disengaged"
else:                                "Steady Mainstream"
```

**Why KMeans?** Simple, fast, scales, and gives interpretable centroids you can label. **Why scale
first?** KMeans uses Euclidean distance — without scaling, `MonthlyCharges` (range ~100) would dominate
`feature_usage_score` (range 0–1). **Why `n_init=10`?** KMeans is sensitive to initialisation; trying
10 random starts and keeping the best avoids bad local optima.

**Alternatives & why not (yet):**
- **DBSCAN / HDBSCAN** — finds arbitrary shapes & outliers, no preset *k*, but harder to label and
  sensitive to its `eps` parameter. Worth it when clusters aren't blob-shaped.
- **GMM** — soft assignment (probabilistic membership), better when clusters overlap.
- **Choosing k** — `k=4` is hardcoded for interpretability; production would pick k via the
  **elbow method** or **silhouette score**.

**Why two segmentation systems at all?** Rules give you *defensible, stable* business buckets;
clustering *discovers* patterns you didn't pre-define. Rules answer "who do I prioritise?"; personas
answer "what kind of customer is this?" — different jobs.

---

## 8. Explainability: SHAP

A probability with no "why" is unusable to a CSM and dangerous to ship. SHAP
([`src/explainability/shap_explainer.py`](src/explainability/shap_explainer.py)) attributes each
prediction to its features.

```python
name = type(self.model).__name__.lower()
if "xgb" in name or "lgbm" in name or "forest" in name:
    return shap.TreeExplainer(self.model)        # exact & fast for trees
return shap.LinearExplainer(self.model, background=np.zeros((1, n_features)))  # linear fallback
```

Then it extracts the **top-k drivers by absolute SHAP value** and prettifies them:

```python
order = np.argsort(np.abs(contribs))[::-1][:top_k]   # top 5 by |contribution|
# "Contract_Month-to-month" → "Contract: Month-to-month"; signed value: + pushes toward churn
```

**What is SHAP?** Rooted in cooperative game theory (Shapley values): it fairly distributes the
prediction among features by averaging each feature's marginal contribution over all orderings. The
output is **additive**: `base_value + Σ(feature contributions) = prediction`.

**Why SHAP over the alternatives?**

| Method | Why not chosen as primary |
| --- | --- |
| **Built-in `feature_importances_`** | *Global* only (whole model), not *per-customer*. A CSM needs "why THIS person." |
| **Permutation importance** | Also global; expensive; unstable with correlated features. |
| **LIME** | Local like SHAP, but uses a random local surrogate → unstable, no consistency guarantees. SHAP has theoretical soundness + a fast exact `TreeExplainer`. |
| **Attention / gradient methods** | For deep nets; irrelevant to tree models. |

**Why `TreeExplainer` specifically?** For tree ensembles it computes *exact* Shapley values in
polynomial time (KernelExplainer is a slow model-agnostic approximation). The `LinearExplainer`
fallback handles the logistic-regression case.

**Production note:** SHAP is *built once and cached to joblib* at train time (`explainer.joblib`) so
the API doesn't rebuild it per request — a real latency optimisation.

---

## 9. The GenAI layer: LLM client & prompts

### The client — [`src/llm/client.py`](src/llm/client.py)

Three interchangeable implementations behind one `complete(system, user, max_tokens)` signature:

1. **`AnthropicLLM`** — direct Anthropic API (local/dev). Model default `claude-opus-4-7`.
2. **`BedrockLLM`** — AWS Bedrock (same Claude family) for production, *so no API keys ship into ECS*.
3. **`_DummyLLM`** — returns a canned response when `ANTHROPIC_API_KEY` is unset, keeping the pipeline
   runnable in CI/tests.

```python
resp = self.client.messages.create(
    model=self.model, max_tokens=...,
    system=[{"type": "text", "text": system,
             "cache_control": {"type": "ephemeral"}}],   # ← prompt caching
    messages=[{"role": "user", "content": user}],
)
```

**Engineering decisions worth internalising:**

- **The interface > the implementation.** Every provider exposes the identical `complete()` signature,
  so the agents never know (or care) which LLM is wired. Swapping Anthropic ↔ Bedrock ↔ a mock is a
  one-line factory change (`get_llm(provider=...)`). This is the **Strategy pattern**, and it's how you
  keep an app from being married to one vendor.
- **Prompt caching** (`cache_control: ephemeral`) caches the large, *reusable* system prompt so repeated
  calls only pay for the changing user message → real token-cost savings.
- **Graceful degradation** — no key ≠ crash. The dummy keeps tests green and lets new devs run the repo
  before they have credentials.
- **Token usage is returned** (`input_tokens`, `output_tokens`) and **accumulated through the pipeline**
  — you can't manage cost you don't measure.

### The prompts — [`src/llm/prompts.py`](src/llm/prompts.py)

The system prompt is itself a **guardrail document**. Excerpt from `RETENTION_SYSTEM`:

> - Always recommend ONE primary action and at most TWO supporting actions.
> - Be specific: name the offer, the discount %, the channel, and the owner.
> - Quantify expected impact.
> - **Use ONLY the information provided… If something is unknown, say "unknown" — never invent ticket
>   counts, NPS, or history.**
> - Output must follow the exact section headers in the user template.

**Why this matters:** the "never invent" rule is **anti-hallucination by construction**, and "follow
the exact headers" makes the free-text output *parseable* downstream (see §11's offer extraction). Good
prompt engineering = constraining the model into a contract.

---

## 10. RAG: retrieval over playbooks

**Why RAG instead of fine-tuning or stuffing everything in the prompt?**

- **Fine-tuning** bakes knowledge into weights — expensive, slow to update, and your retention
  policies change quarterly. You'd retrain on every policy edit.
- **Stuffing all docs in the prompt** blows the context window and cost, and buries the relevant
  passage in noise.
- **RAG** keeps knowledge in an external, *instantly editable* store. Edit a markdown file, re-ingest,
  done — no retraining. The LLM only sees the 3–4 *relevant* chunks per query.

### How it's built — [`src/rag/`](src/rag/)

**Knowledge base** = four markdown files (`customer_success_playbook.md`, `successful_campaigns.md`,
`discount_guidelines.md`, `retention_policies.md`) containing real retention policy: discount authority
matrices, eligibility gates, past-campaign results.

**Chunking** ([`knowledge_base.py`](src/rag/knowledge_base.py)) — **header-aware** splitting on `##`,
packed into ≤900-char chunks with 120-char overlap:

```python
def _split_markdown(text, max_chars=900, overlap=120):
    # split on H2 headers so a whole policy stays together; overlap so a policy
    # spanning a boundary is still retrievable as one idea.
```

*Why header-aware + overlap?* Naively chopping every N chars splits a policy mid-sentence and destroys
meaning. Splitting on semantic boundaries (headers) keeps each chunk self-contained; the overlap
prevents losing context that straddles a boundary.

**Embeddings** ([`retriever.py`](src/rag/retriever.py)) — `sentence-transformers/all-MiniLM-L6-v2`,
384-dim, `normalize_embeddings=True`.

**Vector store** — FAISS `IndexFlatIP`:

```python
index = faiss.IndexFlatIP(384)   # inner product on normalised vectors == cosine similarity
index.add(embs)
scores, ids = index.search(query_emb, k=4)   # top-4 chunks
```

**Decisions & alternatives:**

| Decision | Why | Alternative & when to switch |
| --- | --- | --- |
| **MiniLM-L6-v2 embeddings** | Tiny (80MB), fast, runs on CPU, good enough for short policy text. | OpenAI/Voyage/Cohere embeddings (better quality, but API cost + latency); larger BGE/E5 models for harder retrieval. |
| **`IndexFlatIP` (brute force)** | Exact search, zero tuning. The KB is ~50 chunks — brute force is *instant*. | `IndexIVFFlat` / `HNSW` for millions of vectors (approximate but sub-linear). Don't add complexity you don't need. |
| **Inner product on normalised vectors** | Equals cosine similarity, the standard text-similarity metric. | — |
| **FAISS (local file)** | No infra, fast, perfect for a fixed corpus. | Managed vector DB (Pinecone/Weaviate/pgvector) when you need filtering, multi-tenancy, or live updates at scale. |
| **k=4** | Enough context without flooding the prompt. | Tune empirically; add a re-ranker for precision. |

**Singleton loader** — the index is loaded **once** into a module-level singleton, not per request:
> "Lazily load a singleton so we don't reload the FAISS index per request."

---

## 11. The multi-agent system (LangGraph)

Rather than one giant prompt doing everything, the work is split into **five specialised agents**, each
with a narrow job, wired as a graph in [`src/agents/orchestrator.py`](src/agents/orchestrator.py).

### Shared state ([`src/agents/state.py`](src/agents/state.py))

A `TypedDict` (`CopilotState`) that each agent reads from and writes to — the customer_id goes in, and
profile/risk/explanation/retention/ROI fields accumulate, plus `errors`, `llm_usage`, and a `trace`.

### The graph

```python
g = StateGraph(CopilotState)
for name, fn in _PIPELINE: g.add_node(name, fn)
g.add_edge(START, "profile")
g.add_edge("profile", "risk")
g.add_edge("risk", "explanation")
g.add_edge("explanation", "retention")
g.add_edge("retention", "roi")
g.add_edge("roi", END)
return g.compile()
```

| Agent | Job | Key tech |
| --- | --- | --- |
| **Profile** | Load the customer 360 from data, build a compact summary | data loader |
| **Risk** | Run production model → probability; compute tiers + KMeans persona | churn model + segmentation |
| **Explanation** | SHAP top-5 drivers → Claude turns them into 2 plain-English sentences for a CSM | SHAP + LLM |
| **Retention** | RAG search (k=4) → Claude writes a structured action plan → regex maps it to a catalog offer key | RAG + LLM |
| **ROI** | Score the chosen offer + rank alternatives in dollars | ROI estimator |

**The offer-extraction bridge** (in `retention_agent`) is a subtle, important pattern — turning
free-text LLM output back into a structured key the deterministic ROI code can use:

```python
_OFFER_PATTERNS = [(re.compile(r"15\s*%.*12\s*month|loyalty.*lock", re.I), "loyalty_discount_15pct_12mo"), ...]
def _pick_offer_key(text, segment):
    for pat, key in _OFFER_PATTERNS:
        if pat.search(text): return key
    # deterministic fallback by segment if the LLM was vague
    return {"High Value + High Risk": "loyalty_discount_15pct_12mo"}.get(segment, "free_tech_support_6mo")
```

### Why multi-agent and not one mega-prompt?

- **Separation of concerns** — each agent is independently testable and swappable. A bug in ROI never
  touches risk scoring.
- **Mix deterministic + probabilistic** — risk/SHAP/ROI are *deterministic math* (auditable, exact);
  only explanation/retention call the LLM. You don't ask an LLM to do arithmetic it might hallucinate.
- **Observability** — the `trace` records per-node timing; `errors` accumulate so one failed node
  **degrades gracefully** instead of crashing the whole pipeline.
- **Reusability** — `/predict_churn` uses only the risk logic; `/copilot/run` uses the whole chain. Same
  building blocks, different compositions.

### Why LangGraph (vs. alternatives)?

- **vs. a plain function chain** — LangGraph gives explicit nodes/edges/state, built-in support for
  branching, loops, retries, and checkpointing as the workflow grows. (Notably, this project keeps a
  **`run_sequential` fallback** that runs the same pipeline as plain function calls if LangGraph isn't
  installed — a nice degradation story.)
- **vs. LangChain agents / ReAct** — those let the *LLM* decide which tool to call next
  (non-deterministic, can loop forever). Here the flow is **fixed and known**, so a hardcoded graph is
  *more reliable and cheaper* than letting an LLM route. Use autonomous agents only when the path
  genuinely can't be predetermined.
- **vs. CrewAI / AutoGen** — heavier multi-agent frameworks aimed at autonomous collaboration; overkill
  for a fixed 5-step pipeline.

> **Lesson:** "multi-agent" doesn't mean "let LLMs decide everything." Here it means *a structured
> pipeline of specialised steps*, most of which aren't even LLMs. Determinism where you can, LLM where
> you must.

---

## 12. ROI estimation: turning a probability into a dollar decision

[`src/models/roi_estimator.py`](src/models/roi_estimator.py) is what makes the system a *decision* tool.
A catalog of six offers, each with an **acceptance rate** and **retention lift** (calibrated from the
historical-campaigns KB):

```python
"loyalty_discount_15pct_12mo": {"cost_kind": "pct_of_monthly", "pct": 0.15, "months": 12,
                                "acceptance_rate": 0.55, "retention_lift": 0.40, ...}
```

The math:

```python
eff_lift              = min(acceptance_rate * retention_lift, baseline_churn_prob)  # can't save >100%
expected_revenue_saved = eff_lift * monthly_charge * horizon_months
cost                  = flat $ or monthly*pct*months
net_value             = expected_revenue_saved - cost
roi_multiple          = expected_revenue_saved / cost
payback_months        = cost / (eff_lift * monthly_charge)
```

Then `rank_offers()` sorts every offer by **net value** so the best action floats to the top.

**Why this matters:** a 78% churn probability is meaningless to a manager; *"this $144 offer saves an
expected $211 over 12 months, 1.47× ROI, 8-month payback"* is a decision. The
`eff_lift = min(accept × lift, baseline)` clamp is a small but important guardrail — **you can't reduce
churn below zero.**

**Production caveat:** the acceptance/lift numbers here are assumptions. In reality you'd estimate them
from **A/B tests / uplift modelling** (which customers are *persuadable* — "persuadables" vs. "sure
things" vs. "lost causes"). True ROI requires **causal** estimates, not just correlational churn scores.
That's the honest next step beyond this project.

---

## 13. Serving: the FastAPI layer

[`src/api/main.py`](src/api/main.py) exposes the system as a REST API.

| Method | Path | Purpose | Uses |
| --- | --- | --- | --- |
| GET | `/health` | Liveness probe (for ECS/k8s) | — |
| POST | `/predict_churn` | Probability + tiers + persona | model + segmentation |
| POST | `/customer_analysis` | SHAP driver explanation | SHAP |
| POST | `/generate_strategy` | RAG + LLM action plan | full copilot |
| POST | `/customer_roi` | ROI table (fast path if offer supplied) | ROI estimator |
| POST | `/copilot/run` | The whole pipeline in one call | LangGraph |

**Design choices:**

- **Pydantic schemas** ([`schemas.py`](src/api/schemas.py)) give automatic request validation, typed
  responses, and free OpenAPI/Swagger docs. Bad input is rejected at the boundary with a 422, never
  reaching your model.
- **404 on unknown customer** via `_profile_or_404` — explicit, correct HTTP semantics.
- **`_persona_for` never raises** — wrapped in try/except returning `"Unknown"`, so a missing
  segmentation bundle can't take down `/predict_churn`. **Defensive serving.**
- **Fast path vs. slow path** in `/customer_roi`: if the caller already knows the offer, skip the
  expensive LLM pipeline and just do the math. Don't pay for an LLM call you don't need.
- **Models loaded from disk** (`load_production()`, `load_preprocessor()`) — see §16 for why you'd cache
  these at startup in real production.

---

## 14. MLOps: MLflow tracking & the model registry

[`src/mlops/tracking.py`](src/mlops/tracking.py) + the training script log **every** run.

**What's tracked per model:** params, all six metrics, and the serialized model artifact. The best model
is then **registered and promoted**:

```python
mv = mlflow.register_model(model_uri, name="churn-prod")
client.transition_model_version_stage(name="churn-prod", version=mv.version,
                                       stage="Production", archive_existing_versions=True)
```

**Why MLflow / a registry at all?**

- **Reproducibility & comparison** — "which hyperparameters gave us 0.87 AUC last month?" is answerable.
- **The registry is the source of truth for *which* model is live.** `Production` stage decouples
  "trained" from "deployed." You promote a version; the API loads "whatever is Production" — rollback is
  just re-promoting the previous version.
- **`archive_existing_versions=True`** keeps exactly one Production model and auto-archives the old one.
- **Auditability** — regulated industries need to prove which model made which decision when.

**Belt-and-suspenders artifact strategy:** the project *also* dumps `production.joblib` to disk. MLflow
is the system-of-record; the joblib is the fast local load path. The `train_model.py` MLflow calls are
all wrapped in try/except so **training never fails just because the tracking server is down** —
tracking is observability, not a hard dependency of producing a model.

**Alternatives:** Weights & Biases, Neptune, DVC, SageMaker Model Registry. MLflow is the open-source
default — free, self-hostable, framework-agnostic.

---

## 15. Deployment: Docker & AWS

### Docker ([`deployment/docker/`](deployment/docker/))

`Dockerfile.api` is a textbook ML service image:

- **`python:3.11-slim`** base — small attack surface, small image.
- Installs only the system libs trees/FAISS/sentence-transformers need (`libgomp1`, `build-essential`).
- **`requirements.txt` copied and installed before source** → Docker layer caching means code changes
  don't reinstall all deps.
- **Artifacts are *mounted at runtime*, not baked in** — "so the same image can serve different model
  versions without a rebuild." Decouples code releases from model releases.
- **`HEALTHCHECK`** hits `/health` so the orchestrator knows when the container is truly ready.

`docker-compose.yml` runs the **full local stack**: API (8000) + Streamlit dashboard (8501) + an MLflow
server (5000), all sharing the `artifacts/` and `mlruns/` volumes.

### AWS ([`deployment/aws/`](deployment/aws/))

| Service | Role in this system |
| --- | --- |
| **S3** | Stores model artifacts + FAISS index (versioned bucket; old versions expire after 30 days). The API pulls them at boot. |
| **SageMaker** | Runs training as a managed job (`SKLearn` estimator, `ml.m5.xlarge`) — scalable, ephemeral compute; artifacts land in S3. |
| **ECS Fargate** | Runs the API containers serverlessly (`DesiredCount: 2` for HA, 1 vCPU / 2 GB each). |
| **ECR** | Private Docker registry, image scanning on push. |
| **Bedrock** | Hosts Claude inside AWS so **no API keys ship into tasks** (the task IAM role grants `bedrock:InvokeModel`). |
| **CloudWatch** | Centralised logs (30-day retention) + the basis for metrics/alarms. |
| **Secrets Manager** | `ANTHROPIC_API_KEY` injected as a secret reference (ECS task def), never hardcoded. |
| **CloudFormation** | Infrastructure-as-code — the whole stack (bucket, ECR, cluster, IAM, SG, service) is reproducible from `cloudformation.yaml`. |

This is a clean, conventional AWS ML-serving topology: **train on SageMaker → artifacts to S3 →
containers on Fargate pull artifacts → Bedrock for the LLM → CloudWatch for observability**, all defined
as code.

---

## 16. Scaling to production

What this repo does for a demo vs. what you'd change for real traffic:

| Concern | In the repo | Production change |
| --- | --- | --- |
| **Model loading** | `load_production()` reads joblib **per request** in some paths | Load **once at startup** into app state (FastAPI lifespan / dependency). Re-reading disk per request is the #1 latency killer. |
| **Single record inference** | One customer at a time | Add a **batch endpoint** + nightly batch scoring of the whole base into a table the dashboard reads. Most churn use-cases are batch, not real-time. |
| **Data source** | CSV / synthetic | Feature store (Feast/Tecton) so training and serving features are computed identically → kills training/serving skew. |
| **Vector store** | FAISS flat file | Managed vector DB (pgvector/Pinecone) once the KB grows or needs live updates/filtering. |
| **Concurrency** | `uvicorn --reload` | Multiple uvicorn/gunicorn workers behind a load balancer; ECS `DesiredCount` ≥ 2 + autoscaling on CPU/RPS. |
| **LLM latency/cost** | Synchronous calls | Cache repeated explanations, batch where possible, use a smaller/faster model for the explanation step, set strict `max_tokens`. |
| **Retraining** | Manual `python scripts/train_model.py` | Scheduled pipeline (Airflow/Step Functions/SageMaker Pipelines) triggered on a cadence *or* on drift detection. |
| **Caching** | Prompt caching only | Add a response cache (Redis) keyed on customer feature hash for hot customers. |

**Rule of thumb:** real-time churn scoring is usually *unnecessary* — you typically score the whole base
nightly and let CSMs work a ranked queue. Build the batch path first; add real-time only where a workflow
demands it.

---

## 17. Guardrails

Guardrails already present, and what you'd add:

**Already in the code:**
- **No-hallucination prompt rule** — "never invent ticket counts/NPS… say unknown."
- **`OneHotEncoder(handle_unknown="ignore")`** — unseen categories don't crash inference.
- **Pydantic validation** — malformed requests rejected at the API boundary.
- **Graceful degradation** — `_persona_for` and the MLflow calls swallow errors; the LLM has a dummy
  fallback; LangGraph has a sequential fallback.
- **ROI clamp** — `min(accept × lift, baseline)` and `max(..., 0)` keep outputs physically sane.
- **Deterministic core** — the money math and risk scoring are *not* LLM-generated, so they can't
  hallucinate.
- **Secrets via Secrets Manager / Bedrock**, never in code or images.

**What production demands on top:**
- **LLM output validation** — parse/validate the structured plan; if it doesn't match the required
  schema, retry or fall back to a template (don't show malformed output to a CSM).
- **PII / privacy** — the prompt rule "never reference the churn score or NPS with the customer" exists
  in the *policy*; enforce it in code for any customer-facing text. Mask PII before it hits the LLM.
- **Cost guardrails** — per-request and daily token budgets; circuit-breaker if spend spikes.
- **Rate limiting & authn/authz** on the API (currently CORS is wide-open `*` — fine for a demo, not
  prod).
- **Fairness checks** — churn models can encode bias (e.g. `SeniorCitizen`, `gender` are features).
  Audit that retention spend isn't unfairly distributed across protected groups.
- **Human-in-the-loop** — high-value offers should require approval (the policy's *discount authority
  matrix* literally encodes this — wire it into the app).

---

## 18. Performance tracking & monitoring

Two very different things must be monitored — **system** health and **model** health.

**System / service metrics (CloudWatch):** latency (p50/p95/p99), error rate, throughput, container
CPU/memory, **LLM token usage & cost** (already captured in `llm_usage`).

**Model metrics (the part teams forget):**

- **You don't get labels immediately.** You predict churn today; you learn if they actually churned in
  60–90 days. So you log every prediction, then **join to outcomes later** to compute *realised*
  ROC-AUC / PR-AUC over time.
- **Data drift** — has the input distribution moved from training? (e.g. `MonthlyCharges` shifts after a
  price change). Detect with PSI / KS-test per feature (Evidently, WhyLabs, SageMaker Model Monitor).
- **Concept drift** — the *relationship* between features and churn changed (a new competitor appears).
  Shows up as decaying live metrics even when inputs look stable → trigger retraining.
- **Prediction drift** — is the average predicted churn rate creeping up/down unexpectedly?
- **Business KPIs** — the only ones that ultimately matter: actual retention rate, revenue saved, offer
  acceptance vs. the *assumed* rates in the ROI catalog (close the loop and recalibrate those numbers!).
- **A/B testing** — measure the lift of *acting on* the model vs. a control group. A great model nobody
  acts on has zero ROI.

The repo's `trace` (per-node timing) and `training_summary.json` are the seeds of this; production wraps
them in dashboards + alerts.

---

## 19. The "perfect score" trap — a critical lesson

`artifacts/reports/training_summary.json` shows **every model scoring 1.0 on every metric** (perfect
ROC-AUC, precision, recall, F1).

**This is not a success — in real life it's a five-alarm fire.** A perfect score almost always means one
of:

1. **Label leakage** — a feature encodes the answer (e.g. accidentally including a "cancellation_date").
2. **The data is too easy / synthetic** — *this* is the cause here. The generator's churn signal
   (§2) is so strong relative to its noise that all four models separate the classes trivially.
3. **Train/test contamination** — test rows leaked into training.

**Why it happens here:** the synthetic generator bakes in a clean logistic relationship; even with the
Gaussian noise term, the behavioural fields (`nps_score`, `last_login_days`, `feature_usage_score`) are
drawn from *non-overlapping* distributions for churners vs. stayers, making the boundary nearly perfect.

**What a real engineer does when they see 1.0:**
- **Don't celebrate — investigate.** Assume leakage until proven otherwise.
- Drop suspected leaky features and re-score; expect real Telco churn to land around **0.84–0.86
  ROC-AUC**, never 1.0.
- Add noise / reduce class separation in the synthetic generator so the demo behaves realistically.
- The README's honest target — **"ROC-AUC > 0.85"** — is the *believable* number; the 1.0 in the report
  is an artifact of synthetic data and should be called out, not quoted.

> **This is the most valuable interview talking point in the whole repo:** noticing the 1.0, explaining
> *why* it's suspicious, and knowing what to do about it demonstrates real ML maturity far better than
> any architecture diagram.

---

## 20. Interview / design-review cheat sheet

**One-liner:** *"An end-to-end retention copilot: a gradient-boosted churn model whose predictions are
explained with SHAP, segmented by value × risk, and turned into a costed action plan by a 5-agent
LangGraph pipeline that grounds an LLM in retention playbooks via RAG — served on FastAPI, tracked in
MLflow, deployable to AWS."*

**Decisions you should be able to defend cold:**

| Decision | The defensible reason |
| --- | --- |
| XGBoost/LightGBM over deep learning | Tabular data, ~7k rows → GBMs beat neural nets and tune easier. |
| Logistic regression kept | Baseline to prove the complexity is worth it. |
| ROC-AUC / PR-AUC over accuracy | Imbalanced classes — accuracy rewards predicting "no churn" for everyone. |
| SHAP over feature_importances/LIME | Per-customer, additive, theoretically grounded, exact for trees. |
| RAG over fine-tuning | Policies change quarterly; edit a file, not the weights. |
| FAISS flat index | ~50 chunks → exact brute-force is instant; don't add ANN complexity prematurely. |
| LangGraph fixed graph over autonomous agents | The flow is known; determinism is cheaper and safer than letting an LLM route. |
| Determinism for math, LLM for language | Never let an LLM do arithmetic it can hallucinate. |
| Bedrock in prod | Keeps API keys out of containers; same Claude family. |
| MLflow registry + Production stage | Decouples "trained" from "deployed"; one-step rollback. |

**The three things that signal seniority:**
1. Spotting the **1.0 metrics** as a leakage/synthetic-data red flag (§19).
2. Knowing churn ROI needs **causal/uplift** estimates, not just correlational scores (§12).
3. Knowing you monitor **delayed labels + drift**, not just latency (§18).

---

*This guide documents the system as implemented in `churn-copilot/`. Code references are clickable and
point at the exact files. For the runnable commands, see the project [README](README.md).*

---

## 21. Likely interview questions — crisp answers

Direct, defensible answers to the questions you'll actually get. Each links back to the deeper section.

### Q1. Why XGBoost over Random Forest?

Both are tree ensembles, but they build them differently — and that difference is the whole answer:

| | Random Forest (bagging) | XGBoost (boosting) |
| --- | --- | --- |
| How trees combine | **Parallel** — many deep, independent trees on bootstrap samples, then averaged | **Sequential** — each shallow tree corrects the *residual errors* of the ones before it |
| What it reduces | **Variance** (averaging decorrelated trees) | **Bias** (each round fits what's still wrong) |
| Typical accuracy on tabular | Strong | Usually **state-of-the-art** |
| Tuning | Low effort, hard to overfit | More knobs (`learning_rate`, `n_estimators`, regularisation) but higher ceiling |

**The answer:** boosting's sequential error-correction extracts more signal from subtle feature
interactions than RF's averaging, so XGBoost almost always edges out Random Forest on structured churn
data. XGBoost also has **native regularisation** (`reg_lambda`), **built-in imbalance handling**
(`scale_pos_weight`), **early stopping** on a validation AUC, and a **histogram tree method** for speed —
none of which RF gives you. RF is still in this project as a **baseline** (it's robust and needs almost
no tuning), but it's the floor, not the ceiling. *Caveat to mention:* RF overfits less out-of-the-box and
is a safer choice if you have no time to tune — XGBoost's advantage only shows up once it's tuned.

> See §5 for the full four-model comparison.

### Q2. Why SHAP over Feature Importance?

Built-in `feature_importances_` is **global** ("contract type matters across the whole model") and
**inconsistent** (impurity-based importance is biased toward high-cardinality features). A CSM on a call
needs **"why is THIS customer at risk?"** — a *local, per-prediction* explanation. SHAP gives exactly
that: it's grounded in game theory (Shapley values), it's **additive**
(`base_value + Σ contributions = prediction`), it's **consistent**, and `TreeExplainer` computes it
**exactly and fast** for tree models. LIME is also local but uses a random local surrogate → unstable
and no theoretical guarantee. So: SHAP = local + consistent + exact for trees.

> See §8.

### Q3. Why FAISS?

The knowledge base is **~50 policy chunks**, fixed and small. FAISS `IndexFlatIP` does **exact**
brute-force cosine search over them in microseconds, with **zero infrastructure** (it's a local file) and
**zero tuning**. Reaching for a managed vector DB (Pinecone/Weaviate) or an approximate index
(HNSW/IVF) here would be **premature optimisation** — you'd pay in cost and ops complexity for a
scale problem you don't have. The honest engineering signal is knowing *when* to switch: move to ANN
indexes at millions of vectors, and to a managed DB when you need metadata filtering, multi-tenancy, or
live updates.

> See §10.

### Q4. How do you handle hallucinations?

Layered defence — the key idea is **don't let the LLM do anything you can verify deterministically**:

1. **Constrain the prompt** — the system prompt hard-rule: *"Use ONLY information provided… if something
   is unknown, say 'unknown' — never invent ticket counts, NPS, or history."* Anti-hallucination by
   construction.
2. **Ground every answer in RAG** — the retention plan is built from *retrieved* real policy chunks, not
   the model's parametric memory, and the API returns `retrieved_sources` so the answer is **traceable**.
3. **Keep the LLM away from the numbers** — risk scoring, SHAP, and all ROI math are **deterministic
   code**. The LLM only writes *language*; it never computes the probability or the dollar figures it's
   describing.
4. **Validate/parse the output** — the free-text plan is mapped back to a known offer key via regex with
   a **deterministic segment-based fallback** if the LLM is vague, so a hallucinated offer can't flow
   into ROI.
5. *(Production add-ons)* schema validation with retry, a faithfulness check (does the answer cite the
   retrieved context?), and human approval for high-value offers.

> See §9 and §17.

### Q5. How do you evaluate RAG quality? *(gap the main guide didn't cover — important)*

RAG quality splits into **retrieval** quality and **generation** quality; evaluate them *separately*
because a bad answer can come from either.

**Retrieval metrics** (build a small labelled set of query → which chunk(s) are correct):
- **Recall@k** — is the right chunk in the top-k? (Most important — if it's not retrieved, the LLM can't
  use it.) This is why `k=4` should be tuned, not guessed.
- **Precision@k / MRR / nDCG** — are the right chunks ranked near the top, with little noise?
- **Hit rate** — fraction of queries where *any* relevant chunk was retrieved.

**Generation / end-to-end metrics** (the "RAG triad", scorable with an LLM-as-judge):
- **Faithfulness / groundedness** — is every claim in the answer supported by the retrieved context?
  (Directly measures hallucination.)
- **Answer relevance** — does the answer actually address the query?
- **Context relevance** — were the retrieved chunks actually on-topic?

**How you'd run it here:** assemble ~30–50 representative customer queries with known-correct playbook
chunks → measure Recall@4 and MRR on retrieval → use an LLM judge (or human review) on faithfulness and
relevance of the final plan → track these over time as the KB grows. Tools: **Ragas**, **TruLens**,
**DeepEval**, or LangSmith. **The closing loop is the real metric:** did acting on the recommendation
improve retention? — measured by A/B test (§18).

### Q6. How would you scale to 1M customers?

Key realisation first: **you almost never need real-time scoring for churn.** At 1M customers:

- **Batch over real-time.** Run a nightly/weekly **batch scoring job** (Spark / SageMaker Batch
  Transform) that scores the whole base into a table; the dashboard and CSM queue read that table. The
  XGBoost forward pass over 1M rows is seconds-to-minutes — trivially parallelisable.
- **Don't run the LLM pipeline on all 1M.** Score *everyone* with the cheap model; only invoke the
  expensive RAG+LLM copilot for the **top-N highest-value-at-risk** customers a CSM will actually work
  (maybe a few thousand). This is the single biggest cost/scale lever.
- **Load models once, not per request** — into app state at startup (the repo currently reloads joblib
  in some paths; fix that first).
- **Feature store** (Feast/Tecton) so 1M feature vectors are precomputed and served consistently with
  training (kills training/serving skew).
- **Horizontal scaling** of the API behind a load balancer (ECS `DesiredCount` + autoscaling on RPS);
  the service is stateless so it scales linearly.
- **Vector store** stays trivial — the KB doesn't grow with customer count (it's policies, not
  customers), so FAISS is still fine. Only the *customer* table grows.
- **Async + queue** for the LLM steps (SQS + workers) so spikes don't block.

> See §16.

### Q7. How would you reduce inference cost?

Separate the two cost centres — the **ML model is nearly free**; the **LLM is the cost**:

**LLM cost (the 90%):**
- **Don't call it on everyone** — only the top-N at-risk customers (see Q6). Biggest lever by far.
- **Prompt caching** — already implemented (`cache_control: ephemeral`) so the large reusable system
  prompt isn't re-billed every call.
- **Right-size the model per task** — use a smaller/cheaper Claude (e.g. Haiku) for the simple
  2-sentence *explanation* step, reserve the larger model for the *retention plan*.
- **Cap `max_tokens`** and tighten prompts — shorter outputs, fewer input tokens (only retrieve k=4, not
  k=20).
- **Response caching (Redis)** keyed on a customer's feature hash — identical situations reuse a prior
  answer instead of re-generating.
- **Batch / off-peak** generation rather than synchronous on every dashboard load.

**ML model cost:**
- **Load once at startup**, cache the model + preprocessor + SHAP explainer in memory (the explainer is
  already pre-built to joblib at train time).
- **Batch inference** amortises overhead far better than one-row-at-a-time calls.
- **Quantisation / smaller models / ONNX** export if CPU-bound — but for a GBM on tabular data this is
  rarely the bottleneck; the LLM is.

> See §16's "LLM latency/cost" row.

---

*Together, §1–20 explain the system and §21 drills the questions. If you can answer the §20 cheat sheet
and the §21 seven cold, you understand this project end to end.*
