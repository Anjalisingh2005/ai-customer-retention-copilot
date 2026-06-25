"""Self-bootstrap: build trained artifacts on first run if they're missing.

The model files (production.joblib, explainer.joblib, the FAISS index, etc.) are
gitignored because they're large and regenerable. On a fresh deploy - e.g.
Streamlit Community Cloud, which only pulls the code from GitHub - those files
won't exist and the app would crash on load. This module rebuilds them once.

It trains only Logistic Regression + Random Forest (pure scikit-learn, no native
OpenMP dependency), which is enough for the dashboard and works on any host.
"""
from __future__ import annotations

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def artifacts_present() -> bool:
    """True only if every artifact the app loads at runtime already exists."""
    needed = [
        settings.model_dir / "production.joblib",
        settings.encoder_dir / "preprocessor.joblib",
        settings.model_dir / "segmentation.joblib",
        settings.model_dir / "explainer.joblib",
    ]
    return all(p.exists() for p in needed)


def ensure_artifacts() -> None:
    """Generate data + train models if artifacts are missing. Idempotent and cheap
    to call repeatedly (returns immediately once the files exist)."""
    if artifacts_present():
        return

    log.info("Model artifacts missing - bootstrapping (generate data + train).")
    from src.data.loader import load_enhanced
    from src.data.preprocessor import prepare, save_preprocessor
    from src.models.churn_predictor import save_model, save_production, train_logistic
    from src.models.segmentation import fit_kmeans, save_bundle
    from src.explainability.shap_explainer import ChurnExplainer, save_explainer_cache

    df = load_enhanced()
    prep = prepare(df)
    save_preprocessor(prep.preprocessor)

    # Logistic Regression only on the boot path: tiny model + lightweight SHAP
    # LinearExplainer, so it stays well within free-tier memory. (Its ROC-AUC
    # ~0.847 is within 0.003 of the RandomForest; run scripts/train_model.py
    # offline for the full four-model comparison.)
    result = train_logistic(prep.X_train, prep.y_train, prep.X_test, prep.y_test)
    save_model(result)
    save_production(result)

    save_bundle(fit_kmeans(df, n_clusters=4))
    save_explainer_cache(ChurnExplainer())
    log.info("Bootstrap complete - artifacts written.")


def ensure_faiss() -> None:
    """Build the RAG FAISS index if missing. Best-effort - never raises, so the
    dashboard still works even if embeddings can't be built on this host."""
    try:
        if (settings.faiss_index_dir / "kb.index").exists():
            return
        from src.rag.retriever import Retriever

        log.info("FAISS index missing - building from knowledge_base/.")
        Retriever().build()
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        log.warning("FAISS ingest skipped (RAG tab may be unavailable): %s", e)
