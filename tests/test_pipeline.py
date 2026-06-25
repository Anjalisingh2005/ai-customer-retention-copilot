"""End-to-end pipeline smoke test.

This is a heavy test - it trains models, builds the FAISS index, and runs the
full 5-agent pipeline. Skip unless the dependencies are all installed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("shap")
pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")


def test_full_pipeline(tmp_path, monkeypatch):
    """Train tiny model + run copilot end-to-end on one customer."""
    from src.data.loader import load_enhanced
    from src.data.preprocessor import prepare, save_preprocessor
    from src.explainability.shap_explainer import ChurnExplainer, save_explainer_cache
    from src.models.churn_predictor import save_production, train_logistic
    from src.models.segmentation import fit_kmeans, save_bundle
    from src.rag.retriever import Retriever

    df = load_enhanced()
    prep = prepare(df)
    save_preprocessor(prep.preprocessor)
    res = train_logistic(prep.X_train, prep.y_train, prep.X_test, prep.y_test)
    save_production(res)
    save_bundle(fit_kmeans(df, n_clusters=4))
    save_explainer_cache(ChurnExplainer())
    Retriever().build()

    # Run the copilot on the first customer in the dataset.
    from src.agents.orchestrator import run as run_copilot

    cid = df.iloc[0]["customerID"]
    state = run_copilot(cid)
    assert "churn_probability" in state
    assert state.get("segment")
    assert state.get("recommended_roi") is not None
