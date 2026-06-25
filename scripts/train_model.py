"""End-to-end training pipeline:

1. Load enhanced dataset.
2. Preprocess (encoders persisted).
3. Train LR / RF / XGBoost / LightGBM.
4. Log every run to MLflow with params + metrics.
5. Promote the best ROC-AUC model to artifacts/models/production.joblib AND
   register/transition in the MLflow model registry.
6. Fit segmentation bundle (KMeans personas).
7. Build and cache the SHAP explainer.
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from src.config import settings
from src.data.loader import load_enhanced
from src.data.preprocessor import prepare, save_preprocessor
from src.explainability.shap_explainer import ChurnExplainer, save_explainer_cache
from src.mlops import tracking as mlops
from src.models.churn_predictor import pick_best, save_model, save_production, train_all
from src.models.segmentation import fit_kmeans, save_bundle
from src.utils.logger import get_logger

log = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    df = load_enhanced(refresh=args.refresh_data)
    prep = prepare(df)
    save_preprocessor(prep.preprocessor)

    results = train_all(prep.X_train, prep.y_train, prep.X_test, prep.y_test)

    # MLflow logging.
    if not args.no_mlflow:
        for r in results:
            try:
                with mlops.run(name=r.name, tags={"phase": "churn-baseline"}):
                    mlops.log_params(r.params)
                    mlops.log_metrics(r.metrics)
                    try:
                        mlops.log_sklearn(r.model, artifact_path="model")
                    except Exception as e:
                        log.warning("MLflow model log failed for %s: %s", r.name, e)
            except Exception as e:
                log.warning("MLflow run skipped for %s: %s", r.name, e)

    # Persist all models + promote winner.
    for r in results:
        save_model(r)
    best = pick_best(results)
    save_production(best)

    if not args.no_mlflow:
        try:
            with mlops.run(name="promotion", tags={"phase": "promote"}):
                mlops.log_params({"chosen_model": best.name})
                mlops.log_metrics(best.metrics)
                uri = mlops.log_sklearn(best.model, artifact_path="production")
                mlops.register_and_promote(uri)
        except Exception as e:
            log.warning("Promotion step failed: %s", e)

    # Segmentation.
    bundle = fit_kmeans(df, n_clusters=4)
    save_bundle(bundle)

    # SHAP cache.
    try:
        explainer = ChurnExplainer()
        save_explainer_cache(explainer)
    except Exception as e:
        log.warning("Could not pre-build SHAP explainer (will rebuild on first request): %s", e)

    summary = {
        "best_model": best.name,
        "best_metrics": best.metrics,
        "all_metrics": {r.name: r.metrics for r in results},
    }
    (settings.report_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Training complete. Best model: %s (ROC-AUC=%.4f)", best.name, best.metrics["roc_auc"])


if __name__ == "__main__":
    main()
