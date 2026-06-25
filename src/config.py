"""Central configuration. All paths and tunables live here so nothing is magic-string'd elsewhere."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]


def _path(env_var: str, default: str) -> Path:
    p = Path(os.getenv(env_var, default))
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-4-7")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # --- MLflow ---
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-copilot")
    mlflow_registered_model: str = os.getenv("MLFLOW_REGISTERED_MODEL", "churn-prod")

    # --- Paths ---
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    model_dir: Path = field(default_factory=lambda: _path("MODEL_DIR", "artifacts/models"))
    encoder_dir: Path = field(default_factory=lambda: _path("ENCODER_DIR", "artifacts/encoders"))
    report_dir: Path = field(default_factory=lambda: _path("REPORT_DIR", "artifacts/reports"))
    faiss_index_dir: Path = field(default_factory=lambda: _path("FAISS_INDEX_DIR", "artifacts/faiss"))
    knowledge_base_dir: Path = field(default_factory=lambda: ROOT / "knowledge_base")

    # --- ML ---
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    test_size: float = 0.2
    target_col: str = "Churn"
    id_col: str = "customerID"

    # --- API ---
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # --- AWS (optional) ---
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "raw" / "telco_churn.csv"

    @property
    def synthetic_data_path(self) -> Path:
        return self.data_dir / "synthetic" / "telco_enhanced.csv"

    @property
    def processed_data_path(self) -> Path:
        return self.data_dir / "processed" / "telco_processed.parquet"


settings = Settings()
