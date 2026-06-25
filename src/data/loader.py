"""Load enhanced customer data; emit a single customer lookup helper."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import settings
from src.data.synthetic_generator import build_dataset
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_enhanced(refresh: bool = False) -> pd.DataFrame:
    """Return the enhanced dataset. Build it if missing or refresh=True."""
    p = settings.synthetic_data_path
    if refresh or not p.exists():
        return build_dataset()
    log.info("Loading enhanced dataset from %s", p)
    return pd.read_csv(p)


def get_customer(customer_id: str) -> Optional[dict]:
    """Look up a single customer record as a plain dict."""
    df = load_enhanced()
    rec = df[df[settings.id_col] == customer_id]
    if rec.empty:
        return None
    return rec.iloc[0].to_dict()
