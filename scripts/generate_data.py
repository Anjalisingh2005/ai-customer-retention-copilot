"""Generate the enhanced Telco dataset (synthesizing if Kaggle raw CSV missing)."""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.data.synthetic_generator import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=7043, help="Used only if no raw Telco CSV is found.")
    args = ap.parse_args()
    df = build_dataset(n=args.rows)
    print(f"Wrote enhanced dataset: {df.shape[0]} rows × {df.shape[1]} cols")


if __name__ == "__main__":
    main()
