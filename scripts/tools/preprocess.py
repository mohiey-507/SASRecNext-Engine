"""Preprocess MovieLens data into training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.data import DataPipeline
from engine.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens data")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)

    pipeline = DataPipeline(cfg.data)
    pipeline.ensure_data()


if __name__ == "__main__":
    main()
