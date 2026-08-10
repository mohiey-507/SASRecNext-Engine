"""Standalone tool to download all configured assets from GitHub releases."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.utils import get_logger, load_config
from engine.utils.download import download_release_asset

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured assets from GitHub.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m/sasrecnext_tied.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not cfg.assets.files:
        logger.info("No assets configured to download in %s.", args.config)
        return

    logger.info("Downloading %d assets for %s", len(cfg.assets.files), cfg.assets.release_tag)

    for filename in cfg.assets.files:
        if filename.endswith(".pt"):
            dest = Path(cfg.data.checkpoint_dir) / filename
        else:
            dest = Path(cfg.data.processed_data_dir) / filename

        if dest.exists():
            logger.info("Asset %s already exists at %s. Skipping.", filename, dest)
            continue

        download_release_asset(cfg, filename, dest)

    logger.info("All assets downloaded successfully.")


if __name__ == "__main__":
    main()
