"""Utility functions for downloading files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from tqdm import tqdm

from engine.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from engine.utils.config import Config


logger = get_logger(__name__)


def download_file(url: str, dest: Path) -> None:
    """Download a file from a URL to a destination path with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s to %s", url, dest)

    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))

        with open(dest, "wb") as file, tqdm(
            desc=dest.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_bytes(chunk_size=8192):
                size = file.write(chunk)
                bar.update(size)


def download_release_asset(cfg: Config, filename: str, dest: Path) -> None:
    """Download a specific asset from the GitHub release configured in Config."""
    url = f"{cfg.assets.base_url}/{cfg.assets.release_tag}/{filename}"
    download_file(url, dest)
