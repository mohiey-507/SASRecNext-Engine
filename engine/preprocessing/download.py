from __future__ import annotations

import shutil
import urllib.request
import zipfile
from typing import TYPE_CHECKING

from engine.utils import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

_MOVIELENS_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
}

_ZIP_FOLDER_NAMES = {
    "ml-1m": "ml-1m",
    "ml-10m": "ml-10M100K",
}

def download_movielens(target_dir: Path, dataset: str = "ml-1m") -> None:
    """Download and extract MovieLens dataset if not already present."""
    extracted_folder_name = _ZIP_FOLDER_NAMES.get(dataset, dataset)
    extracted_dir = target_dir.parent / extracted_folder_name
    if extracted_dir.exists() and extracted_dir.resolve() != target_dir.resolve():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        extracted_dir.rename(target_dir)

    ratings_path = target_dir / "ratings.dat"
    if ratings_path.exists():
        logger.info("MovieLens %s already present at %s — skipping download", dataset, target_dir)
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir.parent / f"{dataset}.zip"
    url = _MOVIELENS_URLS[dataset]

    logger.info("Downloading MovieLens %s from %s ...", dataset, url)
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310

    logger.info("Extracting to %s ...", target_dir.parent)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir.parent)  # noqa: S202

    zip_path.unlink()

    if extracted_dir.exists() and extracted_dir.resolve() != target_dir.resolve():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        extracted_dir.rename(target_dir)

    logger.info("MovieLens %s download complete — zip removed", dataset)
