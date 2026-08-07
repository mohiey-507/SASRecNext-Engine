"""Build movie_catalog.json — one-time offline enrichment script.

Run order:
  1. scripts/preprocess.py      → id_mapping.json, movie_metadata.json
  2. scripts/build_movie_catalog.py → movie_catalog.json  (this script)

Requires TMDB_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from api.services.metadata.client import TMDBClient
from dotenv import load_dotenv
from engine.utils.config import load_config
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\s*\(\d{4}\)$")
_SAVE_INTERVAL = 100


def _clean_title(raw: str) -> str:
    return _YEAR_RE.sub("", str(raw)).strip()


def _extract_year(raw: str) -> str | None:
    m = re.search(r"\((\d{4})\)$", str(raw).strip())
    return m.group(1) if m else None


async def _build_catalog(
    client: TMDBClient,
    movies: pd.DataFrame,
    id_mapping: dict[str, dict[str, int]],
    movie_metadata: dict[str, dict[str, object]],
    out_path: Path,
) -> None:
    orig_to_internal: dict[str, int] = id_mapping["original_to_internal"]
    internal_to_orig: dict[str, int] = id_mapping["internal_to_original"]

    # Resume from partial output if it exists
    catalog: dict[str, Any] = {}
    if out_path.exists():
        catalog = json.loads(out_path.read_text())
        logger.info("Resuming — loaded %d existing catalog entries.", len(catalog.get("movies", {})))

    existing_movies: dict[str, Any] = catalog.get("movies", {})

    total = len(movies)
    pbar = tqdm(movies.iterrows(), total=total, desc="Building Catalog")
    for idx, row in pbar:
        ml_id_int = int(row["movie_id"])
        ml_id = str(ml_id_int)

        if ml_id in existing_movies:
            continue

        internal_id = orig_to_internal.get(ml_id)
        if internal_id is None:
            # Movie not in ML-1M interaction dataset — skip
            continue

        raw_title: str = str(row["title"])
        clean = _clean_title(raw_title)
        year = _extract_year(raw_title)

        # Retrieve ML-1M genres as fallback
        meta_entry = movie_metadata.get(str(internal_id), {})
        ml_genres: str = str(meta_entry.get("genres", ""))

        tmdb_id: int | None = None
        poster_path: str | None = None
        overview: str | None = None
        release_date: str | None = None
        tmdb_genres: str | None = None

        try:
            results = await client.search(clean, year=year)
            if not results and year:
                results = await client.search(clean)

            if results:
                match = results[0]
                tmdb_id = match.id
                pbar.set_postfix({"last_match": f"TMDB {tmdb_id}"})

                details = await client.get_details(tmdb_id)
                poster_path = details.poster_path
                overview = details.overview
                release_date = details.release_date
                if details.genres:
                    tmdb_genres = "|".join(details.genres)
            else:
                pass # Silent when not found to avoid spamming the progress bar
        except Exception as exc:  # noqa: BLE001
            logger.error("Error for '%s': %s — saving progress.", clean, exc)
            _save(out_path, orig_to_internal, internal_to_orig, existing_movies)
            await asyncio.sleep(5)
            continue

        existing_movies[ml_id] = {
            "title": raw_title,
            "clean_title": clean,
            "year": year,
            "genres": tmdb_genres if tmdb_genres else ml_genres,
            "tmdb_id": tmdb_id,
            "poster_path": poster_path,
            "overview": overview,
            "release_date": release_date,
        }

        if (int(str(idx)) + 1) % _SAVE_INTERVAL == 0:
            _save(out_path, orig_to_internal, internal_to_orig, existing_movies)

    _save(out_path, orig_to_internal, internal_to_orig, existing_movies)
    logger.info("Done — %d movies written to %s", len(existing_movies), out_path)


def _save(
    path: Path,
    orig_to_internal: dict[str, int],
    internal_to_orig: dict[str, int],
    movies: dict[str, object],
) -> None:
    catalog = {
        "id_mappings": {
            "original_to_internal": orig_to_internal,
            "internal_to_original": internal_to_orig,
        },
        "movies": movies,
    }
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build movie_catalog.json")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m.yaml"))
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("TMDB_API_KEY"):
        logger.error("TMDB_API_KEY is not set.")
        return

    cfg = load_config(args.config)
    processed_dir = Path(cfg.data.processed_data_dir)
    raw_dir = Path(cfg.data.raw_data_dir)

    id_mapping_path = processed_dir / "id_mapping.json"
    metadata_path = processed_dir / "movie_metadata.json"
    raw_movies_path = raw_dir / "movies.dat"
    out_path = processed_dir / "movie_catalog.json"

    for p in (id_mapping_path, metadata_path, raw_movies_path):
        if not p.exists():
            logger.error("Required file not found: %s", p)
            return

    id_mapping: dict[str, dict[str, int]] = json.loads(id_mapping_path.read_text())
    movie_metadata: dict[str, dict[str, object]] = json.loads(metadata_path.read_text())

    movies = pd.read_csv(
        raw_movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    client = TMDBClient(cfg)
    async with client:
        await _build_catalog(client, movies, id_mapping, movie_metadata, out_path)


if __name__ == "__main__":
    asyncio.run(main())
