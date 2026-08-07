"""Download the MovieLens dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.preprocessing import download_movielens


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-1m", "ml-10m"],
        help="Which dataset to download",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Directory to extract data into (defaults to data/raw/<dataset>)",
    )
    args = parser.parse_args()
    target_dir = args.target_dir or Path(f"data/raw/{args.dataset}")
    download_movielens(target_dir, args.dataset)


if __name__ == "__main__":
    main()
