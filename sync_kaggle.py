"""Optional Kaggle dataset sync for the gym assistant.

This script requires:
- Kaggle CLI installed (`pip install kaggle`)
- Kaggle API credentials available in `~/.kaggle/kaggle.json`
  or `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables

It downloads selected public datasets into `data/kaggle/`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATASETS = [
    "exercisedb/fitness-exercises-dataset",
    "niharika41298/gym-exercise-data",
    "gokulprasantht/nutrition-dataset",
    "smayanj/fitness-tracker-dataset",
]


def sync_dataset(slug: str, target_dir: Path) -> None:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure your API token."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [kaggle, "datasets", "download", "-d", slug, "-p", str(target_dir), "--unzip"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle fitness datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Kaggle dataset slug. Repeat to download multiple datasets.",
    )
    args = parser.parse_args()

    datasets = args.dataset or DEFAULT_DATASETS
    target_dir = Path("data") / "kaggle"

    for slug in datasets:
        print(f"Downloading {slug} -> {target_dir}")
        sync_dataset(slug, target_dir / slug.replace("/", "__"))

    print("Done. Restart the server or hit the /retrain endpoint to refresh the assistant.")


if __name__ == "__main__":
    main()
