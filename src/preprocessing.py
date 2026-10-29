#!/usr/bin/env python3

from pathlib import Path
import shutil
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from preprocessors import PREPROCESSORS
from datetime import datetime
import sys

try:
    preprocess_index = int(sys.argv[1])
except ValueError:
    preprocess_index = 0
except IndexError:
    preprocess_index = 0
PREPROCESSOR_NAME = [
    "noop", "bilateral", "clahe",
    "gamma", "sharpen",
    "chainA", "chainB", "chainC"
][preprocess_index]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_DIR = Path("data/building_dataset")
BACKUP_DIR = DATA_DIR.with_name(
    f"{DATA_DIR.name}.before_{PREPROCESSOR_NAME}.{TIMESTAMP}"
)

preprocessor = PREPROCESSORS[PREPROCESSOR_NAME]


def process_file(path: Path) -> str:
    with np.load(path) as data:
        image = data["image"]
        mask = data["mask"]

    processed_image = preprocessor(image)
    np.savez_compressed(path, image=processed_image, mask=mask)

    return path.name


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    print(f"Backing up {DATA_DIR} -> {BACKUP_DIR}")
    shutil.copytree(DATA_DIR, BACKUP_DIR)

    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"Found {len(files)} dataset files")

    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {workers} worker processes")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, filename in enumerate(
            executor.map(process_file, files),
            start=1,
        ):
            print(f"[{i}/{len(files)}] Processed {filename}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
