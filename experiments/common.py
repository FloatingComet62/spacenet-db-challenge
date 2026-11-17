from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

OUT = Path('experiments_output')
DATASET = OUT / 'dataset'
SPLITS = OUT / 'splits.json'
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def write_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(content, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path: Path, records: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepared_ids(split: str) -> list[str]:
    splits = read_json(SPLITS)
    return splits[split]


def check_prepared():
    if not SPLITS.is_file() or not (OUT / 'manifest.json').is_file():
        raise RuntimeError('First run: python exp.py prepare')
    manifest = read_json(OUT / 'manifest.json')
    splits = read_json(SPLITS)
    for split in ('train', 'val', 'test'):
        for image_id in splits[split]:
            path = DATASET / (image_id + '.npz')
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; rerun prepare with --force if intended')
    if manifest.get('reference_sha256') != sha256_file(Path('data/Germany_Training_Public_reference.csv')):
        raise RuntimeError('The reference CSV changed after prepare: data and splits must be rebuilt deliberately.')


def load_tile(image_id: str):
    path = DATASET / f'{image_id}.npz'
    with np.load(path) as npz:
        return npz['image'].copy(), npz['mask'].copy()


def patch_starts(n: int, size: int, stride: int) -> list[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


def normalize(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise TypeError(f'Expected uint8 RGB, got {image.dtype}; inspect raw TIFF bit depth')
    return np.moveaxis((image.astype(np.float32) / 255.0 - MEAN) / STD, -1, 0).copy()
