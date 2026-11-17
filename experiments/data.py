from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid

from .common import OUT, DATASET, SPLITS, csv_write, write_json, sha256_file

SOURCE = Path('data/PRE-event')
REFERENCE = Path('data/Germany_Training_Public_reference.csv')


def valid_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty and geom.area > 0 else []
    if isinstance(geom, (GeometryCollection, MultiPolygon)):
        return [p for sub in geom.geoms for p in valid_polygons(sub)]
    return []


def read_annotations():
    by_id = defaultdict(list)
    stats = {'building_rows': 0, 'bad_wkt': 0, 'empty_or_nonpolygon_rows': 0}
    with REFERENCE.open(newline='') as f:
        reader = csv.DictReader(f)
        expected = {'ImageId', 'Wkt_Pix', 'Object'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f'{REFERENCE} needs columns {expected}; found {reader.fieldnames}')
        for row in reader:
            if row['Object'].strip().lower() != 'building':
                continue
            stats['building_rows'] += 1
            image_id = row['ImageId'].strip()
            if not image_id:
                raise ValueError('Found a building annotation with an empty ImageId')
            geometry = (row['Wkt_Pix'] or '').strip()
            if not geometry:
                stats['empty_or_nonpolygon_rows'] += 1
                continue
            try:
                polys = valid_polygons(wkt.loads(geometry))
            except Exception:
                stats['bad_wkt'] += 1
                continue
            if not polys:
                stats['empty_or_nonpolygon_rows'] += 1
            by_id[image_id].extend(polys)
    return by_id, stats


def tile_group(image_id: str, block_size: int, mode: str):
    if mode == 'tile':
        return image_id
    # Germany tile convention: <scene>_<index>_<row>_<column>
    match = re.fullmatch(r'(.+)_(-?\d+)_(-?\d+)_(-?\d+)', image_id)
    if match is None:
        raise ValueError(f'Cannot derive spatial block from {image_id!r}; use --split-mode tile explicitly')
    prefix, idx, row, col = match.groups()
    return f'{prefix}_{idx}_block_{int(row)//block_size}_{int(col)//block_size}'


def make_splits(ids, seed=111, block_size=4, mode='spatial'):
    grouped = defaultdict(list)
    for image_id in ids:
        grouped[tile_group(image_id, block_size, mode)].append(image_id)
    if len(grouped) < 3:
        raise ValueError(f'Only {len(grouped)} spatial groups: reduce --block-size or use --split-mode tile')
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    n = len(ids)
    goal_test = max(1, round(n * 0.15))
    goal_val = max(1, round(n * 0.15))
    selected = {'train': [], 'val': [], 'test': []}
    # Collect complete groups for test, then validation. Train gets all remaining groups.
    for split, desired in (('test', goal_test), ('val', goal_val)):
        while groups and len(selected[split]) < desired and len(groups) > (2 if split == 'test' else 1):
            group = groups.pop()
            selected[split].extend(grouped[group])
    for group in groups:
        selected['train'].extend(grouped[group])
    if not all(selected.values()):
        raise ValueError('A split is empty; decrease block size')
    for v in selected.values():
        v.sort()
    if set(selected['train']) & set(selected['val']) or set(selected['val']) & set(selected['test']) or set(selected['train']) & set(selected['test']):
        raise AssertionError('Split overlap')
    assert set(sum(selected.values(), [])) == set(ids)
    groups_by_split = {split: sorted({tile_group(x, block_size, mode) for x in selected[split]}) for split in selected}
    return selected, groups_by_split


def prepare(seed=111, block_size=4, mode='spatial', force=False):
    if not REFERENCE.is_file() or not SOURCE.is_dir():
        raise FileNotFoundError(f'Expected {REFERENCE} and {SOURCE}/')
    tiffs = sorted(SOURCE.glob('*.tif'))
    if not tiffs:
        raise ValueError(f'No TIFFs in {SOURCE}')
    if DATASET.exists() and not force:
        raise FileExistsError(f'{DATASET} already exists; use --force only to deliberately rebuild')
    if force and (OUT / 'runs').exists():
        raise RuntimeError('Refusing to rebuild dataset while runs exist; archive or move experiments_output/runs first.')
    if force and DATASET.exists():
        import shutil
        shutil.rmtree(DATASET)
    DATASET.mkdir(parents=True, exist_ok=True)
    annotations, annotation_stats = read_annotations()
    ids = [x.stem for x in tiffs]
    missing = sorted(set(annotations) - set(ids))
    if missing:
        print(f'WARNING: {len(missing)} annotated ImageIds have no PRE TIFF (first five: {missing[:5]})')
    records = []
    for i, path in enumerate(tiffs, 1):
        with rasterio.open(path) as ds:
            if ds.count < 3:
                raise ValueError(f'{path}: fewer than 3 channels')
            if any(ds.dtypes[j] != 'uint8' for j in range(3)):
                raise ValueError(f'{path}: {ds.dtypes[:3]}; the experiment assumes uint8 RGB; do not divide 16-bit pixels by 255')
            image = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
            height, width = ds.height, ds.width
        polygons = annotations.get(path.stem, [])
        mask = rasterize([(p, 1) for p in polygons], out_shape=(height, width), fill=0, dtype=np.uint8)
        output = DATASET / f'{path.stem}.npz'
        np.savez_compressed(output, image=image, mask=mask)
        count = int(mask.sum())
        records.append({'id': path.stem, 'height': height, 'width': width,
                        'polygons': len(polygons), 'positive_pixels': count,
                        'positive_fraction': round(count/(width*height), 7),
                        'npz_sha256': sha256_file(output)})
        print(f'[{i:3}/{len(tiffs)}] {path.stem}: {len(polygons)} polygons; {count} building pixels', flush=True)
    splits, groups_by_split = make_splits(ids, seed, block_size, mode)
    write_json(SPLITS, {**splits, 'seed': seed, 'split_mode': mode, 'block_size': block_size,
                        'groups': groups_by_split})
    csv_write(OUT/'tiles.csv', records, list(records[0]))
    metadata = {'dataset': 'SpaceNet8 Germany_Training_Public', 'n_tiles': len(ids),
                'n_empty_tiles': sum(r['positive_pixels'] == 0 for r in records),
                'annotation_stats': annotation_stats, 'annotations_without_tiff': missing,
                'reference_sha256': sha256_file(REFERENCE), 'tiles': records,
                'note': 'Annotation Wkt_Pix in pixel coordinates; only Object=Building used.'}
    write_json(OUT/'manifest.json', metadata)
    print('Split sizes:', {k: len(v) for k,v in splits.items()})
    print('Created:', DATASET, OUT/'manifest.json', SPLITS)
    if metadata['n_empty_tiles'] == len(ids):
        raise RuntimeError('ALL masks empty: inspect CSV labels and image names before training')
    if metadata['n_empty_tiles'] > len(ids)//2:
        print('WARNING: >50% empty masks. Inspect annotations and compare with GeoJSONs.')
    return metadata


def inspect(top=10):
    manifest = json.loads((OUT/'manifest.json').read_text())
    splits = json.loads(SPLITS.read_text())
    print('Tiles:', manifest['n_tiles'], 'empty:', manifest['n_empty_tiles'])
    print('CSV annotation rows:', manifest['annotation_stats'])
    print('Train/val/test:', *(len(splits[s]) for s in ('train','val','test')))
    tiles = sorted(manifest['tiles'], key=lambda d: d['positive_pixels'], reverse=True)
    print('Highest building coverage:')
    for tile in tiles[:top]:
        print(f"  {tile['id']}: {tile['polygons']} polygons, {tile['positive_fraction']:.2%} pixels")
