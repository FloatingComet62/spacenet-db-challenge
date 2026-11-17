from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.strtree import STRtree
from PIL import Image

from .common import OUT, SPLITS, check_prepared, csv_write, read_json, load_tile, write_json
from .data import read_annotations
from .engine import metrics_from_counts


def load_prediction(pred_dir, image_id, kind='binary', threshold=.5, channel=0):
    pred_dir = Path(pred_dir)
    candidates = [pred_dir/f'{image_id}{suffix}{ext}' for suffix in ('_buildingpred','')
                  for ext in ('.tif','.tiff','.npy','.npz','.png')]
    matches = [p for p in candidates if p.is_file()]
    if len(matches) == 0:
        raise FileNotFoundError(f'No prediction for {image_id}; expected {image_id}_buildingpred.tif or {image_id}.npy/.tif/.png')
    if len(matches) > 1:
        raise ValueError(f'Multiple predictions for {image_id}: {matches}. Leave exactly one per tile.')
    path = matches[0]
    if path.suffix in ('.tif','.tiff'):
        with rasterio.open(path) as f:
            if f.count <= channel:
                raise ValueError(f'{path} contains {f.count} channels, requested channel={channel}')
            arr = f.read(channel+1)
    elif path.suffix == '.npy':
        arr = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            key = 'probabilities' if 'probabilities' in f else 'mask'
            if key not in f:
                raise ValueError(f'{path} needs key probabilities or mask')
            arr = f[key]
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = arr[channel]
        elif arr.shape[2] <= 8:
            arr = arr[:,:,channel]
        else:
            raise ValueError(f'Unexpected prediction shape {arr.shape}, expected HxW or small-channel CHW/HWC')
    if arr.ndim != 2:
        raise ValueError(f'{path}: expected a 2D prediction, found {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError(f'{path}: NaN/inf in prediction')
    if kind == 'binary':
        return arr > 0
    if kind == 'probabilities':
        arr = arr.astype(np.float32)
        if arr.max() > 1 and arr.max() <= 255 and np.issubdtype(np.asarray(arr).dtype, np.floating):
            raise ValueError('Probability images must be in [0,1]. Use --input-type binary for 0/255 masks.')
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError('Probabilities must be between 0 and 1')
        return arr >= threshold
    if kind == 'logits':
        return arr >= np.log(threshold/(1-threshold))
    raise ValueError(kind)


def instance_counts(pred, gt_polygons, min_area=16, match_iou=.5):
    # Polygonize predicted binary labels in *pixel* coordinates, not world coordinates.
    predicted = []
    for geojson,value in shapes(pred.astype(np.uint8),mask=pred.astype(bool)):
        if value != 1:
            continue
        poly = shape(geojson)
        if poly.area >= min_area and not poly.is_empty:
            predicted.append(poly)
    if not predicted or not gt_polygons:
        return 0,len(predicted),len(gt_polygons)
    tree = STRtree(gt_polygons)
    pairs = []
    for i,p in enumerate(predicted):
        for j in tree.query(p):
            g = gt_polygons[int(j)]
            inter = p.intersection(g).area
            if not inter:
                continue
            score = inter/(p.area+g.area-inter)
            if score >= match_iou:
                pairs.append((score,i,int(j)))
    used_pred, used_gt = set(),set()
    for score,i,j in sorted(pairs,reverse=True):
        if i not in used_pred and j not in used_gt:
            used_pred.add(i)
            used_gt.add(j)
    tp = len(used_pred)
    return tp,len(predicted)-tp,len(gt_polygons)-tp


def evaluate_external(pred_dir,split='test',kind='binary',threshold=.5,min_area=16,
                      instance=True,name='official_external',channel=0):
    check_prepared()
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be in (0,1)')
    if not name.replace('_','').replace('-','').isalnum():
        raise ValueError('Name must contain only letters, digits, underscore, hyphen')
    ids = read_json(SPLITS)[split]
    gt_instances,_ = read_annotations() if instance else ({},{})
    ptotal = np.zeros(3, np.int64)
    itotal = np.zeros(3, np.int64)
    rows=[]
    for i,image_id in enumerate(ids,1):
        _,gt = load_tile(image_id)
        p = load_prediction(pred_dir,image_id,kind,threshold,channel)
        if p.shape != gt.shape:
            raise ValueError(f'{image_id}: prediction {p.shape}, GT {gt.shape}. No silent resizing or geospatial warping.')
        gt = gt.astype(bool)
        tp,fp,fn = int((p&gt).sum()),int((p&~gt).sum()),int((~p&gt).sum())
        ptotal += (tp,fp,fn)
        row = {'id':image_id,**metrics_from_counts(tp,fp,fn)}
        if instance:
            itp,ifp,ifn = instance_counts(p,gt_instances.get(image_id,[]),min_area=min_area)
            itotal += (itp,ifp,ifn)
            row.update({'instance_tp':itp,'instance_fp':ifp,'instance_fn':ifn})
        rows.append(row)
        print(f'[{i}/{len(ids)}] {image_id}: IoU={row["iou"]:.4f}',flush=True)
    results = {'model':name,'split':split,'input_type':kind,'threshold':threshold,'tiles':len(ids),
               'pixel':metrics_from_counts(*ptotal)}
    if instance:
        tp,fp,fn = map(int,itotal)
        results['instance'] = {'f1_at_0_5':2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.,
                               'precision':tp/(tp+fp) if tp+fp else (1. if fn==0 else 0.),
                               'recall':tp/(tp+fn) if tp+fn else 1.,
                               'tp':tp,'fp':fp,'fn':fn, 'minimum_predicted_area_px':min_area,
                               'matching_iou':.5}
    dest = OUT/'external'/name
    write_json(dest/f'{split}_metrics.json',results)
    csv_write(dest/f'{split}_per_tile.csv',rows,list(rows[0]))
    print(f"Pixel IoU {results['pixel']['iou']:.4f}, Dice {results['pixel']['dice']:.4f}")
    if instance:
        print(f"Instance F1@0.5 {results['instance']['f1_at_0_5']:.4f}")
    print('Saved:',dest)
    return results


def export_official_csv(inputs):
    """Filter EXISTING official prepared CSVs to match our split; no fabricated road labels."""
    check_prepared()
    split = read_json(SPLITS)
    lookup = {image_id:which for which in ('train','val','test') for image_id in split[which]}
    rows = {'train':[],'val':[],'test':[]}
    seen=set()
    fieldnames=None
    for inp in inputs:
        with Path(inp).open(newline='') as f:
            reader=csv.DictReader(f)
            if 'preimg' not in (reader.fieldnames or []):
                raise ValueError(f'{inp}: official CSV needs preimg column')
            if fieldnames is None:
                fieldnames=reader.fieldnames
            elif fieldnames != reader.fieldnames:
                raise ValueError('CSV files have different columns')
            for row in reader:
                stem=Path(row['preimg']).stem
                if stem not in lookup:
                    continue
                if stem in seen:
                    raise ValueError(f'Official source has duplicate PRE tile: {stem}')
                seen.add(stem)
                rows[lookup[stem]].append(row)
    missing = set(lookup)-seen
    if missing:
        raise ValueError(f'Official CSVs missing {len(missing)} of our pre-event tiles, first: {sorted(missing)[:6]}. Run official data preparation with matching data first.')
    out=OUT/'official_csv'
    for which,data in rows.items():
        csv_write(out/f'{which}.csv',data,fieldnames)
        print(f'{which}: {len(data)} -> {out/f"{which}.csv"}')
    print('Official baseline can now train on its own building/road masks with the exact same tile split.')
