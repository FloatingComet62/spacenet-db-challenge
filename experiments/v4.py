"""SpaceNet8 v4: validation-only, checkpoint-frozen building instance study.

The selected model, validation threshold and train/validation/test splits remain
untouched. All searches are on *validation* tiles, and test metrics are never
loaded. Requires scipy + scikit-image only for watershed proposals.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from .common import OUT, SPLITS, csv_write, read_json, write_json
from .data import read_annotations
from .engine import metrics_from_counts
from .v3 import (_plot, _valid_run, iter_val_probs, match_objects,
                 predicted_polygons, size_bin)

DEFAULT_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
DEFAULT_AREAS = (0, 8, 16, 32, 64)
DEFAULT_PEAKS = (6, 12, 20)


@dataclass
class Tile:
    image_id: str
    truth: np.ndarray
    probabilities: np.ndarray
    gt_polygons: list


def gt_polygons_for_tile(polygons, image_shape):
    """Match the clipping/splitting convention of v3.building_sizes."""
    h, w = image_shape
    frame = box(0, 0, w, h)
    result = []
    for poly in polygons:
        clipped = poly.intersection(frame)
        if clipped.is_empty or clipped.area <= 0:
            continue
        if clipped.geom_type == 'Polygon':
            result.append(clipped)
        elif hasattr(clipped, 'geoms'):
            result.extend(g for g in clipped.geoms if g.geom_type == 'Polygon' and g.area > 0)
    return result


def validation_tiles(run_dir, batch_size=2, device_name='auto', refresh=False):
    """Load only validation predictions, reusing v3's checkpoint-hashed cache."""
    annotation_map, _ = read_annotations()
    tiles = []
    for image_id, _, gt, probs in iter_val_probs(run_dir, batch_size, device_name, refresh):
        polys = gt_polygons_for_tile(annotation_map.get(image_id, []), gt.shape)
        tiles.append(Tile(image_id, gt, probs, polys))
    if not tiles:
        raise RuntimeError('No validation tiles found')
    return tiles


def filter_components(mask, min_area):
    """Remove complete 4-connected components smaller than min_area pixels."""
    if min_area < 0:
        raise ValueError('min_area must be >= 0')
    mask = np.asarray(mask, dtype=bool)
    if min_area <= 1 or not mask.any():
        return mask.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def watershed_labels(mask, min_distance=12):
    """Distance-transform watershed of a fixed binary foreground.

    Peak seeds are constrained to the foreground. Components without detected
    peaks receive an explicit fallback seed, so watershed never erases an
    otherwise valid foreground component. Watershed lines are disabled to keep
    pixel IoU unchanged unless a separate component-area filter is in effect.
    """
    if min_distance < 1:
        raise ValueError('min_distance must be positive')
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError as exc:
        raise RuntimeError('Watershed requires scipy and scikit-image: pip install scipy scikit-image') from exc
    dist = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(dist, min_distance=min_distance, threshold_abs=1.0,
                           labels=mask.astype(np.uint8), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(peaks, 1):
        markers[int(r), int(c)] = idx
    components, component_count = ndi.label(mask, structure=ndi.generate_binary_structure(2, 1))
    covered = set(int(x) for x in np.unique(components[markers > 0]))
    next_id = len(peaks) + 1
    for label_id, slices in enumerate(ndi.find_objects(components), 1):
        if slices is None or label_id in covered:
            continue
        roi_dist = dist[slices]
        roi_component = components[slices] == label_id
        pick = np.unravel_index(np.argmax(np.where(roi_component, roi_dist, -1.0)), roi_dist.shape)
        rr = slices[0].start + pick[0]
        cc = slices[1].start + pick[1]
        markers[rr, cc] = next_id
        next_id += 1
    separated = watershed(-dist, markers, mask=mask, watershed_line=False, connectivity=1)
    separated = np.asarray(separated, dtype=np.int32)
    if not np.array_equal(separated > 0, mask):
        raise AssertionError('Watershed unexpectedly changed the binary foreground')
    return separated


def instance_polygons(labels):
    """Extract polygons without merging watershed regions that share edges."""
    labels = np.asarray(labels, dtype=np.int32)
    if not np.any(labels):
        return []
    return [shape(geom) for geom, value in shapes(labels, mask=labels > 0, connectivity=4)
            if int(value) > 0]


def make_prediction(probs, method, threshold, min_area=16, peak_distance=None):
    """Return (pixel mask, list of predicted instance polygons).

    `raw` intentionally matches v3 exactly: min_area affects polygon matching,
    NOT the pixel IoU mask. `filter` removes small foreground components first.
    `watershed` splits instance labels without changing the filtered pixel mask.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError('threshold must be in (0,1)')
    binary = np.asarray(probs) >= threshold
    if method == 'raw':
        return binary, predicted_polygons(binary, min_area=min_area)
    if method not in ('filter', 'watershed'):
        raise ValueError(f'Unknown method: {method}')
    binary = filter_components(binary, min_area)
    if method == 'filter':
        return binary, predicted_polygons(binary, min_area=0)
    if peak_distance is None:
        raise ValueError('watershed needs peak_distance')
    return binary, instance_polygons(watershed_labels(binary, min_distance=peak_distance))


def counts(mask, gt, predicted, gt_polys, match_iou=.5):
    """(pixel TP/FP/FN, instance TP/FP/FN) on one original tile."""
    mask = np.asarray(mask, bool)
    gt = np.asarray(gt, bool)
    px = np.array([np.count_nonzero(mask & gt),
                   np.count_nonzero(mask & ~gt),
                   np.count_nonzero(~mask & gt)], dtype=np.int64)
    matches = match_objects(predicted, gt_polys, match_iou=match_iou)
    tp = len(matches)
    ob = np.array([tp, len(predicted) - tp, len(gt_polys) - tp], dtype=np.int64)
    return px, ob


def object_metrics(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {'instance_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'instance_precision': tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0),
            'instance_recall': tp / (tp + fn) if tp + fn else 1.0,
            'instance_tp': tp, 'instance_fp': fp, 'instance_fn': fn}


def score_configuration(tiles, config, per_tile=False):
    px_total, ob_total = np.zeros(3, np.int64), np.zeros(3, np.int64)
    rows = []
    for tile in tiles:
        mask, objects = make_prediction(tile.probabilities, config['method'],
                                        config['threshold'], config['min_area'],
                                        config.get('peak_distance'))
        px, ob = counts(mask, tile.truth, objects, tile.gt_polygons)
        px_total += px
        ob_total += ob
        if per_tile:
            pm = metrics_from_counts(*map(int, px))
            om = object_metrics(*map(int, ob))
            rows.append({'tile_id': tile.image_id, 'pixel_iou': pm['iou'],
                         'pixel_tp': pm['tp'], 'pixel_fp': pm['fp'], 'pixel_fn': pm['fn'],
                         **om})
    pixel = metrics_from_counts(*map(int, px_total))
    objs = object_metrics(*map(int, ob_total))
    result = {'method': config['method'], 'threshold': float(config['threshold']),
              'min_area': int(config['min_area']),
              'peak_distance': config.get('peak_distance'),
              'pixel_iou': pixel['iou'], 'pixel_dice': pixel['dice'],
              'pixel_precision': pixel['precision'], 'pixel_recall': pixel['recall'],
              **objs}
    return (result, rows) if per_tile else result


def _sort_key(row):
    # Prefer F1; use pixel IoU, then simpler postprocessing in ties.
    complexity = {'raw': 0, 'filter': 1, 'watershed': 2}[row['method']]
    return (row['instance_f1'], row['pixel_iou'], -complexity)


def _config_from_row(row):
    return {key: row.get(key) for key in ('method', 'threshold', 'min_area', 'peak_distance')}


def select_best(rows, baseline_iou, max_iou_drop=0):
    """Choose best validation instance F1 with a constrained global pixel-IoU drop."""
    if max_iou_drop < 0:
        raise ValueError('max_iou_drop must be nonnegative')
    for row in rows:
        row['eligible'] = bool(row['pixel_iou'] + 1e-12 >= baseline_iou - max_iou_drop)
    eligible = [r for r in rows if r['eligible']]
    if not eligible:
        raise RuntimeError('Nothing meets pixel IoU constraint; baseline must always be included')
    return max(eligible, key=_sort_key)


def _plot_study(rows, baseline, selected, dest):
    plt = _plot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    marks = {'raw': 'o', 'filter': 's', 'watershed': '^'}
    for method in marks:
        group = [r for r in rows if r['method'] == method]
        if group:
            axes[0].scatter([r['pixel_iou'] for r in group],
                            [r['instance_f1'] for r in group],
                            marker=marks[method], alpha=.65, s=36, label=method)
    axes[0].scatter([baseline['pixel_iou']], [baseline['instance_f1']],
                    marker='D', s=110, edgecolors='black', facecolors='none',
                    linewidths=1.5, label='v3 reference')
    axes[0].scatter([selected['pixel_iou']], [selected['instance_f1']],
                    marker='*', s=190, c='red', label='selected')
    axes[0].set_xlabel('Validation global pixel IoU')
    axes[0].set_ylabel('Validation instance F1 @ polygon IoU 0.50')
    axes[0].set_title('Pixel / instance tradeoff')
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for method in ('raw', 'filter', 'watershed'):
        group = sorted([r for r in rows if r['method'] == method],
                       key=lambda x: x['threshold'])
        if group:
            axes[1].scatter([r['threshold'] for r in group], [r['instance_f1'] for r in group],
                            marker=marks[method], s=22, alpha=.7, label=method)
    axes[1].axvline(baseline['threshold'], linestyle='--', color='gray',
                    label=f'Saved pixel threshold {baseline["threshold"]:.2f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Instance F1')
    axes[1].set_title('Instance F1 versus probability threshold')
    axes[1].grid(alpha=.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(dest / 'validation_postprocess_tradeoff.png', dpi=140)
    plt.close(fig)


def postprocess_study(run_dir, batch_size=2, device_name='auto', refresh=False,
                      thresholds=DEFAULT_THRESHOLDS, areas=DEFAULT_AREAS,
                      peak_distances=DEFAULT_PEAKS, watershed_top=2,
                      max_iou_drop=0.0):
    """Grid-search postprocessing exclusively against validation masks."""
    run_dir, cfg, tuned, ids = _valid_run(run_dir)
    if watershed_top < 0 or watershed_top > 10:
        raise ValueError('watershed_top must be in 0..10')
    ts = sorted({float(t) for t in thresholds} | {float(tuned)})
    ar = sorted({int(a) for a in areas})
    peaks = sorted({int(d) for d in peak_distances})
    if not ts or not ar or any(not 0 < t < 1 for t in ts):
        raise ValueError('Thresholds must lie in (0,1)')
    if any(a < 0 for a in ar) or any(d < 1 for d in peaks):
        raise ValueError('Areas must be >= 0 and peak distances >= 1')
    if max_iou_drop < 0:
        raise ValueError('max-iou-drop must be nonnegative')
    tiles = validation_tiles(run_dir, batch_size, device_name, refresh)
    baseline_config = {'method': 'raw', 'threshold': tuned, 'min_area': 16, 'peak_distance': None}
    base = score_configuration(tiles, baseline_config)
    base['stage'] = 'baseline'
    rows = [base]
    print(f'v3 reference: pixel IoU={base["pixel_iou"]:.4f}, '
          f'instance F1={base["instance_f1"]:.4f} ({len(ids)} validation tiles)', flush=True)
    for threshold in ts:
        config = {'method': 'raw', 'threshold': threshold, 'min_area': 16}
        if abs(threshold - tuned) < 1e-8:
            continue
        result = score_configuration(tiles, config)
        result['stage'] = 'threshold'
        rows.append(result)
        print(f'  threshold {threshold:.2f}: F1={result["instance_f1"]:.4f}, '
              f'pixel IoU={result["pixel_iou"]:.4f}', flush=True)
    for threshold in ts:
        for area in ar:
            config = {'method': 'filter', 'threshold': threshold, 'min_area': area}
            result = score_configuration(tiles, config)
            result['stage'] = 'area_filter'
            rows.append(result)
            print(f'  area threshold={threshold:.2f}, minimum={area}: '
                  f'F1={result["instance_f1"]:.4f}, pixel IoU={result["pixel_iou"]:.4f}', flush=True)
    if watershed_top:
        # Spend watershed compute only on the strongest *eligible* starting points.
        select_best(rows, base['pixel_iou'], max_iou_drop)
        candidates = sorted([r for r in rows if r['eligible']], key=_sort_key, reverse=True)
        selected_binary = []
        for candidate in candidates:
            key = (candidate['threshold'], candidate['min_area'] if candidate['method'] == 'filter' else 0)
            if key not in selected_binary:
                selected_binary.append(key)
            if len(selected_binary) >= watershed_top:
                break
        for threshold, area in selected_binary:
            for peak in peaks:
                config = {'method': 'watershed', 'threshold': threshold,
                          'min_area': area, 'peak_distance': peak}
                result = score_configuration(tiles, config)
                result['stage'] = 'watershed'
                rows.append(result)
                print(f'  watershed threshold={threshold:.2f}, area={area}, peak={peak}: '
                      f'F1={result["instance_f1"]:.4f}, pixel IoU={result["pixel_iou"]:.4f}', flush=True)
    selected = select_best(rows, base['pixel_iou'], max_iou_drop)
    dest = run_dir / 'diagnostics_v4'
    dest.mkdir(parents=True, exist_ok=True)
    csv_write(dest / 'validation_postprocess_grid.csv', rows, list(rows[0]) + ['eligible'])
    _, base_tiles = score_configuration(tiles, baseline_config, per_tile=True)
    _, selected_tiles = score_configuration(tiles, _config_from_row(selected), per_tile=True)
    paired = []
    for a, b in zip(base_tiles, selected_tiles):
        assert a['tile_id'] == b['tile_id']
        paired.append({'tile_id': a['tile_id'],
                       'baseline_pixel_iou': a['pixel_iou'],
                       'selected_pixel_iou': b['pixel_iou'],
                       'delta_pixel_iou': b['pixel_iou'] - a['pixel_iou'],
                       'baseline_instance_f1': a['instance_f1'],
                       'selected_instance_f1': b['instance_f1'],
                       'delta_instance_f1': b['instance_f1'] - a['instance_f1'],
                       'baseline_instance_tp': a['instance_tp'],
                       'selected_instance_tp': b['instance_tp'],
                       'gt_instances': a['instance_tp'] + a['instance_fn']})
    csv_write(dest / 'validation_paired_tiles.csv', paired, list(paired[0]))
    report = {'split': 'val', 'run': run_dir.name, 'validation_tiles': len(ids),
              'split_sha256': cfg['split_sha256'],
              'checkpoint_sha256': __import__('hashlib').sha256((run_dir/'best.pt').read_bytes()).hexdigest(),
              'match_iou': .5, 'baseline_min_pred_area_px2': 16,
              'pixel_iou_max_allowed_drop': float(max_iou_drop),
              'baseline': base, 'selected': selected,
              'delta_instance_f1': selected['instance_f1'] - base['instance_f1'],
              'delta_pixel_iou': selected['pixel_iou'] - base['pixel_iou'],
              'evaluated_configurations': len(rows),
              'note': 'Validation-only hyperparameter selection; test split not accessed. '
                      'This uses one-to-one polygon matching at IoU >= 0.50, '
                      'not the complete SpaceNet 8 leaderboard score.'}
    write_json(dest / 'validation_selection.json', report)
    _plot_study(rows, base, selected, dest)
    print(f'v4 SELECTED: {selected["method"]} threshold={selected["threshold"]:.2f} '
          f'min_area={selected["min_area"]} peak_distance={selected["peak_distance"]}')
    print(f'  Instance F1: {base["instance_f1"]:.4f} -> {selected["instance_f1"]:.4f} '
          f'({report["delta_instance_f1"]:+.4f})')
    print(f'  Pixel IoU  : {base["pixel_iou"]:.4f} -> {selected["pixel_iou"]:.4f} '
          f'({report["delta_pixel_iou"]:+.4f})')
    print('Saved:', dest / 'validation_selection.json')
    return report


def failure_records(tile, objects, match_iou=.5, small_max=100, medium_max=1000):
    """Per-GT-object explanation features, using actual polygon intersections."""
    gt = tile.gt_polygons
    matched = match_objects(objects, gt, match_iou=match_iou)
    tree = STRtree(objects) if objects else None
    out = []
    for j, g in enumerate(gt):
        overlaps = []
        if tree is not None:
            for idx in tree.query(g):
                idx = int(idx)
                p = objects[idx]
                inter = float(p.intersection(g).area)
                if inter > 0:
                    iou = inter / (p.area + g.area - inter)
                    overlaps.append((idx, inter, float(iou)))
        overlaps.sort(key=lambda v: v[2], reverse=True)
        best_iou = overlaps[0][2] if overlaps else 0.0
        best_coverage = overlaps[0][1] / g.area if overlaps else 0.0
        covered = min(1.0, sum(v[1] for v in overlaps) / g.area) if g.area else 0.0
        substantial = [v for v in overlaps if v[1] / g.area >= 0.10]
        merged = False
        if overlaps:
            best_pred = objects[overlaps[0][0]]
            for k, other in enumerate(gt):
                if k != j and best_pred.intersects(other):
                    if float(best_pred.intersection(other).area) / other.area >= .10:
                        merged = True
                        break
        if j in matched:
            reason = 'matched'
        elif not overlaps:
            reason = 'no_predicted_overlap'
        elif covered < .10:
            reason = 'very_low_coverage'
        elif merged:
            reason = 'possible_merged_buildings'
        elif len(substantial) >= 2:
            reason = 'possible_fragmentation'
        else:
            reason = 'below_iou_0_50'
        minx, miny, maxx, maxy = g.bounds
        out.append({'tile_id': tile.image_id, 'gt_index': j,
                    'area_px2': round(float(g.area), 3),
                    'size_bin': size_bin(float(g.area), small_max, medium_max),
                    'matched': int(j in matched), 'reason': reason,
                    'best_pred_iou': round(best_iou, 5),
                    'best_pred_coverage': round(best_coverage, 5),
                    'combined_coverage': round(covered, 5),
                    'overlapping_pred_objects': len(overlaps),
                    'min_x': round(minx, 2), 'min_y': round(miny, 2),
                    'max_x': round(maxx, 2), 'max_y': round(maxy, 2)})
    return out


def _draw_crop(ax, tile, pred_mask, gt_poly, title, half_width=96):
    from rasterio.features import rasterize
    from .common import load_tile
    image, _ = load_tile(tile.image_id)
    h, w = tile.truth.shape
    cx = (gt_poly.bounds[0] + gt_poly.bounds[2]) / 2
    cy = (gt_poly.bounds[1] + gt_poly.bounds[3]) / 2
    radius = max(half_width, (gt_poly.bounds[2] - gt_poly.bounds[0]) / 2 + 22,
                 (gt_poly.bounds[3] - gt_poly.bounds[1]) / 2 + 22)
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius))
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius))
    gt_roi = tile.truth[y0:y1, x0:x1]
    pr_roi = pred_mask[y0:y1, x0:x1]
    frame = image[y0:y1, x0:x1].astype(np.float32)
    tp, fp, fn = pr_roi & gt_roi, pr_roi & ~gt_roi, ~pr_roi & gt_roi
    frame[tp] = .65 * frame[tp] + .35 * np.array([0,255,0])
    frame[fp] = .55 * frame[fp] + .45 * np.array([255,0,0])
    frame[fn] = .55 * frame[fn] + .45 * np.array([0,60,255])
    target = rasterize([(gt_poly, 1)], out_shape=(h,w), dtype=np.uint8)
    boundary = cv2.morphologyEx(target[y0:y1,x0:x1], cv2.MORPH_GRADIENT,
                               np.ones((3,3),dtype=np.uint8)) > 0
    frame[boundary] = [255,255,0]  # highlighted GT object boundary
    ax.imshow(np.uint8(np.clip(frame, 0, 255)))
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def instance_audit(run_dir, selection='best', batch_size=2, device_name='auto',
                   refresh=False, count=12, small_max=100, medium_max=1000):
    if count < 1 or count > 40:
        raise ValueError('count must be 1..40')
    if not 0 < small_max < medium_max:
        raise ValueError('Expected 0 < small_max < medium_max')
    run_dir, _, tuned, _ = _valid_run(run_dir)
    dest = run_dir / 'diagnostics_v4'
    if selection == 'best':
        path = dest / 'validation_selection.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; run postprocess-study first')
        report = read_json(path)
        if report['split_sha256'] != hashlib.sha256(SPLITS.read_bytes()).hexdigest():
            raise RuntimeError('Selected postprocessing configuration has a different split hash')
        if report['checkpoint_sha256'] != hashlib.sha256((run_dir/'best.pt').read_bytes()).hexdigest():
            raise RuntimeError('Selected postprocessing configuration has a different checkpoint hash')
        config = _config_from_row(report['selected'])
    elif selection == 'baseline':
        config = {'method': 'raw', 'threshold': tuned, 'min_area': 16}
    else:
        raise ValueError('selection must be best or baseline')
    all_rows, tile_preds = [], []
    for tile in validation_tiles(run_dir, batch_size, device_name, refresh):
        mask, objects = make_prediction(tile.probabilities, **config)
        rows = failure_records(tile, objects, small_max=small_max, medium_max=medium_max)
        all_rows.extend(rows)
        tile_preds.append((tile, mask, rows))
    if not all_rows:
        raise RuntimeError('No annotated GT building objects on validation split')
    outdir = dest / f'{selection}_instance_audit'
    outdir.mkdir(parents=True, exist_ok=True)
    csv_write(outdir / 'validation_per_building.csv', all_rows, list(all_rows[0]))
    summary = []
    for label in ('small', 'medium', 'large'):
        grp = [x for x in all_rows if x['size_bin'] == label]
        hits = sum(x['matched'] for x in grp)
        summary.append({'size_bin': label, 'objects': len(grp), 'matched': hits,
                        'missed': len(grp)-hits,
                        'recall': hits / len(grp) if grp else None})
    csv_write(outdir / 'validation_size_recall.csv', summary, list(summary[0]))
    # Select diverse failures with a preference for tiny and complete misses.
    by_key = {(tile.image_id, row['gt_index']): (tile, mask, row)
              for tile, mask, rows in tile_preds for row in rows}
    failures = [r for r in all_rows if not r['matched']]
    reason_order = {'no_predicted_overlap':0, 'very_low_coverage':1,
                    'possible_merged_buildings':2, 'possible_fragmentation':3,
                    'below_iou_0_50':4}
    failures.sort(key=lambda r:(0 if r['size_bin']=='small' else 1 if r['size_bin']=='medium' else 2,
                                reason_order[r['reason']], r['best_pred_iou'], -r['area_px2']))
    show = failures[:count]
    if show:
        plt = _plot()
        columns = 3
        fig, axes = plt.subplots((len(show)+columns-1)//columns, columns,
                                 figsize=(14, 4.4*((len(show)+columns-1)//columns)), squeeze=False)
        for n, row in enumerate(show):
            ax = axes.flat[n]
            tile, mask, _ = by_key[(row['tile_id'],row['gt_index'])]
            gt = tile.gt_polygons[row['gt_index']]
            title = (f'{row["tile_id"][-12:]} | {row["size_bin"]} | {row["reason"]}\n'
                     f'area={row["area_px2"]:.0f} | bestIoU={row["best_pred_iou"]:.2f} '
                     f'| coverage={row["combined_coverage"]:.2f}')
            _draw_crop(ax, tile, mask, gt, title)
        for ax in axes.flat[len(show):]:
            ax.axis('off')
        fig.suptitle(f'Validation unmatched GT buildings | {run_dir.name} | {selection}\n'
                     'green TP, red FP, blue FN, yellow selected GT building', fontsize=12)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(outdir / 'validation_instance_failures.png', dpi=130)
        plt.close(fig)
    tally = {}
    for r in all_rows:
        tally[r['reason']] = tally.get(r['reason'],0) + 1
    write_json(outdir / 'validation_failure_summary.json',
               {'split':'val', 'run':run_dir.name, 'selection':selection,
                'config':config,'by_reason':tally,'by_size':summary,
                'note':'Reason categories are heuristic diagnostics, not verified causes.'})
    print(f'{run_dir.name} ({selection}) unmatched {len(failures)}/{len(all_rows)} '
          f'validation GT buildings')
    for label in summary:
        print(f'  {label["size_bin"]}: {label["matched"]}/{label["objects"]} '
              f'recall={label["recall"]}')
    print('  Failure categories:',tally)
    print('Saved:',outdir / 'validation_per_building.csv')
    if show:
        print('Saved:',outdir / 'validation_instance_failures.png')
    return summary,tally


def compare_external_instances(run_dir, pred_dir, selection='best', kind='binary',
                               external_threshold=.5, channel=0,
                               batch_size=2, device_name='auto', refresh=False):
    """Compare external binary building predictions with the same v4 validation GT."""
    from .compare import load_prediction
    run_dir, _, tuned, ids = _valid_run(run_dir)
    if not 0 < external_threshold < 1:
        raise ValueError('external-threshold must be in (0,1)')
    if selection == 'best':
        path = run_dir / 'diagnostics_v4' / 'validation_selection.json'
        if not path.is_file():
            raise FileNotFoundError('Run postprocess-study first or use --selection baseline')
        saved = read_json(path)
        if saved['split_sha256'] != hashlib.sha256(SPLITS.read_bytes()).hexdigest():
            raise RuntimeError('Saved split differs')
        if saved['checkpoint_sha256'] != hashlib.sha256((run_dir/'best.pt').read_bytes()).hexdigest():
            raise RuntimeError('Saved checkpoint differs')
        config = _config_from_row(saved['selected'])
    elif selection == 'baseline':
        config = {'method':'raw','threshold':tuned,'min_area':16}
    else:
        raise ValueError('selection must be best or baseline')
    internal_px = np.zeros(3,np.int64)
    external_px = np.zeros(3,np.int64)
    internal_ob = np.zeros(3,np.int64)
    external_ob = np.zeros(3,np.int64)
    rows = []
    for tile in validation_tiles(run_dir, batch_size, device_name, refresh):
        own_mask, own_objs = make_prediction(tile.probabilities, **config)
        ext_mask = load_prediction(pred_dir,tile.image_id,kind,external_threshold,channel)
        if ext_mask.shape != tile.truth.shape:
            raise ValueError(f'{tile.image_id}: external mask size does not match validation GT')
        ext_objs = predicted_polygons(ext_mask,min_area=16)
        ia, ib = counts(own_mask,tile.truth,own_objs,tile.gt_polygons)
        ea, eb = counts(ext_mask,tile.truth,ext_objs,tile.gt_polygons)
        internal_px += ia
        external_px += ea
        internal_ob += ib
        external_ob += eb
        rows.append({'tile_id': tile.image_id,
                     'ours_pixel_iou': metrics_from_counts(*map(int,ia))['iou'],
                     'external_pixel_iou': metrics_from_counts(*map(int,ea))['iou'],
                     'ours_instance_f1': object_metrics(*map(int,ib))['instance_f1'],
                     'external_instance_f1': object_metrics(*map(int,eb))['instance_f1']})
    internal_pixel = metrics_from_counts(*map(int,internal_px))
    external_pixel = metrics_from_counts(*map(int,external_px))
    internal_objects = object_metrics(*map(int,internal_ob))
    external_objects = object_metrics(*map(int,external_ob))
    comparison = {'split':'val','internal_run':run_dir.name, 'selection':selection,
                  'external_dir': str(Path(pred_dir)), 'external_kind':kind,
                  'external_threshold':external_threshold, 'config':config,
                  'tiles':len(ids),
                  'ours_pixel':internal_pixel,'external_pixel':external_pixel,
                  'ours_instance':internal_objects,'external_instance':external_objects,
                  'delta_external_minus_ours_pixel_iou':external_pixel['iou']-internal_pixel['iou'],
                  'delta_external_minus_ours_instance_f1':external_objects['instance_f1']-internal_objects['instance_f1'],
                  'note':'External provenance unverified; building-only validation comparison, not a SpaceNet8 leaderboard score.'}
    dest = run_dir / 'diagnostics_v4' / 'external_comparison'
    write_json(dest / 'validation_comparison.json',comparison)
    csv_write(dest / 'validation_paired_tiles.csv',rows,list(rows[0]))
    print('VALIDATION, same tiles and masks:')
    print(f'  Pixel IoU : ours={internal_pixel["iou"]:.4f}, external={external_pixel["iou"]:.4f}')
    print(f'  Instance F1: ours={internal_objects["instance_f1"]:.4f}, external={external_objects["instance_f1"]:.4f}')
    print('Saved:',dest/'validation_comparison.json')
    return comparison
