"""SpaceNet8 v3: validation-only model diagnostics and matched baseline comparison.

Nothing here accesses the held-out test split, mutates saved checkpoints,
changes the frozen data split, or rewrites a tuned threshold.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from .common import OUT, SPLITS, DATASET, check_prepared, csv_write, load_tile, read_json, write_json, sha256_file
from .data import read_annotations
from .engine import load_checkpoint, metrics_from_counts, predict_full, preprocess
from .compare import load_prediction


def _plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _device(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def _valid_run(run_dir):
    check_prepared()
    run_dir = Path(run_dir)
    cfg_path, ckpt_path, threshold_path = (run_dir/'config.json', run_dir/'best.pt', run_dir/'threshold.json')
    for path in (cfg_path, ckpt_path, threshold_path):
        if not path.is_file():
            raise FileNotFoundError(f'Run is incomplete; missing {path}')
    cfg = read_json(cfg_path)
    if cfg['split_sha256'] != hashlib.sha256(SPLITS.read_bytes()).hexdigest():
        raise RuntimeError(f'{run_dir}: saved split hash differs from frozen split')
    chosen = float(read_json(threshold_path)['threshold'])
    if not 0 < chosen < 1:
        raise ValueError('Saved validation threshold must be between 0 and 1')
    return run_dir, cfg, chosen, list(read_json(SPLITS)['val'])


def iter_val_probs(run_dir, batch_size=2, device_name='auto', refresh=False):
    """Emit (id, image, bool GT, float32 probabilities), cache keyed to checkpoint hash.

    Existing compatible caches are reused, including after interrupted inference.
    A changed checkpoint/split cannot silently reuse stale predictions.
    """
    run_dir, cfg, _, ids = _valid_run(run_dir)
    cache = run_dir/'diagnostics_v3'/'validation_probs'
    meta_file = cache/'metadata.json'
    expected = {'checkpoint_sha256': sha256_file(run_dir/'best.pt'),
                'split_sha256': cfg['split_sha256'],
                'preprocessor': cfg['preprocessor'], 'stride': cfg['stride'], 'ids': ids,
                'format': 'float32_probability_v1'}
    if refresh and cache.exists():
        import shutil
        shutil.rmtree(cache)
    if meta_file.is_file():
        if read_json(meta_file) != expected:
            raise RuntimeError(f'Stale validation prediction cache at {cache}. Remove it or pass --refresh-cache.')
    elif cache.exists() and any(cache.iterdir()):
        raise RuntimeError(f'Unidentified cache files at {cache}. Remove them or pass --refresh-cache.')
    cache.mkdir(parents=True, exist_ok=True)
    if not meta_file.exists():
        write_json(meta_file, expected)
    model = None
    device = None
    for i, image_id in enumerate(ids, 1):
        image, gt = load_tile(image_id)
        cache_file = cache/f'{image_id}.npy'
        if cache_file.exists():
            p = np.load(cache_file, allow_pickle=False)
        else:
            if model is None:
                device = _device(device_name)
                model, ckpt = load_checkpoint(run_dir/'best.pt', device)
                if ckpt['config']['split_sha256'] != cfg['split_sha256']:
                    raise RuntimeError('Checkpoint configuration differs from saved run configuration')
            p = predict_full(model, preprocess(image, cfg['preprocessor']),device,
                             batch_size=batch_size,stride=cfg['stride']).astype(np.float32)
            np.save(cache_file,p,allow_pickle=False)
            print(f'  Cached validation probability {i}/{len(ids)}: {image_id}',flush=True)
        if p.shape != gt.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError(f'Invalid probability cache: {cache_file}')
        yield image_id,image,gt.astype(bool),p.astype(np.float32,copy=False)


def threshold_table(prob_gt, thresholds):
    """Micro-averaged pixel metrics, counting each original tile pixel once."""
    totals = np.zeros((len(thresholds),3),dtype=np.int64)
    for probs,gt in prob_gt:
        gt = np.asarray(gt,dtype=bool)
        probs = np.asarray(probs,dtype=np.float32)
        if gt.shape != probs.shape:
            raise ValueError('Prediction and GT dimensions differ')
        for i, threshold in enumerate(thresholds):
            pred = probs >= threshold
            totals[i] += (np.count_nonzero(pred & gt),
                          np.count_nonzero(pred & ~gt),
                          np.count_nonzero(~pred & gt))
    return [{'threshold':round(float(t),4),**metrics_from_counts(*map(int,c))}
            for t,c in zip(thresholds,totals)]


def threshold_curves(run_dir,batch_size=2,device_name='auto',refresh=False):
    run_dir,_,tuned,_ = _valid_run(run_dir)
    thresholds = sorted(set([round(float(t),2) for t in np.arange(.05,.951,.05)]+[tuned]))
    # Cache is primarily on disk to allow repeated threshold sweeps without inference.
    probs_gt = [(prob,gt) for _,_,gt,prob in iter_val_probs(run_dir,batch_size,device_name,refresh)]
    rows = threshold_table(probs_gt,thresholds)
    dest = run_dir/'diagnostics_v3'
    csv_write(dest/'validation_threshold_curve.csv',rows,list(rows[0]))
    plt = _plot()
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    x = [r['threshold'] for r in rows]
    for col in ('iou','dice'):
        axes[0].plot(x,[r[col] for r in rows],label=col.upper(),marker='o',markersize=3)
    for col in ('precision','recall'):
        axes[1].plot(x,[r[col] for r in rows],label=col.capitalize(),marker='o',markersize=3)
    for ax in axes:
        ax.axvline(tuned,color='gray',linestyle='--',label=f'Saved threshold {tuned:.2f}')
        ax.set_xlabel('Probability threshold (validation only)')
        ax.set_ylabel('Global pixel score')
        ax.set_ylim(0,1)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_title('Overlap versus threshold')
    axes[1].set_title('Precision versus recall')
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(dest/'validation_threshold_curves.png',dpi=140)
    plt.close(fig)
    best=max(rows,key=lambda row:row['iou'])
    print(f'{run_dir.name}: saved={tuned:.2f}, scanned grid best={best["threshold"]:.2f} '
          f'(IoU={best["iou"]:.4f}); saved threshold NOT changed')
    print('Saved:',dest/'validation_threshold_curves.png')
    return rows


def predicted_polygons(mask,min_area=16):
    if min_area < 0:
        raise ValueError('min_area must be nonnegative')
    return [shape(geom) for geom,value in shapes(np.asarray(mask,dtype=np.uint8),
             mask=np.asarray(mask,dtype=bool))
            if value == 1 and shape(geom).area >= min_area]


def match_objects(pred_polys,gt_polys,match_iou=.5):
    """Greedy one-to-one IoU matching; returns GT index -> (pred index, IoU)."""
    if not 0 < match_iou <= 1:
        raise ValueError('match_iou must be in (0,1]')
    if not pred_polys or not gt_polys:
        return {}
    tree=STRtree(gt_polys)
    candidates=[]
    for i,p in enumerate(pred_polys):
        for j in tree.query(p):
            j=int(j)  # shapely>=2 STRtree returns integer indices
            g=gt_polys[j]
            inter=p.intersection(g).area
            if inter <= 0:
                continue
            score=inter/(p.area+g.area-inter)
            if score >= match_iou:
                candidates.append((float(score),i,j))
    used_pred,used_gt=set(),set()
    matched={}
    for score,i,j in sorted(candidates,key=lambda v:(-v[0],v[1],v[2])):
        if i not in used_pred and j not in used_gt:
            used_pred.add(i)
            used_gt.add(j)
            matched[j]=(i,score)
    return matched


def size_bin(area,small_max=100,medium_max=1000):
    if area < small_max:
        return 'small'
    if area < medium_max:
        return 'medium'
    return 'large'


def building_sizes(run_dir,small_max=100,medium_max=1000,min_area=16,
                   match_iou=.5,batch_size=2,device_name='auto',refresh=False):
    if not 0 < small_max < medium_max:
        raise ValueError('Expected 0 < small_max < medium_max')
    run_dir,_,tuned,_ = _valid_run(run_dir)
    annotation_map,annotation_stats=read_annotations()
    totals={key:{'objects':0,'matched':0,'unmatched':0} for key in ('small','medium','large')}
    predicted_total=0
    matched_total=0
    per_gt=[]
    for image_id,_,gt,prob in iter_val_probs(run_dir,batch_size,device_name,refresh):
        h,w=gt.shape
        frame=box(0,0,w,h)
        gt_polys=[]
        for polygon in annotation_map.get(image_id,[]):
            clipped=polygon.intersection(frame)
            if not clipped.is_empty and clipped.area > 0:
                # Valid polygons may be split by the tile boundary.
                if clipped.geom_type == 'Polygon':
                    gt_polys.append(clipped)
                elif hasattr(clipped,'geoms'):
                    gt_polys.extend(g for g in clipped.geoms if g.geom_type=='Polygon' and g.area>0)
        pred_polys=predicted_polygons(prob>=tuned,min_area=min_area)
        matched=match_objects(pred_polys,gt_polys,match_iou=match_iou)
        predicted_total+=len(pred_polys)
        matched_total+=len(matched)
        for j,g in enumerate(gt_polys):
            label=size_bin(g.area,small_max,medium_max)
            totals[label]['objects']+=1
            found=j in matched
            totals[label]['matched']+=int(found)
            totals[label]['unmatched']+=int(not found)
            per_gt.append({'tile_id':image_id,'object_index':j,'area_px2':round(g.area,3),
                           'size_bin':label,'matched':int(found),
                           'match_iou':round(matched[j][1],5) if found else 0.})
    rows=[]
    for label in ('small','medium','large'):
        x=totals[label]
        rows.append({'size_bin':label,'objects':x['objects'],'matched':x['matched'],
                     'missed':x['unmatched'],
                     'recall':x['matched']/x['objects'] if x['objects'] else None})
    gt_total=sum(x['objects'] for x in totals.values())
    fp=predicted_total-matched_total
    fn=gt_total-matched_total
    overall={'object_precision':matched_total/predicted_total if predicted_total else (1. if gt_total==0 else 0.),
             'object_recall':matched_total/gt_total if gt_total else 1.,
             'object_f1':2*matched_total/(2*matched_total+fp+fn) if 2*matched_total+fp+fn else 1.,
             'predicted_objects':predicted_total,'gt_objects':gt_total,
             'tp':matched_total,'fp':fp,'fn':fn,'threshold':tuned,
             'min_pred_area_px2':min_area,'match_iou':match_iou,
             'area_bins_px2':{'small_less_than':small_max,'medium_less_than':medium_max}}
    dest=run_dir/'diagnostics_v3'
    csv_write(dest/'building_size_recall.csv',rows,list(rows[0]))
    if per_gt:
        csv_write(dest/'building_objects_validation.csv',per_gt,list(per_gt[0]))
    write_json(dest/'building_instance_summary.json',overall)
    plt=_plot()
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([r['size_bin'] for r in rows],[r['recall'] or 0 for r in rows])
    for i,r in enumerate(rows):
        ax.text(i,(r['recall'] or 0)+.02,f'n={r["objects"]}',ha='center')
    ax.set_ylim(0,1.1)
    ax.set_ylabel('Matched GT object recall (IoU >= %.2f)'%match_iou)
    ax.set_title(f'{run_dir.name}: buildings by footprint area (validation)')
    fig.tight_layout()
    fig.savefig(dest/'building_size_recall.png',dpi=140)
    plt.close(fig)
    print(f'{run_dir.name}: instance F1={overall["object_f1"]:.4f}, '
          f'precision={overall["object_precision"]:.4f}, recall={overall["object_recall"]:.4f}')
    for row in rows:
        print(f'  {row["size_bin"]:6s}: {row["matched"]}/{row["objects"]} matched, recall={row["recall"]}')
    print('Saved:',dest/'building_size_recall.png')
    return rows,overall


def annotation_audit(run_dir=None, batch_size=2,device_name='auto',refresh=False,
                     boundary_px=3,count=6):
    """Validate prepared GT against WKT and optionally rank distant false positives.

    A matching WKT mask proves pipeline consistency, NOT annotation completeness.
    Review original imagery to decide whether red roof-like regions are missing GT.
    """
    check_prepared()
    if boundary_px < 1 or count < 1 or count > 30:
        raise ValueError('--boundary-px must be >=1 and --count 1..30')
    annotations,_=read_annotations()
    valid_ids=read_json(SPLITS)['val']
    probs_by_id={}
    threshold=None
    if run_dir is not None:
        run_dir,_,threshold,_=_valid_run(run_dir)
        iterator=iter_val_probs(run_dir,batch_size,device_name,refresh)
        probs_by_id={image_id:p for image_id,_,_,p in iterator}
    records=[]
    candidates=[]
    for image_id in valid_ids:
        image,gt=load_tile(image_id)
        binary_gt=gt.astype(bool)
        h,w=gt.shape
        wkt_mask=rasterize([(p,1) for p in annotations.get(image_id,[])],
                          out_shape=(h,w),fill=0,dtype=np.uint8)
        difference=int(np.count_nonzero(wkt_mask!=gt))
        record={'tile_id':image_id,'annotation_polygons':len(annotations.get(image_id,[])),
                'building_pixels':int(binary_gt.sum()),'mask_vs_reference_different_pixels':difference}
        if image_id in probs_by_id:
            pred=probs_by_id[image_id]>=threshold
            fp=pred & ~binary_gt
            fn=~pred & binary_gt
            kernel=np.ones((boundary_px*2+1,)*2,dtype=np.uint8)
            dilated=cv2.dilate(gt.astype(np.uint8),kernel,iterations=1).astype(bool)
            eroded=cv2.erode(gt.astype(np.uint8),kernel,iterations=1).astype(bool)
            fp_near=int(np.count_nonzero(fp & dilated))
            fp_far=int(np.count_nonzero(fp & ~dilated))
            fn_edge=int(np.count_nonzero(fn & ~eroded))
            fn_inner=int(np.count_nonzero(fn & eroded))
            record.update(fp_near_reference=fp_near,fp_far_from_reference=fp_far,
                          fn_near_reference_edge=fn_edge,fn_interior=fn_inner)
            candidates.append((fp_far,image_id,image.copy(),binary_gt.copy(),pred.copy()))
        records.append(record)
    dest=(run_dir/'diagnostics_v3') if run_dir is not None else (OUT/'diagnostics_v3')
    csv_write(dest/'validation_annotation_audit.csv',records,list(records[0]))
    if run_dir is not None:
        candidates.sort(key=lambda v:(-v[0],v[1]))
        top=candidates[:count]
        plt=_plot()
        fig,axes=plt.subplots(len(top),2,figsize=(10,4*len(top)),squeeze=False)
        for row,(_,image_id,image,gt,pred) in enumerate(top):
            fp=pred & ~gt
            fn=~pred & gt
            overlay=image.astype(np.float32)
            overlay[gt]=.4*overlay[gt]+.6*np.array([0,255,0])
            errors=image.astype(np.float32)
            errors[fp]=.4*errors[fp]+.6*np.array([255,0,0])
            errors[fn]=.4*errors[fn]+.6*np.array([0,70,255])
            axes[row,0].imshow(np.uint8(np.clip(overlay,0,255)))
            axes[row,0].set_title(f'{image_id} | GT polygons (green)')
            axes[row,1].imshow(np.uint8(np.clip(errors,0,255)))
            axes[row,1].set_title('FP red / FN blue: inspect possible missing labels')
            for ax in axes[row]:ax.axis('off')
        fig.tight_layout()
        fig.savefig(dest/'annotation_review_candidates.png',dpi=120)
        plt.close(fig)
        print('Saved:',dest/'annotation_review_candidates.png')
    mismatch=sum(r['mask_vs_reference_different_pixels'] for r in records)
    print(f'Validation mask vs original WKT differences: {mismatch} pixels across {len(records)} tiles')
    if mismatch:
        print('WARNING: prepared masks differ from reference rasterization. Inspect data pipeline.')
    else:
        print('WKT consistency verified; this cannot establish whether labels are complete.')
    return records


def compare_official_validation(run_dir,pred_dir,kind='binary',external_threshold=.5,
                                channel=0,batch_size=2,device_name='auto',refresh=False):
    """Compare genuine external baseline masks against our run, on identical validation GT.

    Never claims an external mask came from the official model without provenance.
    Official pipeline inference is outside this package.
    """
    run_dir,_,ours_threshold,ids=_valid_run(run_dir)
    external_threshold=float(external_threshold)
    if not 0 < external_threshold < 1:
        raise ValueError('External threshold must be in (0,1)')
    ours_total=np.zeros(3,dtype=np.int64)
    external_total=np.zeros(3,dtype=np.int64)
    rows=[]
    for image_id,_,gt,prob in iter_val_probs(run_dir,batch_size,device_name,refresh):
        ours=prob>=ours_threshold
        external=load_prediction(pred_dir,image_id,kind,external_threshold,channel)
        if external.shape!=gt.shape:
            raise ValueError(f'{image_id}: external prediction {external.shape} != GT {gt.shape}')
        def counts(p):
            return np.array([np.count_nonzero(p & gt),np.count_nonzero(p & ~gt),np.count_nonzero(~p & gt)],dtype=np.int64)
        a,b=counts(ours),counts(external)
        ours_total+=a
        external_total+=b
        ours_iou=metrics_from_counts(*map(int,a))['iou']
        external_iou=metrics_from_counts(*map(int,b))['iou']
        rows.append({'tile_id':image_id,'ours_tile_iou':ours_iou,
                     'external_tile_iou':external_iou,'external_minus_ours_iou':external_iou-ours_iou})
    ours_metric=metrics_from_counts(*map(int,ours_total))
    external_metric=metrics_from_counts(*map(int,external_total))
    report={'split':'val', 'ours_run':run_dir.name,'external_pred_dir':str(Path(pred_dir)),
            'external_kind':kind,'external_threshold':external_threshold,
            'ours_threshold':ours_threshold,'tiles':len(ids),
            'ours_global_pixel':ours_metric,'external_global_pixel':external_metric,
            'global_delta_iou':external_metric['iou']-ours_metric['iou'],
            'note':'External provenance must be independently verified. Building-only validation pixel comparison, not SpaceNet8 leaderboard score.'}
    dest=run_dir/'diagnostics_v3'/'official_comparison'
    write_json(dest/'validation_comparison.json',report)
    csv_write(dest/'validation_paired_tiles.csv',rows,list(rows[0]))
    plt=_plot()
    fig,ax=plt.subplots(figsize=(8,4))
    deltas=sorted(r['external_minus_ours_iou'] for r in rows)
    ax.bar(range(len(deltas)),deltas)
    ax.axhline(0,color='black',linewidth=1)
    ax.set_xlabel('Validation tiles (sorted by external - ours IoU)')
    ax.set_ylabel('External minus internal tile IoU')
    ax.set_title('Matched tile-level baseline comparison')
    fig.tight_layout()
    fig.savefig(dest/'validation_paired_delta.png',dpi=140)
    plt.close(fig)
    print(f'Validation (global pixel IoU): ours={ours_metric["iou"]:.4f}, '
          f'external={external_metric["iou"]:.4f}, external - ours={report["global_delta_iou"]:+.4f}')
    print('Saved:',dest/'validation_comparison.json')
    return report


def geometry_audit(samples=40,count=4,seed=111):
    """Check synchronized transforms and save actual training-patch mask previews."""
    import random
    from .engine import apply_augment, SIZE
    from .common import patch_starts
    if not 1 <= samples <= 10000 or not 1 <= count <= 12:
        raise ValueError('samples 1..10000 and count 1..12 required')
    check_prepared()
    original_rng=random.getstate()
    failures=[]
    stats={}
    reference=np.zeros((64,64),dtype=np.uint8)
    reference[2:17,9:23]=1
    reference[27:53,42:56]=1
    marked=np.zeros((64,64,3),dtype=np.uint8)
    marked[:,:,0]=reference*255
    try:
        for mode in ('geometry','flip_only','rotation_only'):
            movements=0
            for i in range(samples):
                random.seed(seed+i)
                im,gt=apply_augment(marked,reference,mode)
                if not np.array_equal(im[:,:,0]>0,gt.astype(bool)):
                    failures.append((mode,i,'image and mask not aligned'))
                if int(gt.sum())!=int(reference.sum()):
                    failures.append((mode,i,'mask area changed'))
                if not np.array_equal(gt,reference): movements+=1
            stats[mode]={'trials':samples,'changed_orientation':movements,'failures':sum(x[0]==mode for x in failures)}
    finally:
        random.setstate(original_rng)
    if failures:
        raise AssertionError(f'Augmentation alignment failed: {failures[:4]}')
    images=[]
    for image_id in read_json(SPLITS)['train']:
        image,gt=load_tile(image_id)
        h,w=gt.shape
        for y in patch_starts(h,SIZE,SIZE):
            for x in patch_starts(w,SIZE,SIZE):
                tile=image[y:y+SIZE,x:x+SIZE]
                mask=gt[y:y+SIZE,x:x+SIZE]
                if mask.any() and tile.shape[:2]==(SIZE,SIZE):
                    images.append((image_id,tile,mask))
                    if len(images)>=count: break
            if len(images)>=count: break
        if len(images)>=count:break
    if not images:
        raise RuntimeError('No full-size training patches with buildings to preview')
    plt=_plot()
    fig,axes=plt.subplots(len(images),4,figsize=(14,3.7*len(images)),squeeze=False)
    for row,(image_id,im,gt) in enumerate(images):
        for col,mode in enumerate(('none','geometry','flip_only','rotation_only')):
            original_state=random.getstate()
            random.seed(seed+1000*row)
            try:
                aug_im,aug_gt=apply_augment(im,gt,mode)
            finally:
                random.setstate(original_state)
            overlay=aug_im.astype(np.float32).copy()
            edge=cv2.morphologyEx(aug_gt.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
            overlay[edge]=[255,0,0]
            axes[row,col].imshow(np.uint8(np.clip(overlay,0,255)))
            axes[row,col].set_title(f'{mode}: {image_id[-10:]}')
            axes[row,col].axis('off')
    fig.tight_layout()
    dest=OUT/'diagnostics_v3'
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'geometry_alignment_audit.png',dpi=110)
    plt.close(fig)
    write_json(dest/'geometry_alignment_audit.json',stats)
    print('Synthetic image/mask alignment checks passed:',stats)
    print('Saved:',dest/'geometry_alignment_audit.png')
    return stats


def split_neighbor_audit():
    """Identify direct tile neighbors assigned to different splits (no split changes)."""
    import re
    check_prepared()
    splits=read_json(SPLITS)
    locations={}
    for split in ('train','val','test'):
        for image_id in splits[split]:
            match=re.fullmatch(r'(.+)_(-?\d+)_(-?\d+)_(-?\d+)',image_id)
            if match is None:
                raise ValueError(f'Cannot parse tile coordinates: {image_id}')
            scene,index,row,col=match.groups()
            locations[(scene,index,int(row),int(col))]=(split,image_id)
    edges=[]
    for (scene,index,r,c),(split,image_id) in locations.items():
        for dr,dc in ((1,0),(0,1)):
            other=locations.get((scene,index,r+dr,c+dc))
            if other is not None and other[0]!=split:
                edges.append({'tile_a':image_id,'split_a':split,
                              'tile_b':other[1],'split_b':other[0],
                              'relationship':'edge-adjacent'})
    dest=OUT/'diagnostics_v3'/'split_neighbor_edges.csv'
    csv_write(dest,edges,['tile_a','split_a','tile_b','split_b','relationship'])
    print(f'Cross-split edge-neighbor tile pairs: {len(edges)}')
    if edges:
        print('WARNING: grouped spatial blocks prevent SAME-BLOCK leakage, '
              'but do not impose a geographical buffer between splits.')
        print('Do not change an established split mid-study; use a new independent '
              'spatially buffered holdout for future final evaluations.')
    print('Saved:',dest)
    return edges
