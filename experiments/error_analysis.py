"""Inspect most false-positive-prone validation scenes, without touching test data."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .common import SPLITS, check_prepared, csv_write, load_tile, read_json
from .engine import load_checkpoint, metrics_from_counts, predict_full, preprocess


def save_validation_errors(run_dir, count=6, batch_size=2, device_name='auto'):
    check_prepared()
    if count < 1 or count > 30:
        raise ValueError('--count must be between 1 and 30')
    run_dir=Path(run_dir)
    if not (run_dir/'threshold.json').is_file():
        raise RuntimeError(f'Tune the validation threshold first: python exp.py tune --run {run_dir}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else torch.device(device_name)
    model,ckpt=load_checkpoint(run_dir/'best.pt',device)
    cfg=ckpt['config']
    if cfg['split_sha256']!=hashlib.sha256(SPLITS.read_bytes()).hexdigest():
        raise RuntimeError('Saved checkpoint was trained on a different split')
    threshold=read_json(run_dir/'threshold.json')['threshold']
    rows=[]
    candidates=[]
    for image_id in read_json(SPLITS)['val']:
        image,gt=load_tile(image_id)
        probs=predict_full(model,preprocess(image,cfg['preprocessor']),device,
                           batch_size=batch_size,stride=cfg['stride'])
        gt=gt.astype(bool)
        pred=probs>=threshold
        tp=int((pred & gt).sum())
        fp=int((pred & ~gt).sum())
        fn=int((~pred & gt).sum())
        metrics=metrics_from_counts(tp,fp,fn)
        fpr_area=fp/gt.size
        false_discovery=fp/max(int(pred.sum()),1)
        row={'tile_id':image_id, 'fp_area_fraction':round(fpr_area,8),
             'false_discovery_fraction':round(false_discovery,8),
             'building_pixel_fraction':round(float(gt.mean()),8),**metrics}
        rows.append(row)
        # Keep only top K scenes in memory, ranked by FP fraction of total area.
        if len(candidates)<count or fpr_area>candidates[-1][0]:
            candidates.append((fpr_area,image_id,image,gt,pred,probs))
            candidates.sort(key=lambda c:(-c[0],c[1]))
            if len(candidates)>count:
                candidates.pop()
    rows.sort(key=lambda r:(-r['fp_area_fraction'],r['tile_id']))
    output_csv=run_dir/'validation_error_report.csv'
    if not rows:
        raise RuntimeError('No validation tiles in the split')
    csv_write(output_csv,rows,list(rows[0]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(candidates),3,figsize=(15,4*len(candidates)),squeeze=False)
    for i,(fraction,image_id,image,gt,pred,probs) in enumerate(candidates):
        tp=pred & gt
        fp=pred & ~gt
        fn=~pred & gt
        overlay=image.astype(np.float32).copy()
        for mask,color in ((tp,(0,255,0)),(fp,(255,0,0)),(fn,(0,80,255))):
            overlay[mask]=overlay[mask]*.4 + np.asarray(color)*.6
        axes[i,0].imshow(image)
        axes[i,0].set_title(f'{image_id}: original')
        axes[i,1].imshow(probs,cmap='magma',vmin=0,vmax=1)
        axes[i,1].set_title(f'Building probability | threshold {threshold:.2f}')
        axes[i,2].imshow(np.uint8(np.clip(overlay,0,255)))
        axes[i,2].set_title(f'TP green, FP red, FN blue | FP area {fraction:.2%}')
        for a in axes[i]:
            a.axis('off')
    fig.tight_layout()
    png=run_dir/'validation_errors.png'
    fig.savefig(png,dpi=110)
    plt.close(fig)
    print(f'Saved {output_csv} and {png} (VALIDATION tiles only)')
    for row in rows[:min(count,len(rows))]:
        print(f"  {row['tile_id']}  FP area={row['fp_area_fraction']:.2%}  "
              f"false discovery={row['false_discovery_fraction']:.2%}  IoU={row['iou']:.4f}")
    return rows
