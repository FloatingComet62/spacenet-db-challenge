"""Identical side-by-side test tile views for your model and an external baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import OUT, SPLITS, load_tile, read_json
from .compare import load_prediction
from .engine import metrics_from_counts


def pairwise_visuals(a_dir,b_dir,a_kind='probabilities',b_kind='binary',
                     a_threshold=.5,b_threshold=.5,split='test',count=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names=read_json(SPLITS)[split]
    tileinfo={r['id']:r['positive_pixels'] for r in read_json(OUT/'manifest.json')['tiles']}
    # Show examples with a range of label coverage, rather than only empty scenes.
    ranked=sorted(names,key=lambda x:tileinfo[x])
    if count < 1:
        raise ValueError('count must be >= 1')
    positions=np.linspace(0,len(ranked)-1,min(count,len(ranked)),dtype=int)
    selected=[ranked[p] for p in positions]
    output=OUT/'visual_comparisons'
    output.mkdir(parents=True,exist_ok=True)
    for image_id in selected:
        img,gt=load_tile(image_id)
        gt=gt.astype(bool)
        a=load_prediction(a_dir,image_id,a_kind,a_threshold)
        b=load_prediction(b_dir,image_id,b_kind,b_threshold)
        if a.shape!=gt.shape or b.shape!=gt.shape:
            raise ValueError(f'{image_id}: prediction dimensions must match the GT exactly')
        def iou(p):
            tp=int((p&gt).sum());fp=int((p&~gt).sum());fn=int((~p&gt).sum())
            return metrics_from_counts(tp,fp,fn)['iou']
        def overlay(p):
            rgb=img.astype(np.float32).copy()
            tp=p&gt; fp=p&~gt; fn=~p&gt
            rgb[tp]=rgb[tp]*.6+np.array([0,220,0])*.4
            rgb[fp]=rgb[fp]*.6+np.array([255,30,30])*.4
            rgb[fn]=rgb[fn]*.6+np.array([30,90,255])*.4
            return rgb.astype(np.uint8)
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        for ax,im,title in zip(axes,[img,gt,overlay(a),overlay(b)],
              ['Input RGB','Ground truth',f'Model A (IoU {iou(a):.3f})',f'Model B (IoU {iou(b):.3f})']):
            ax.imshow(im,cmap='gray' if im.ndim==2 else None)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'{image_id} | TP green, FP red, FN blue')
        fig.tight_layout()
        fig.savefig(output/f'{image_id}.png',dpi=140)
        plt.close(fig)
    print(f'Saved {len(selected)} comparisons in {output}')
