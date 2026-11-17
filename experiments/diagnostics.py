"""Save an augmentation image/mask overlay grid using TRAINING samples only."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .common import OUT, check_prepared, load_tile, read_json, SPLITS, patch_starts
from .engine import SIZE, apply_augment


def save_augmentation_preview(count=4, seed=111):
    if count < 1 or count > 16:
        raise ValueError('--count must be in [1, 16]')
    check_prepared()
    # Agg works in remote/headless sessions, including EXWM/ssh.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    samples = []
    for image_id in read_json(SPLITS)['train']:
        _, mask = load_tile(image_id)
        for y in patch_starts(mask.shape[0], SIZE, SIZE):
            for x in patch_starts(mask.shape[1], SIZE, SIZE):
                if mask[y:y+SIZE,x:x+SIZE].any():
                    samples.append((image_id,y,x))
    if not samples:
        raise RuntimeError('No training patches contain building pixels')
    rng = random.Random(seed)
    choices = rng.sample(samples, min(count, len(samples)))
    policies = ('none','geometry','photometric','combined')
    # One column per augmented mode, one row per example.
    fig, axes = plt.subplots(len(choices),len(policies),figsize=(15,3.8*len(choices)),squeeze=False)
    axes=np.asarray(axes).reshape(count,len(policies))
    for row,(image_id,y,x) in enumerate(choices):
        image,mask = load_tile(image_id)
        im = image[y:y+SIZE,x:x+SIZE]
        gt = mask[y:y+SIZE,x:x+SIZE]
        # All columns share the same patch; each random policy starts with same seed.
        for col,policy in enumerate(policies):
            random.seed(seed * 1000 + row)
            aug_im,aug_gt = apply_augment(im.copy(),gt.copy(),policy)
            ax = axes[row,col]
            gt_u8 = aug_gt.astype(np.uint8)
            contours,_ = cv2.findContours(gt_u8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            overlay=np.zeros_like(aug_im)
            cv2.drawContours(overlay,contours,-1,(255,0,0),2)
            overlay_mask=overlay.any(axis=2)
            rendered=aug_im.copy()
            rendered[overlay_mask]=overlay[overlay_mask]
            ax.clear(); ax.imshow(rendered)
            ax.set_title(f'{policy} | #{row+1} | GT {int(aug_gt.sum()):,}')
            ax.axis('off')
    path=OUT/'augmentation_preview.png'
    fig.tight_layout()
    fig.savefig(path,dpi=120)
    plt.close(fig)
    print('Saved',path,'(training patches only; red outlines = masks)')
    return path
