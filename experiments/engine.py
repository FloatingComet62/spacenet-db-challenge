from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .common import (OUT, DATASET, SPLITS, read_json, write_json, load_tile,
                     patch_starts, normalize, check_prepared, csv_write, seed_all)

SIZE = 512


def preprocess(image, name):
    if name == 'raw':
        return image
    if name == 'clahe':
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    if name == 'gamma':
        lut = np.array([((i / 255) ** 0.8) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, lut)
    if name == 'sharpen':
        blur = cv2.GaussianBlur(image, (0,0), sigmaX=1.0)
        return cv2.addWeighted(image, 2.0, blur, -1.0, 0)
    if name == 'bilateral':
        return cv2.bilateralFilter(image, 5, 50, 50)
    if name == 'chainA':
        return preprocess(preprocess(image, 'clahe'), 'sharpen')
    if name == 'chainB':
        return preprocess(preprocess(image, 'bilateral'), 'clahe')
    if name == 'chainC':
        return preprocess(preprocess(preprocess(image, 'bilateral'), 'clahe'), 'sharpen')
    raise ValueError(f'Unknown preprocessor: {name}')


AUGMENT_MODES = ('none', 'geometry', 'photometric', 'combined', 'flip_only', 'rotation_only')


def apply_augment(image: np.ndarray, mask: np.ndarray, mode='none'):
    """Apply synchronized geometry and/or image-only color jitter to uint8 RGB and mask.

    The combined policy intentionally matches the original suite's augmentation
    policy, so previous seed111 augmentation runs remain comparable.
    """
    if mode is True:
        mode = 'combined'  # Compatibility with original training configuration
    elif mode is False:
        mode = 'none'
    if mode not in AUGMENT_MODES:
        raise ValueError(f'Unknown augmentation mode: {mode}')
    if mode == 'rotation_only':
        k = random.randrange(4)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    if mode == 'flip_only':
        if random.random() < .5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if random.random() < .5:
            image, mask = np.flipud(image), np.flipud(mask)
    if mode in ('geometry', 'combined'):
        k = random.randrange(4)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
        if random.random() < .5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if random.random() < .5:
            image, mask = np.flipud(image), np.flipud(mask)
    if mode in ('photometric', 'combined') and random.random() < .6:
        image = np.clip(image.astype(np.float32) * random.uniform(.85, 1.15)
                        + random.uniform(-15, 15), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class PatchDataset(Dataset):
    """Lazy file-backed patches. Cache up to two decoded tiles per worker."""
    def __init__(self, ids, preprocessor='raw', augment=False, stride=512):
        self.samples = []
        self.preprocessor = preprocessor
        self.augment = augment
        self.cache = OrderedDict()
        self.positive = []
        for image_id in ids:
            with np.load(DATASET/f'{image_id}.npz') as f:
                mask = f['mask']
            for y in patch_starts(mask.shape[0], SIZE, stride):
                for x in patch_starts(mask.shape[1], SIZE, stride):
                    self.samples.append((image_id, y, x))
                    self.positive.append(bool(mask[y:y+SIZE,x:x+SIZE].any()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id, y, x = self.samples[index]
        if image_id not in self.cache:
            image, mask = load_tile(image_id)
            self.cache[image_id] = (preprocess(image, self.preprocessor), mask)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(image_id)
        image, mask = self.cache[image_id]
        im = image[y:y+SIZE, x:x+SIZE].copy()
        gt = mask[y:y+SIZE, x:x+SIZE].copy()
        if im.shape[:2] != (SIZE, SIZE):
            h,w = im.shape[:2]
            im = cv2.copyMakeBorder(im, 0,SIZE-h,0,SIZE-w,cv2.BORDER_CONSTANT,value=(0,0,0))
            gt = np.pad(gt, ((0,SIZE-h),(0,SIZE-w)))
        im, gt = apply_augment(im, gt, self.augment)
        return torch.from_numpy(normalize(im)), torch.from_numpy(np.ascontiguousarray(gt[None], dtype=np.float32))


def balanced_sampler(dataset):
    pos = np.asarray(dataset.positive, dtype=bool)
    npos = int(pos.sum())
    nneg = len(pos) - npos
    if not npos:
        raise ValueError('Training contains no building pixels')
    if not nneg:
        weights = np.ones(len(pos), dtype=np.float64)
    else:
        weights = np.where(pos, 0.5/npos, 0.5/nneg)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(pos), replacement=True)


def load_architecture():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise RuntimeError('Missing segmentation-models-pytorch. See experiment_requirements.txt') from e
    return smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3,
                    classes=1, activation=None)


def new_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3,
                    classes=1, activation=None)


def bce_dice(logits, target, name):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    if name == 'bce':
        return bce
    prob = logits.float().sigmoid()
    target = target.float()
    intersection = (prob * target).sum(dim=(1,2,3))
    denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = 1 - ((2*intersection+1e-6)/(denom+1e-6)).mean()
    return bce + dice


def metrics_from_counts(tp, fp, fn):
    denom = tp+fp+fn
    return {'iou': tp/denom if denom else 1.,
            'dice': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.,
            'precision': tp/(tp+fp) if tp+fp else (1. if fn == 0 else 0.),
            'recall': tp/(tp+fn) if tp+fn else 1.,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def predict_full(model, image, device, batch_size=2, stride=512, amp=True):
    """Accumulate logits on whole tile so every image pixel is scored once."""
    h,w = image.shape[:2]
    coords = [(y,x) for y in patch_starts(h,SIZE,stride) for x in patch_starts(w,SIZE,stride)]
    accum = np.zeros((h,w), np.float32)
    counts = np.zeros((h,w), np.float32)
    for start in range(0, len(coords), batch_size):
        subset = coords[start:start+batch_size]
        tensors = []
        for y,x in subset:
            patch = image[y:y+SIZE,x:x+SIZE]
            if patch.shape[:2] != (SIZE,SIZE):
                ph,pw = patch.shape[:2]
                patch = cv2.copyMakeBorder(patch,0,SIZE-ph,0,SIZE-pw,
                                           cv2.BORDER_CONSTANT,value=(0,0,0))
            tensors.append(normalize(patch))
        batch = torch.from_numpy(np.stack(tensors)).to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and device.type=='cuda'):
                pred = model(batch)
        logits = pred.float().cpu().numpy()[:,0]
        for (y,x), logit in zip(subset,logits):
            ph,pw = min(h-y,SIZE), min(w-x,SIZE)
            accum[y:y+ph,x:x+pw] += logit[:ph,:pw]
            counts[y:y+ph,x:x+pw] += 1
    logits = accum / np.maximum(counts,1)
    return 1.0/(1.0+np.exp(-np.clip(logits,-80,80)))


def evaluate_model(model, ids, preprocessor, device, threshold=0.5, batch_size=2,
                   stride=512, save_dir=None, thresholds=None):
    model.eval()
    total = np.zeros((len(thresholds),3), np.int64) if thresholds is not None else np.zeros(3, np.int64)
    rows = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for i,image_id in enumerate(ids,1):
        image, gt = load_tile(image_id)
        probs = predict_full(model,preprocess(image,preprocessor),device,batch_size,stride)
        gt = gt.astype(bool)
        if save_dir is not None:
            np.save(save_dir/f'{image_id}.npy',probs.astype(np.float16))
        if thresholds is not None:
            for j,t in enumerate(thresholds):
                p = probs >= t
                total[j] += (int((p&gt).sum()),int((p&~gt).sum()),int((~p&gt).sum()))
        else:
            p = probs >= threshold
            tp,fp,fn = int((p&gt).sum()),int((p&~gt).sum()),int((~p&gt).sum())
            total += (tp,fp,fn)
            rows.append({'id':image_id,'building_pixels':int(gt.sum()),'predicted_pixels':int(p.sum()),
                         **metrics_from_counts(tp,fp,fn)})
    elapsed = time.monotonic() - start
    perf = {'tiles':len(ids), 'seconds_total':round(elapsed,3),
            'seconds_per_tile':round(elapsed/max(len(ids),1),3),
            'peak_vram_mib':round(torch.cuda.max_memory_allocated(device)/2**20,2) if device.type=='cuda' else None}
    if thresholds is not None:
        return [metrics_from_counts(*c) for c in total],perf
    if save_dir is not None:
        csv_write(save_dir.parent/'per_tile.csv',rows,list(rows[0]))
    return metrics_from_counts(*total),perf


PRESETS = {
 'quick': ['matched_bce','raw_bce_dice','clahe','augmentation'],
 'all': ['matched_bce','raw_bce_dice','clahe','gamma','bilateral','sharpen',
         'chainA','chainB','chainC','augmentation','augmentation_geometry',
         'augmentation_photometric','augmentation_flip', 'augmentation_rotation','lr_schedule']
}


def config_for(name):
    if name not in PRESETS['all']:
        raise ValueError(f'Unknown experiment: {name}')
    augmentation_mode = {
        'augmentation': 'combined',
        'augmentation_geometry': 'geometry',
        'augmentation_photometric': 'photometric',
        'augmentation_flip': 'flip_only',
        'augmentation_rotation': 'rotation_only',
    }.get(name, 'none')
    return {'name':name,
            'preprocessor': name if name in ('clahe','gamma','bilateral','sharpen',
                                            'chainA','chainB','chainC') else 'raw',
            'loss': 'bce' if name == 'matched_bce' else 'bce_dice',
            'augment': augmentation_mode != 'none',
            'augment_mode': augmentation_mode,
            'scheduler': name == 'lr_schedule'}


def train_one(name, seed=111, epochs=50, batch_size=2, lr=1e-4,
              workers=2, stride=512, overwrite=False, device_name='auto'):
    check_prepared()
    run_dir = OUT/'runs'/f'{name}_seed{seed}'
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f'{run_dir} exists; use --overwrite to deliberately retrain')
    if not 0 < lr <= 1 or epochs < 1 or batch_size < 1 or workers < 0 or not 1 <= stride <= SIZE:
        raise ValueError('Invalid training hyperparameter')
    if run_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    config = config_for(name)
    config.update({'seed':seed,'epochs':epochs,'batch_size':batch_size,'lr':lr,
                   'workers':workers,'stride':stride, 'split_sha256':__import__('hashlib').sha256(SPLITS.read_bytes()).hexdigest()})
    write_json(run_dir/'config.json',config)
    seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else torch.device(device_name)
    ids = read_json(SPLITS)
    print(f'=== {name} / seed {seed} / device {device} ===',flush=True)
    train = PatchDataset(ids['train'],config['preprocessor'],config['augment_mode'],stride=stride)
    sampler = balanced_sampler(train)
    loader = DataLoader(train,batch_size=batch_size,sampler=sampler,num_workers=workers,
                        pin_memory=(device.type=='cuda'),persistent_workers=(workers>0))
    model = new_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5) if config['scheduler'] else None
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best = -1
    history = []
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum = 0.
        seen = 0
        started = time.monotonic()
        for image,gt in loader:
            image = image.to(device,non_blocking=True)
            gt = gt.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits = model(image)
                loss = bce_dice(logits,gt,config['loss'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())*len(image)
            seen += len(image)
        if scheduler is not None:
            scheduler.step()
        val,perf = evaluate_model(model,ids['val'],config['preprocessor'],device,
                                  batch_size=batch_size,stride=stride)
        row = {'epoch':epoch, 'loss':round(loss_sum/max(seen,1),5),
               'val_iou':round(val['iou'],6),'val_dice':round(val['dice'],6),
               'val_precision':round(val['precision'],6),'val_recall':round(val['recall'],6),
               'seconds':round(time.monotonic()-started,2)}
        history.append(row)
        csv_write(run_dir/'history.csv',history,list(row))
        if val['iou'] > best:
            best = float(val['iou'])
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,
                        'config':config,'val_iou_0_5':best},run_dir/'best.pt')
        print(f"epoch {epoch:03d}/{epochs} loss={row['loss']:.4f} val_full_iou={val['iou']:.4f} best={best:.4f} ({row['seconds']:.1f}s)",flush=True)
    write_json(run_dir/'completed.json', {'completed_epochs':len(history),'best_val_iou':float(best)})
    print('Saved:',run_dir/'best.pt',flush=True)
    return run_dir


def load_checkpoint(path, device):
    ckpt = torch.load(path,map_location=device,weights_only=True)
    model = load_architecture().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model,ckpt


def optimize_threshold(run_dir, batch_size=2, device_name='auto'):
    check_prepared()
    run_dir = Path(run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else torch.device(device_name)
    model,ckpt = load_checkpoint(run_dir/'best.pt',device)
    cfg = ckpt['config']
    if cfg['split_sha256'] != __import__('hashlib').sha256(SPLITS.read_bytes()).hexdigest():
        raise RuntimeError('Split changed since training; comparison invalid.')
    thresholds = [round(float(x),2) for x in np.arange(.20,.801,.05)]
    scores,perf = evaluate_model(model,read_json(SPLITS)['val'],cfg['preprocessor'],device,
                                  batch_size=batch_size,stride=cfg['stride'],thresholds=thresholds)
    winners = sorted(zip(thresholds,scores),key=lambda x:(-x[1]['iou'],abs(x[0]-.5)))
    selected,metrics = winners[0]
    result = {'threshold':selected, 'validation':metrics,'runtime':perf,
              'grid':[{'threshold':t,**s} for t,s in zip(thresholds,scores)]}
    write_json(run_dir/'threshold.json',result)
    print(f'{run_dir.name}: best validation threshold {selected:.2f} (IoU {metrics["iou"]:.4f})')
    return result


def finalize(run_dir, split='test', batch_size=2, device_name='auto', save_predictions=True):
    check_prepared()
    run_dir = Path(run_dir)
    if split != 'test':
        raise ValueError('finalize evaluates held-out test only; use tune for validation')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else torch.device(device_name)
    model,ckpt = load_checkpoint(run_dir/'best.pt',device)
    cfg = ckpt['config']
    if cfg['split_sha256'] != __import__('hashlib').sha256(SPLITS.read_bytes()).hexdigest():
        raise RuntimeError('Split changed since training; comparison invalid.')
    tuned_path = run_dir/'threshold.json'
    if not tuned_path.is_file():
        optimize_threshold(run_dir,batch_size=batch_size,device_name=device_name)
    tuned = read_json(tuned_path)['threshold']
    dest = run_dir/'test_predictions' if save_predictions else None
    m,p = evaluate_model(model,read_json(SPLITS)['test'],cfg['preprocessor'],device,
                         threshold=tuned,batch_size=batch_size,stride=cfg['stride'],save_dir=dest)
    result = {'name':cfg['name'],'seed':cfg['seed'],'split':'test','threshold':tuned,
              'metric_type':'full_tile_global_pixel_binary',**m,**p}
    write_json(run_dir/'test_metrics.json',result)
    print(f'{run_dir.name}: TEST IoU {m["iou"]:.4f}, Dice {m["dice"]:.4f}, Precision {m["precision"]:.4f}, Recall {m["recall"]:.4f}')
    return result


def results_table():
    runs_dir = OUT/'runs'
    if not runs_dir.exists():
        print('No experiments yet')
        return
    records = []
    for path in sorted(runs_dir.glob('*/test_metrics.json')):
        m=read_json(path)
        records.append({'run':path.parent.name,'iou':m['iou'],'dice':m['dice'],
                        'precision':m['precision'],'recall':m['recall'],
                        'seconds_per_tile':m['seconds_per_tile'], 'peak_vram_mib':m['peak_vram_mib']})
    if records:
        csv_write(OUT/'results.csv',records,list(records[0]))
        print(f"{'run':28} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}")
        for r in sorted(records,key=lambda x:-x['iou']):
            print(f"{r['run']:28} {r['iou']:8.4f} {r['dice']:8.4f} {r['precision']:8.4f} {r['recall']:8.4f}")
        print('Wrote',OUT/'results.csv')
    else:
        print('No test metrics yet. Use python exp.py finalize --all after model selection.')
