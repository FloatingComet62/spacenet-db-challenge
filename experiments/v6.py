"""v6: controlled, resumable training-budget and sampling ablations.

All selection and metrics are VALIDATION-only. Nothing in this module reads the
held-out test split except to retain the split file's provenance hash.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from rasterio.features import rasterize

from .common import (DATASET, OUT, SPLITS, check_prepared, csv_write, load_tile,
                     read_json, seed_all, sha256_file, write_json)
from .data import read_annotations
from .engine import (PatchDataset, SIZE, balanced_sampler, bce_dice, evaluate_model,
                     new_model, optimize_threshold)

ROOT = OUT / 'diagnostics_v6'
VERSION = 'v6.1'
SEEDS = (111, 222, 333)
BASES = ('raw_bce_dice', 'augmentation_photometric')


@dataclass(frozen=True)
class Task:
    base: str
    epochs: int
    sampling: str = 'balanced'  # balanced or small_aware
    objective: str = 'bce_dice'  # bce_dice or boundary

    def validate(self):
        if self.base not in BASES or self.epochs < 1:
            raise ValueError(f'Invalid v6 model or epoch count: {self}')
        if self.sampling not in ('balanced', 'small_aware'):
            raise ValueError(f'Invalid sampling mode: {self.sampling}')
        if self.objective not in ('bce_dice', 'boundary'):
            raise ValueError(f'Invalid objective: {self.objective}')

    @property
    def name(self):
        self.validate()
        ending = f'{self.sampling}_e{self.epochs}'
        if self.objective != 'bce_dice':
            ending += f'_{self.objective}'
        return f'v6_{self.base}_{ending}'

    def run_dir(self, seed):
        return OUT / 'runs' / f'{self.name}_seed{seed}'


def planned_tasks(phase='duration'):
    durations = [Task(base, epochs) for base in BASES for epochs in (16, 60)]
    sampling = [Task('augmentation_photometric', 60, 'small_aware')]
    boundary = [Task('augmentation_photometric', 60, 'balanced', 'boundary')]
    if phase == 'duration':
        return durations
    if phase == 'sampling':
        return [Task('augmentation_photometric', 60), *sampling]
    if phase == 'boundary':
        return [Task('augmentation_photometric', 60), *boundary]
    if phase == 'full':
        return durations + sampling + boundary
    raise ValueError(f'Unknown v6 phase: {phase}')


def config_for_task(task, seed, batch_size, lr, stride, workers, val_every,
                    small_area, small_fraction, boundary_weight):
    task.validate()
    if not (0 <= seed <= 2**32-1 and 1 <= batch_size and 0 < lr <= 1
            and workers >= 0 and 1 <= stride <= SIZE and val_every >= 1
            and 0 < small_area <= 10000 and 0 < small_fraction < 0.5
            and 0 <= boundary_weight <= 10):
        raise ValueError('Invalid v6 hyperparameter')
    return {
        'suite_version': VERSION, 'name':task.name, 'base_name':task.base,
        'preprocessor':'raw',
        'augment_mode': 'photometric' if task.base == 'augmentation_photometric' else 'none',
        'loss':task.objective, 'scheduler': False, 'sampler':task.sampling,
        'small_area_max':float(small_area), 'small_positive_fraction':float(small_fraction),
        'boundary_weight':float(boundary_weight),
        'seed':int(seed), 'epochs':int(task.epochs), 'batch_size':int(batch_size),
        'lr':float(lr), 'stride':int(stride), 'workers':int(workers),
        'val_every':int(val_every),
        'split_sha256':sha256_file(SPLITS),
        'manifest_sha256':sha256_file(OUT/'manifest.json'),
    }


def small_patch_flags(dataset, area_max=100.):
    """Identify patches overlapping a TRAINING-only small GT footprint.

    Polygon area uses the same pixel-square units as the v3 size audit.
    Rasterizing the original per-instance polygons avoids confusing one small
    footprint with a small fragment of a large building at the patch edge.
    """
    if area_max <= 0:
        raise ValueError('area_max must be positive')
    annotations, _ = read_annotations()
    samples_by_id = defaultdict(list)
    for idx, (image_id, y, x) in enumerate(dataset.samples):
        samples_by_id[image_id].append((idx, int(y), int(x)))
    has_small = np.zeros(len(dataset), dtype=bool)
    for image_id, samples in samples_by_id.items():
        small_polys = [p for p in annotations.get(image_id, []) if 0 < p.area <= area_max]
        if not small_polys:
            continue
        _, mask = load_tile(image_id)
        small = rasterize([(p, 1) for p in small_polys], out_shape=mask.shape,
                          fill=0, dtype=np.uint8)
        # Sum via integral image; avoids materializing every 512x512 patch.
        integral = np.pad(small.astype(np.int64), ((1,0),(1,0))).cumsum(0).cumsum(1)
        height, width = mask.shape
        for idx, y, x in samples:
            y2, x2 = min(y + SIZE, height), min(x + SIZE, width)
            n = integral[y2,x2] - integral[y,x2] - integral[y2,x] + integral[y,x]
            has_small[idx] = n > 0
    return has_small


def sampling_weights(positive, small_flags, small_fraction=.35):
    """Keep the old 50/50 positive/empty quota, bias positive samples to small GT.

    For 3 nonempty bins: small=.35, other_positive=.15, empty=.50.
    If no empty/other-positive patches exist, redistribute that category's
    quota within its parent class; never sample an absent category.
    """
    positive = np.asarray(positive, bool)
    small_flags = np.asarray(small_flags, bool)
    if positive.shape != small_flags.shape or positive.ndim != 1 or not len(positive):
        raise ValueError('positive/small flags must be aligned 1D arrays')
    if not 0 < small_fraction < .5:
        raise ValueError('small_fraction must lie in (0,0.5)')
    if np.any(small_flags & ~positive):
        raise ValueError('A patch with a small building must be positive')
    small = small_flags
    other = positive & ~small
    empty = ~positive
    if not small.any():
        raise ValueError('No small building appears in training patches: cannot run the small-aware ablation')
    quotas = {'small':small_fraction, 'other_positive':.5-small_fraction, 'empty':.5}
    if not other.any():
        quotas['small'] += quotas['other_positive']
        quotas['other_positive'] = 0.
    if not empty.any():
        # Match balanced_sampler behavior for datasets with no empty patches.
        scale = 1. / (quotas['small'] + quotas['other_positive'])
        quotas['small'] *= scale
        quotas['other_positive'] *= scale
        quotas['empty'] = 0.
    weights = np.zeros(len(positive), dtype=np.float64)
    counts = {'small':int(small.sum()),'other_positive':int(other.sum()),'empty':int(empty.sum())}
    for k,indices in [('small',small),('other_positive',other),('empty',empty)]:
        if counts[k]:
            weights[indices] = quotas[k]/counts[k]
    if not np.isfinite(weights).all() or not np.any(weights):
        raise AssertionError('Invalid sampling weights')
    return weights, {'patch_counts':counts,'sampling_quotas':quotas,
                     'positive_fraction':float(positive.mean()),
                     'small_patch_fraction':float(small.mean())}


def make_sampler(dataset, cfg, run_dir):
    if cfg['sampler'] == 'balanced':
        sampler = balanced_sampler(dataset)
        pos = np.asarray(dataset.positive, bool)
        meta = {'mode':'balanced','patch_counts':{'positive':int(pos.sum()),
                                               'empty':int((~pos).sum())},
                'sampling_quotas':{'positive':.5 if (~pos).any() else 1.,
                                   'empty':.5 if (~pos).any() else 0.}}
    else:
        flags = small_patch_flags(dataset, cfg['small_area_max'])
        weights, meta = sampling_weights(dataset.positive,flags,cfg['small_positive_fraction'])
        meta['mode'] = 'small_aware'
        sampler = WeightedRandomSampler(torch.from_numpy(weights),len(dataset),replacement=True)
    meta['train_patches'] = len(dataset)
    path = run_dir/'sampler.json'
    if path.exists() and read_json(path) != meta:
        raise RuntimeError('Sampler metadata changed since run began; refusing inconsistent resumption')
    write_json(path,meta)
    return sampler


def boundary_weighted_loss(logits, targets, weight=.5):
    """BCE+Dice plus additional BCE near foreground/background transitions.

    This does NOT learn individual shared walls, since its GT is a binary mask.
    It is an optional isolated ablation, not the main training-duration test.
    """
    base = bce_dice(logits,targets,'bce_dice')
    if weight <= 0:
        return base
    target = targets.float()
    local_max = nn.functional.max_pool2d(target,3,stride=1,padding=1)
    local_min = -nn.functional.max_pool2d(-target,3,stride=1,padding=1)
    edge = (local_max - local_min).detach()
    extra = nn.functional.binary_cross_entropy_with_logits(logits.float(),target,
                                                             reduction='none')
    # Normalize by edge count, making the scale comparable across examples.
    return base + weight * (edge * extra).sum() / edge.sum().clamp_min(1)


def _state_save(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def _inspect_run(run_dir, expected):
    if not run_dir.exists():
        return 'TRAIN'
    cfg_path = run_dir/'config.json'
    if not cfg_path.is_file():
        raise RuntimeError(f'Existing {run_dir} has no config; archive it manually before retrying')
    actual = read_json(cfg_path)
    if actual != expected:
        changes = {k:[actual.get(k),v] for k,v in expected.items() if actual.get(k)!=v}
        raise RuntimeError(f'Cannot reuse {run_dir}, configuration mismatch: {changes}')
    if (run_dir/'completed.json').exists():
        if not (run_dir/'best.pt').exists():
            raise RuntimeError(f'Completed run lacks best.pt: {run_dir}')
        if read_json(run_dir/'completed.json').get('completed_epochs') != expected['epochs']:
            raise RuntimeError(f'Wrong completed epoch count: {run_dir}')
        return 'REUSE'
    if (run_dir/'latest.pt').exists():
        return 'RESUME'
    if (run_dir/'best.pt').exists():
        raise RuntimeError(f'Inconsistent partial checkpoint at {run_dir}; inspect manually')
    return 'TRAIN'  # config-only interruption, safe to start fresh


def _resolve_device(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name=='auto' else torch.device(name)


def train_task(task, seed=111, batch_size=2, lr=1e-4, workers=2,
               stride=512, val_every=2, small_area=100., small_fraction=.35,
               boundary_weight=.5, device='auto'):
    check_prepared()
    cfg = config_for_task(task,seed,batch_size,lr,stride,workers,val_every,
                          small_area,small_fraction,boundary_weight)
    run_dir = task.run_dir(seed)
    action = _inspect_run(run_dir,cfg)
    if action == 'REUSE':
        print('REUSE',run_dir,flush=True)
        if not (run_dir/'threshold.json').exists():
            optimize_threshold(run_dir,batch_size=batch_size,device_name=device)
        return run_dir
    run_dir.mkdir(parents=True,exist_ok=True)
    if not (run_dir/'config.json').exists():
        write_json(run_dir/'config.json',cfg)
    seed_all(seed)
    device = _resolve_device(device)
    splits = read_json(SPLITS)
    dataset = PatchDataset(splits['train'],preprocessor='raw',
                          augment=cfg['augment_mode'],stride=stride)
    sampler = make_sampler(dataset,cfg,run_dir)
    model = new_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start_epoch, best, best_epoch, history = 0, -1., 0, []
    if action == 'RESUME':
        resume = torch.load(run_dir/'latest.pt',map_location=device,weights_only=True)
        if resume['config'] != cfg:
            raise RuntimeError(f'Resume config mismatch: {run_dir}')
        model.load_state_dict(resume['model_state_dict'])
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scaler.load_state_dict(resume['scaler_state_dict'])
        start_epoch = int(resume['epoch'])
        best = float(resume['best_val_iou'])
        best_epoch = int(resume['best_epoch'])
        history = resume['history']
        if not (0 <= start_epoch <= task.epochs) or len(history) != start_epoch:
            raise RuntimeError('Corrupt/interrupted run state; do not guess resume position')
        if best_epoch and not (run_dir/'best.pt').exists():
            raise RuntimeError('Cannot resume: previously selected best.pt missing')
    print(f'{action} {run_dir.name}: {task.epochs} epochs, starting after epoch {start_epoch}, device={device}',flush=True)
    for epoch in range(start_epoch+1,task.epochs+1):
        # Deterministic per-epoch RNG: resuming does not change sampler or worker seeds.
        epoch_seed = (seed * 1000003 + epoch * 9176) % (2**32)
        seed_all(epoch_seed)
        sampler.generator = torch.Generator().manual_seed(epoch_seed)
        loader = DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=workers,
                            persistent_workers=False,pin_memory=device.type=='cuda',
                            generator=torch.Generator().manual_seed(epoch_seed+1))
        model.train()
        total_loss, seen = 0., 0
        started = time.monotonic()
        for images, truth in loader:
            images = images.to(device,non_blocking=True)
            truth = truth.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,dtype=torch.float16,
                                    enabled=device.type=='cuda'):
                logits = model(images)
                loss = (boundary_weighted_loss(logits,truth,boundary_weight)
                        if task.objective=='boundary' else bce_dice(logits,truth,'bce_dice'))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item())*len(images)
            seen += len(images)
        row = {'epoch':epoch,'train_loss':round(total_loss/max(seen,1),7),
               'validation_iou_0_5':None,'validation_dice_0_5':None,
               'seconds':round(time.monotonic()-started,3)}
        if epoch % val_every == 0 or epoch == task.epochs:
            val,_ = evaluate_model(model,splits['val'],'raw',device,batch_size=batch_size,
                                   stride=stride)
            row['validation_iou_0_5'] = float(val['iou'])
            row['validation_dice_0_5'] = float(val['dice'])
            if val['iou'] > best:
                best = float(val['iou'])
                best_epoch = epoch
                _state_save(run_dir/'best.pt',{'model_state_dict':model.state_dict(),
                    'config':cfg,'epoch':epoch,'val_iou_0_5':best})
        history.append(row)
        csv_write(run_dir/'history.csv',history,list(row))
        _state_save(run_dir/'latest.pt',{'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scaler_state_dict':scaler.state_dict(), 'epoch':epoch,
            'best_val_iou':best,'best_epoch':best_epoch,'history':history,'config':cfg})
        print(f'  epoch {epoch:03d}/{task.epochs}: loss={row["train_loss"]:.4f} '
              f'val_iou={row["validation_iou_0_5"] if row["validation_iou_0_5"] is not None else "skipped"} '
              f'best={best:.4f} epoch={best_epoch}',flush=True)
    write_json(run_dir/'completed.json',{'completed_epochs':task.epochs,
               'best_val_iou_0_5':best,'best_epoch':best_epoch})
    optimize_threshold(run_dir,batch_size=batch_size,device_name=str(device))
    return run_dir


def run_study(phase='duration',seeds=SEEDS,batch_size=2,lr=1e-4,workers=2,
              stride=512,val_every=2,small_area=100.,small_fraction=.35,
              boundary_weight=.5,device='auto',dry_run=False):
    check_prepared()
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        raise ValueError('Supply distinct nonnegative seeds')
    tasks=planned_tasks(phase)
    plan=[]
    # Preflight ALL runs before expensive work: no accidental mixed settings.
    for task in tasks:
        for seed in seeds:
            cfg=config_for_task(task,seed,batch_size,lr,stride,workers,val_every,
                                small_area,small_fraction,boundary_weight)
            run=task.run_dir(seed)
            plan.append((task,seed,run,_inspect_run(run,cfg)))
    print(f'V6 {phase}: frozen split {sha256_file(SPLITS)[:12]} | lr={lr} | batch={batch_size} | val_every={val_every}',flush=True)
    for task,seed,run,action in plan:
        print(f'  {action:6} {run.name}',flush=True)
    if dry_run:
        print('Dry run: no modifications to runs, no training, no test evaluation.')
        return plan
    for task,seed,run,action in plan:
        train_task(task,seed,batch_size,lr,workers,stride,val_every,
                   small_area,small_fraction,boundary_weight,device)
    print('V6 training complete; use python exp.py v6-report to compare validation results. TEST untouched.')
    return plan


def _collect_rows(tasks,seeds,pixel_only=False,batch_size=2,device='auto'):
    rows=[]
    for task in tasks:
        for seed in seeds:
            run=task.run_dir(seed)
            if not (run/'completed.json').exists() or not (run/'threshold.json').exists():
                print(f'MISSING {run.name} (not included in summaries)')
                continue
            cfg=read_json(run/'config.json')
            if cfg['split_sha256'] != sha256_file(SPLITS) or cfg['manifest_sha256'] != sha256_file(OUT/'manifest.json'):
                raise RuntimeError(f'Dataset/split provenance changed: {run}')
            tuned=read_json(run/'threshold.json')
            done=read_json(run/'completed.json')
            row={'run':run.name,'model':task.base,'sampling':task.sampling,
                 'objective':task.objective,'epochs':task.epochs,'seed':seed,
                 'best_epoch':done['best_epoch'],'pixel_threshold':tuned['threshold'],
                 'val_iou':float(tuned['validation']['iou']),
                 'val_dice':float(tuned['validation']['dice']),
                 'instance_f1':None,'small_recall':None,'small_matched':None,
                 'instance_precision':None,'instance_recall':None,
                 'split_sha256':cfg['split_sha256'],
                 'checkpoint_sha256':sha256_file(run/'best.pt')}
            if not pixel_only:
                from .v4 import validation_tiles
                from .v5 import Policy, score_policy
                tiles=validation_tiles(run,batch_size=batch_size,device_name=device)
                metrics=score_policy(tiles,Policy('baseline_saved'),float(tuned['threshold']))
                row.update(instance_f1=metrics['instance_f1'],small_recall=metrics['small_recall'],
                           small_matched=metrics['small_matched'],
                           instance_precision=metrics['instance_precision'],
                           instance_recall=metrics['instance_recall'])
                if abs(metrics['pixel_iou']-row['val_iou'])>1e-4:
                    raise RuntimeError(f'Cached full-tile predictions disagree with saved validation IoU: {run}')
            rows.append(row)
    return rows


def _plot_curves(tasks,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for task in tasks:
        curves=[]
        for seed in seeds:
            h=task.run_dir(seed)/'history.csv'
            if not h.exists():
                continue
            import csv
            with h.open() as f:
                records=list(csv.DictReader(f))
            ep=[int(r['epoch']) for r in records if r['validation_iou_0_5']]
            values=[float(r['validation_iou_0_5']) for r in records if r['validation_iou_0_5']]
            if ep:
                axes[0].plot(ep,values,alpha=.32,label=None)
                curves.append((ep,values))
            ep=[int(r['epoch']) for r in records]
            losses=[float(r['train_loss']) for r in records]
            if losses:
                axes[1].plot(ep,losses,alpha=.25)
        if curves:
            # One bold mean trace when each seed has same evaluation epoch grid.
            if len({tuple(x[0]) for x in curves})==1:
                xs=curves[0][0]
                avg=np.mean([x[1] for x in curves],axis=0)
                axes[0].plot(xs,avg,linewidth=2,label=f'{task.base} / {task.sampling} / {task.objective} / {task.epochs}e')
    axes[0].set_title('Validation IoU @ 0.5 by epoch (best checkpoint chosen here)')
    axes[1].set_title('Training loss by epoch (individual seeds)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Full-tile global pixel IoU')
    axes[1].set_ylabel('Train loss (not comparable across different loss objectives)')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    ROOT.mkdir(parents=True,exist_ok=True)
    fig.savefig(ROOT/'validation_learning_curves.png',dpi=160)
    plt.close(fig)


def validation_report(phase='full',seeds=SEEDS,pixel_only=False,batch_size=2,device='auto'):
    check_prepared()
    tasks=planned_tasks(phase)
    rows=_collect_rows(tasks,seeds,pixel_only,batch_size,device)
    if not rows:
        print('No completed v6 runs to report. Use v6-study first.')
        return []
    ROOT.mkdir(parents=True,exist_ok=True)
    csv_write(ROOT/'validation_by_seed.csv',rows,list(rows[0]))
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['model'],r['sampling'],r['objective'],r['epochs'])].append(r)
    summary=[]
    print('\nV6 VALIDATION ONLY (test set not accessed)')
    print(f'{"Experiment":62} {"n":>3} {"IoU":>8} {"SD":>8} {"Inst.F1":>9} {"Small":>8}')
    for key,entries in sorted(grouped.items()):
        ious=[r['val_iou'] for r in entries]
        f1=[r['instance_f1'] for r in entries if r['instance_f1'] is not None]
        small=[r['small_recall'] for r in entries if r['small_recall'] is not None]
        result={'model':key[0], 'sampling':key[1],'objective':key[2], 'epochs':key[3],
                'n':len(entries),'mean_val_iou':statistics.mean(ious),
                'std_val_iou':statistics.stdev(ious) if len(ious)>1 else None,
                'mean_instance_f1':statistics.mean(f1) if f1 else None,
                'mean_small_recall':statistics.mean(small) if small else None,
                'seeds':','.join(str(r['seed']) for r in entries)}
        summary.append(result)
        sd=result['std_val_iou'] if result['std_val_iou'] is not None else float('nan')
        inst=result['mean_instance_f1'] if result['mean_instance_f1'] is not None else float('nan')
        sm=result['mean_small_recall'] if result['mean_small_recall'] is not None else float('nan')
        label=f'{key[0]} | {key[1]} | {key[2]} | {key[3]} epochs'
        print(f'{label:62} {len(entries):3d} {result["mean_val_iou"]:8.4f} {sd:8.4f} {inst:9.4f} {sm:8.4f}')
    csv_write(ROOT/'validation_summary.csv',summary,list(summary[0]))
    by_identity={(r['model'],r['sampling'],r['objective'],r['epochs'],r['seed']):r for r in rows}
    pairs=[
        ('raw 60 vs raw 16',('raw_bce_dice','balanced','bce_dice',60),('raw_bce_dice','balanced','bce_dice',16)),
        ('photo 60 vs photo 16',('augmentation_photometric','balanced','bce_dice',60),('augmentation_photometric','balanced','bce_dice',16)),
        ('small-aware vs balanced @60',('augmentation_photometric','small_aware','bce_dice',60),('augmentation_photometric','balanced','bce_dice',60)),
        ('boundary vs normal @60',('augmentation_photometric','balanced','boundary',60),('augmentation_photometric','balanced','bce_dice',60)),
    ]
    paired=[]
    for label,a,b in pairs:
        for seed in seeds:
            x=by_identity.get((*a,seed)); y=by_identity.get((*b,seed))
            if x is None or y is None:
                continue
            paired.append({'comparison':label,'seed':seed,'delta_val_iou':x['val_iou']-y['val_iou'],
                           'delta_instance_f1':(x['instance_f1']-y['instance_f1']) if x['instance_f1'] is not None and y['instance_f1'] is not None else None,
                           'delta_small_recall':(x['small_recall']-y['small_recall']) if x['small_recall'] is not None and y['small_recall'] is not None else None})
    if paired:
        csv_write(ROOT/'validation_paired_deltas.csv',paired,list(paired[0]))
        print('\nPAIRED VALIDATION DIFFERENCES (same seed; positive favors first method)')
        for label in dict.fromkeys(r['comparison'] for r in paired):
            group=[r for r in paired if r['comparison']==label]
            iou=statistics.mean(r['delta_val_iou'] for r in group)
            f1=[r['delta_instance_f1'] for r in group if r['delta_instance_f1'] is not None]
            print(f'  {label}: mean delta IoU={iou:+.4f}; delta instance F1={statistics.mean(f1) if f1 else float("nan"):+.4f}; n={len(group)}')
    _plot_curves(tasks,seeds)
    print('Saved:',ROOT/'validation_summary.csv',ROOT/'validation_paired_deltas.csv',ROOT/'validation_learning_curves.png')
    print('Report selected validation models; do not run finalize --all for selection.')
    return rows
