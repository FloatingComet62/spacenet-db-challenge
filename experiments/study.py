"""Run a deliberately controlled multi-seed study without touching the test set."""
from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict

from .common import OUT, SPLITS, check_prepared, csv_write, read_json
from .engine import config_for, optimize_threshold, train_one

CONFIRM = ('raw_bce_dice', 'clahe')
AUGMENT = ('augmentation', 'augmentation_geometry', 'augmentation_photometric')
GEOMETRY = ('augmentation_flip', 'augmentation_rotation')
ALL = CONFIRM + AUGMENT


def planned_tasks(phase='initial', seeds=(111, 222, 333)):
    if len(set(seeds)) != len(seeds) or not seeds or any(s < 0 for s in seeds):
        raise ValueError('Supply distinct nonnegative seeds')
    if phase == 'geometry':
        return [(n, s) for n in GEOMETRY for s in seeds]
    if phase == 'confirm':
        return [(n, s) for n in CONFIRM for s in seeds]
    if phase == 'diagnose':
        return [(n, seeds[0]) for n in AUGMENT]
    if phase == 'initial':
        return ([(n, s) for n in CONFIRM for s in seeds]
                + [(n, seeds[0]) for n in AUGMENT])
    if phase == 'full':
        return [(n, s) for n in ALL for s in seeds]
    raise ValueError(f'Unknown phase: {phase}')


def _validate_existing(run_dir, name, seed, epochs, batch_size, lr, stride, split_sha256, workers=2):
    """Never silently mix different budgets/splits or overwrite prior experiments."""
    if not run_dir.is_dir():
        return 'TRAIN'
    if not all((run_dir / filename).is_file() for filename in
               ('config.json', 'completed.json', 'best.pt')):
        raise RuntimeError(f'Interrupted run at {run_dir}; inspect it and remove/rename '
                           'that run directory before rerunning. Existing data were not modified.')
    actual = read_json(run_dir / 'config.json')
    expected = config_for(name)
    expected.update({'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
                     'lr': lr, 'stride': stride, 'workers': workers, 'split_sha256': split_sha256})
    keys = ('name', 'preprocessor', 'loss', 'scheduler', 'seed', 'epochs',
            'batch_size', 'lr', 'stride', 'workers', 'split_sha256')
    mismatches = [(key, actual.get(key), expected[key]) for key in keys
                  if actual.get(key) != expected[key]]
    original_mode = actual.get('augment_mode',
                               'combined' if actual.get('augment', False) else 'none')
    if original_mode != expected['augment_mode']:
        mismatches.append(('augment_mode', original_mode, expected['augment_mode']))
    if mismatches:
        details = ', '.join(f'{key}: stored={stored!r}, requested={wanted!r}'
                            for key, stored, wanted in mismatches)
        raise RuntimeError(f'Cannot reuse {run_dir}: {details}. '
                           'Choose matching parameters, or manually use a different run identity.')
    completed = read_json(run_dir / 'completed.json')
    if completed.get('completed_epochs') != epochs:
        raise RuntimeError(f'Incomplete run at {run_dir}: expected {epochs} epochs')
    return 'REUSE'


def run_study(phase='initial', seeds=(111, 222, 333), epochs=16,
              batch_size=2, lr=1e-4, workers=2, stride=512,
              device='auto', dry_run=False):
    check_prepared()
    tasks = planned_tasks(phase, seeds)
    split_hash = hashlib.sha256(SPLITS.read_bytes()).hexdigest()
    # Verify every existing run *before* starting new work.
    plan = []
    for name, seed in tasks:
        run_dir = OUT / 'runs' / f'{name}_seed{seed}'
        action = _validate_existing(run_dir, name, seed, epochs, batch_size,
                                    lr, stride, split_hash, workers=workers)
        plan.append((name, seed, run_dir, action))

    print(f'Phase: {phase} | {epochs} epochs | batch={batch_size} | lr={lr:g} | stride={stride}')
    print('Using frozen split:', SPLITS, f'(SHA256 {split_hash[:12]})')
    print('No calls to test evaluation are made by this command.')
    for name, seed, run_dir, action in plan:
        threshold = 'TUNE' if action == 'REUSE' and not (run_dir / 'threshold.json').is_file() else ''
        print(f'  {action:5} {name}_seed{seed} {threshold}')
    if dry_run:
        print('Dry run: no files written.')
        return plan

    for name, seed, run_dir, action in plan:
        if action == 'TRAIN':
            train_one(name, seed=seed, epochs=epochs, batch_size=batch_size,
                      lr=lr, workers=workers, stride=stride, device_name=device)
        else:
            print(f'Reusing {run_dir}/best.pt')
        if not (run_dir / 'threshold.json').is_file():
            optimize_threshold(run_dir, batch_size=batch_size, device_name=device)
    validation_report(names=(ALL if phase in ('initial', 'full') else
                             CONFIRM if phase == 'confirm' else
                             GEOMETRY + ('raw_bce_dice', 'augmentation_geometry') if phase == 'geometry' else AUGMENT), seeds=seeds)
    print('Finished validation study. Do NOT use finalize --all for model selection.')
    return plan


def validation_report(names=None, seeds=(111, 222, 333)):
    """Summarize validation metrics and paired seed differences, never read test metrics."""
    names = list(names or ALL)
    rows = []
    skipped = []
    for name in names:
        for seed in seeds:
            run_dir = OUT / 'runs' / f'{name}_seed{seed}'
            threshold_path = run_dir / 'threshold.json'
            config_path = run_dir / 'config.json'
            if not threshold_path.is_file() or not config_path.is_file():
                skipped.append(f'{name}_seed{seed}')
                continue
            cfg = read_json(config_path)
            tuned = read_json(threshold_path)
            if cfg.get('seed') != seed or cfg.get('name') != name:
                raise ValueError(f'Run identity mismatch: {run_dir}')
            if SPLITS.is_file():
                current_split_hash = hashlib.sha256(SPLITS.read_bytes()).hexdigest()
                if cfg.get('split_sha256') != current_split_hash:
                    raise ValueError(f'Split mismatch: {run_dir}')
            m = tuned['validation']
            rows.append({'experiment':name, 'seed':seed,
                         'epochs':cfg['epochs'], 'batch_size':cfg['batch_size'],
                         'lr':cfg['lr'], 'stride':cfg['stride'],
                         'threshold':tuned['threshold'],
                         'val_iou':m['iou'], 'val_dice':m['dice'],
                         'val_precision':m['precision'], 'val_recall':m['recall']})
    if not rows:
        print('No tuned validation runs to report. Run python exp.py study first.')
        return []
    # A group mean is only meaningful when training settings and split are consistent.
    configs = {(row['epochs'], row['batch_size'], row['lr'], row['stride']) for row in rows}
    if len(configs) != 1:
        raise ValueError(f'Cannot aggregate different training budgets: {configs}')
    csv_write(OUT / 'validation_by_seed.csv', rows, list(rows[0]))
    by_name = defaultdict(list)
    for row in rows:
        by_name[row['experiment']].append(row)
    summary = []
    print('\nVALIDATION (selected threshold; test set not accessed)')
    print(f"{'Experiment':29} {'n':>3} {'Mean IoU':>10} {'Std IoU':>10} {'Mean Dice':>10}")
    for name, entries in sorted(by_name.items()):
        ious = [float(x['val_iou']) for x in entries]
        dices = [float(x['val_dice']) for x in entries]
        std = statistics.stdev(ious) if len(ious) > 1 else None
        r = {'experiment':name,'n':len(ious),'mean_val_iou':statistics.mean(ious),
             'std_val_iou':std,'mean_val_dice':statistics.mean(dices),
             'seeds':','.join(str(x['seed']) for x in entries)}
        summary.append(r)
        print(f"{name:29} {len(ious):3d} {r['mean_val_iou']:10.4f} "
              f"{std if std is not None else float('nan'):10.4f} {r['mean_val_dice']:10.4f}")
    csv_write(OUT / 'validation_summary.csv', summary, list(summary[0]))
    paired = []
    raw_by_seed = {r['seed']: r for r in by_name.get('raw_bce_dice', [])}
    for name in names:
        if name == 'raw_bce_dice':
            continue
        deltas = [(r['seed'],r['val_iou'] - raw_by_seed[r['seed']]['val_iou'])
                  for r in by_name.get(name, []) if r['seed'] in raw_by_seed]
        if deltas:
            paired.append({'experiment':name,'baseline':'raw_bce_dice',
                           'paired_seeds':len(deltas),
                           'mean_delta_iou':statistics.mean(d for _,d in deltas),
                           'std_delta_iou':statistics.stdev(d for _,d in deltas) if len(deltas)>1 else None,
                           'seeds':','.join(str(s) for s,_ in deltas)})
    if paired:
        csv_write(OUT/'validation_paired_deltas.csv',paired,list(paired[0]))
        print('\nPAIRED VALIDATION IoU DIFFERENCE (experiment minus raw_bce_dice)')
        for r in paired:
            print(f"  {r['experiment']}: {r['mean_delta_iou']:+.4f} (n={r['paired_seeds']})")
    if skipped:
        print('\nNot yet available:', ', '.join(skipped))
    print('Wrote experiments_output/validation_by_seed.csv and validation_summary.csv')
    print('n=1 has no standard deviation; do not treat it as evidence of reproducibility.')
    return rows
