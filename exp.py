#!/usr/bin/env python3
"""Entry point: run from repo root (alongside src/ and data/)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='SpaceNet8 Germany building segmentation experiment suite')
    sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare',help='Rebuild masks and freeze spatial train/val/test splits')
    prep.add_argument('--seed',type=int,default=111)
    prep.add_argument('--block-size',type=int,default=4)
    prep.add_argument('--split-mode',choices=['spatial','tile'],default='spatial')
    prep.add_argument('--force',action='store_true')
    sub.add_parser('inspect',help='Print data quality and split summary')
    sub.add_parser('doctor',help='Check dependencies and required input paths')
    train=sub.add_parser('run',help='Train a set of controlled experiments')
    train.add_argument('--preset',choices=['quick','all'],default='quick')
    train.add_argument('--names',nargs='+',help='Override preset with experiment names')
    train.add_argument('--epochs',type=int,default=None)
    train.add_argument('--seed',type=int,default=111)
    train.add_argument('--batch-size',type=int,default=2)
    train.add_argument('--lr',type=float,default=1e-4)
    train.add_argument('--workers',type=int,default=2)
    train.add_argument('--stride',type=int,choices=[256,512],default=512)
    train.add_argument('--device',default='auto')
    train.add_argument('--overwrite',action='store_true')
    train.add_argument('--skip-existing',action='store_true',help='Skip runs with existing best.pt checkpoints')
    study=sub.add_parser('study',help='Reproducibility study; trains only missing runs, validation-only')
    study.add_argument('--phase',choices=['initial','full','confirm','diagnose','geometry'],default='initial')
    study.add_argument('--seeds',nargs='+',type=int,default=[111,222,333])
    study.add_argument('--epochs',type=int,default=16)
    study.add_argument('--batch-size',type=int,default=2)
    study.add_argument('--lr',type=float,default=1e-4)
    study.add_argument('--workers',type=int,default=2)
    study.add_argument('--stride',type=int,choices=[256,512],default=512)
    study.add_argument('--device',default='auto')
    study.add_argument('--dry-run',action='store_true')
    report=sub.add_parser('validation-report',help='Aggregate tuned VALIDATION IoU across seeds')
    report.add_argument('--names',nargs='+',default=None)
    report.add_argument('--seeds',nargs='+',type=int,default=[111,222,333])
    inspect_aug=sub.add_parser('inspect-augmentation',help='Render training patch/mask augmentations')
    inspect_aug.add_argument('--count',type=int,default=4)
    inspect_aug.add_argument('--seed',type=int,default=111)
    errors=sub.add_parser('inspect-errors',help='Rank false positives on VALIDATION tiles and save overlays')
    errors.add_argument('--run',required=True)
    errors.add_argument('--count',type=int,default=6)
    errors.add_argument('--batch-size',type=int,default=2)
    errors.add_argument('--device',default='auto')
    tune=sub.add_parser('tune',help='Select a prediction threshold using validation only')
    tune.add_argument('--run',required=True,help='Directory under experiments_output/runs/')
    tune.add_argument('--batch-size',type=int,default=2)
    tune.add_argument('--device',default='auto')
    final=sub.add_parser('finalize',help='Evaluate trained model(s) once on held-out test tiles')
    final.add_argument('--run',help='Run directory (default: use --all)')
    final.add_argument('--all',action='store_true',help='Evaluate every finished run')
    final.add_argument('--batch-size',type=int,default=2)
    final.add_argument('--device',default='auto')
    final.add_argument('--no-predictions',action='store_true')
    sub.add_parser('results',help='Show test leaderboard and write results.csv')
    ext=sub.add_parser('compare-external',help='Evaluate masks exported by official or other models')
    ext.add_argument('--pred-dir',required=True)
    ext.add_argument('--split',choices=['val','test'],default='test')
    ext.add_argument('--input-type',choices=['binary','probabilities','logits'],default='binary')
    ext.add_argument('--threshold',type=float,default=.5)
    ext.add_argument('--name',default='official_external')
    ext.add_argument('--channel',type=int,default=0)
    ext.add_argument('--min-area',type=float,default=16)
    ext.add_argument('--no-instances',action='store_true')
    vis=sub.add_parser('visualize-compare',help='Save side-by-side overlays from two prediction directories')
    vis.add_argument('--a-dir',required=True)
    vis.add_argument('--b-dir',required=True)
    vis.add_argument('--a-kind',choices=['binary','probabilities','logits'],default='probabilities')
    vis.add_argument('--b-kind',choices=['binary','probabilities','logits'],default='binary')
    vis.add_argument('--a-threshold',type=float,default=.5)
    vis.add_argument('--b-threshold',type=float,default=.5)
    vis.add_argument('--split',choices=['val','test'],default='test')
    vis.add_argument('--count',type=int,default=8)
    official=sub.add_parser('export-official-csv',help='Filter baseline-prepared official CSV(s) to identical splits')
    official.add_argument('--input-csv',nargs='+',required=True)
    threshold3=sub.add_parser('threshold-curves',help='Plot threshold sensitivity using validation ONLY')
    threshold3.add_argument('--run',required=True)
    threshold3.add_argument('--batch-size',type=int,default=2)
    threshold3.add_argument('--device',default='auto')
    threshold3.add_argument('--refresh-cache',action='store_true')
    sizes=sub.add_parser('building-sizes',help='Building-object recall stratified by GT footprint area; VALIDATION only')
    sizes.add_argument('--run',required=True)
    sizes.add_argument('--small-max',type=float,default=100)
    sizes.add_argument('--medium-max',type=float,default=1000)
    sizes.add_argument('--min-area',type=float,default=16)
    sizes.add_argument('--match-iou',type=float,default=.5)
    sizes.add_argument('--batch-size',type=int,default=2)
    sizes.add_argument('--device',default='auto')
    sizes.add_argument('--refresh-cache',action='store_true')
    ann=sub.add_parser('annotation-audit',help='Check WKT-vs-prepared GT and manually review suspicious predictions (VALIDATION)')
    ann.add_argument('--run',default=None)
    ann.add_argument('--boundary-px',type=int,default=3)
    ann.add_argument('--count',type=int,default=6)
    ann.add_argument('--batch-size',type=int,default=2)
    ann.add_argument('--device',default='auto')
    ann.add_argument('--refresh-cache',action='store_true')
    geo=sub.add_parser('geometry-audit',help='Audit synchronized geometric transforms and preview TRAIN patches')
    geo.add_argument('--samples',type=int,default=40)
    geo.add_argument('--count',type=int,default=4)
    geo.add_argument('--seed',type=int,default=111)
    comp3=sub.add_parser('compare-official-validation',help='Pair external building masks against internal run on SAME validation tiles')
    comp3.add_argument('--run',required=True)
    comp3.add_argument('--pred-dir',required=True)
    comp3.add_argument('--input-type',choices=['binary','probabilities','logits'],default='binary')
    comp3.add_argument('--threshold',type=float,default=.5)
    comp3.add_argument('--channel',type=int,default=0)
    comp3.add_argument('--batch-size',type=int,default=2)
    comp3.add_argument('--device',default='auto')
    comp3.add_argument('--refresh-cache',action='store_true')
    sub.add_parser('split-neighbor-audit',help='List edge-adjacent tiles in different frozen splits')
    args=p.parse_args()
    if args.action=='prepare':
        from experiments.data import prepare
        prepare(args.seed,args.block_size,args.split_mode,args.force)
    elif args.action=='doctor':
        from importlib import import_module
        modules=['numpy','torch','segmentation_models_pytorch','rasterio','shapely','cv2','PIL','matplotlib']
        missing=[]
        for name in modules:
            try:
                mod=import_module(name)
                print(f'{name}: OK ({getattr(mod, "__version__", "installed")})')
            except Exception as exc:
                missing.append(name)
                print(f'{name}: MISSING ({exc})')
        for path in [Path('data/PRE-event'),Path('data/Germany_Training_Public_reference.csv')]:
            print(path, ':', 'FOUND' if path.exists() else 'MISSING')
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise SystemExit('Missing requirements: '+', '.join(missing))
    elif args.action=='inspect':
        from experiments.data import inspect
        inspect()
    elif args.action=='run':
        from experiments.engine import PRESETS, train_one
        names=args.names or PRESETS[args.preset]
        for name in names:
            existing = Path('experiments_output/runs')/f'{name}_seed{args.seed}'
            if args.skip_existing and (existing/'completed.json').exists() and not args.overwrite:
                print(f'Skipping completed run: {existing}')
                continue
            train_one(name,seed=args.seed,epochs=args.epochs or (16 if args.preset=='quick' else 50),
                      batch_size=args.batch_size,lr=args.lr,workers=args.workers,
                      stride=args.stride,overwrite=args.overwrite,device_name=args.device)
    elif args.action=='study':
        from experiments.study import run_study
        run_study(phase=args.phase,seeds=args.seeds,epochs=args.epochs,
                  batch_size=args.batch_size,lr=args.lr,workers=args.workers,
                  stride=args.stride,device=args.device,dry_run=args.dry_run)
    elif args.action=='validation-report':
        from experiments.study import validation_report
        validation_report(names=args.names,seeds=args.seeds)
    elif args.action=='inspect-augmentation':
        from experiments.diagnostics import save_augmentation_preview
        save_augmentation_preview(count=args.count,seed=args.seed)
    elif args.action=='inspect-errors':
        from experiments.error_analysis import save_validation_errors
        save_validation_errors(args.run,count=args.count,batch_size=args.batch_size,
                               device_name=args.device)
    elif args.action=='tune':
        from experiments.engine import optimize_threshold
        optimize_threshold(Path(args.run),args.batch_size,args.device)
    elif args.action=='finalize':
        from experiments.engine import finalize,results_table
        if not args.run and not args.all:
            p.error('finalize needs --run <directory> or --all')
        paths = sorted(Path('experiments_output/runs').glob('*/best.pt')) if args.all else [Path(args.run)/'best.pt']
        if not paths:
            raise RuntimeError('No trained checkpoints to finalize')
        for path in paths:
            finalize(path.parent,batch_size=args.batch_size,device_name=args.device,
                     save_predictions=not args.no_predictions)
        results_table()
    elif args.action=='results':
        from experiments.engine import results_table
        results_table()
    elif args.action=='compare-external':
        from experiments.compare import evaluate_external
        evaluate_external(args.pred_dir,args.split,args.input_type,args.threshold,
                          args.min_area,not args.no_instances,args.name,args.channel)
    elif args.action=='visualize-compare':
        from experiments.visuals import pairwise_visuals
        pairwise_visuals(args.a_dir,args.b_dir,args.a_kind,args.b_kind,
                         args.a_threshold,args.b_threshold,args.split,args.count)
    elif args.action=='split-neighbor-audit':
        from experiments.v3 import split_neighbor_audit
        split_neighbor_audit()
    elif args.action=='threshold-curves':
        from experiments.v3 import threshold_curves
        threshold_curves(args.run,args.batch_size,args.device,args.refresh_cache)
    elif args.action=='building-sizes':
        from experiments.v3 import building_sizes
        building_sizes(args.run,args.small_max,args.medium_max,args.min_area,
                       args.match_iou,args.batch_size,args.device,args.refresh_cache)
    elif args.action=='annotation-audit':
        from experiments.v3 import annotation_audit
        annotation_audit(args.run,args.batch_size,args.device,args.refresh_cache,args.boundary_px,args.count)
    elif args.action=='geometry-audit':
        from experiments.v3 import geometry_audit
        geometry_audit(args.samples,args.count,args.seed)
    elif args.action=='compare-official-validation':
        from experiments.v3 import compare_official_validation
        compare_official_validation(args.run,args.pred_dir,args.input_type,args.threshold,
                                    args.channel,args.batch_size,args.device,args.refresh_cache)
    elif args.action=='export-official-csv':
        from experiments.compare import export_official_csv
        export_official_csv(args.input_csv)


if __name__ == '__main__':
    main()
