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
    pp=sub.add_parser('postprocess-study',help='VALIDATION ONLY: instance-F1 threshold, filtering, watershed search')
    pp.add_argument('--run',required=True)
    pp.add_argument('--batch-size',type=int,default=2)
    pp.add_argument('--device',default='auto')
    pp.add_argument('--refresh-cache',action='store_true')
    pp.add_argument('--thresholds',type=float,nargs='+',default=[.15,.20,.25,.30,.35,.40,.50,.60])
    pp.add_argument('--areas',type=int,nargs='+',default=[0,8,16,32,64])
    pp.add_argument('--peak-distances',type=int,nargs='+',default=[6,12,20])
    pp.add_argument('--watershed-top',type=int,default=2,help='How many strong binary configurations to try watershed on; 0 skips')
    pp.add_argument('--max-iou-drop',type=float,default=0.0,help='Maximum tolerated global pixel IoU loss versus unchanged v3 baseline; default 0')
    ia=sub.add_parser('instance-audit',help='VALIDATION ONLY: diagnose unmatched GT buildings and save crop gallery')
    ia.add_argument('--run',required=True)
    ia.add_argument('--selection',choices=['best','baseline'],default='best')
    ia.add_argument('--batch-size',type=int,default=2)
    ia.add_argument('--device',default='auto')
    ia.add_argument('--refresh-cache',action='store_true')
    ia.add_argument('--count',type=int,default=12)
    ia.add_argument('--small-max',type=float,default=100)
    ia.add_argument('--medium-max',type=float,default=1000)
    ex4=sub.add_parser('compare-v4-external',help='VALIDATION ONLY: pixel+instance scores for genuine external masks on same tiles')
    ex4.add_argument('--run',required=True)
    ex4.add_argument('--pred-dir',required=True)
    ex4.add_argument('--selection',choices=['best','baseline'],default='best')
    ex4.add_argument('--input-type',choices=['binary','probabilities','logits'],default='binary')
    ex4.add_argument('--threshold',type=float,default=.5)
    ex4.add_argument('--channel',type=int,default=0)
    ex4.add_argument('--batch-size',type=int,default=2)
    ex4.add_argument('--device',default='auto')
    ex4.add_argument('--refresh-cache',action='store_true')
    p5=sub.add_parser('v5-study',help='VALIDATION ONLY: select common confidence-aware policy across seeds')
    p5.add_argument('--models',nargs='+',default=['raw_bce_dice','augmentation_photometric'])
    p5.add_argument('--seeds',type=int,nargs='+',default=[111,222,333])
    p5.add_argument('--batch-size',type=int,default=2)
    p5.add_argument('--device',default='auto')
    p5.add_argument('--refresh-cache',action='store_true')
    p5.add_argument('--thresholds',nargs='+',default=['saved','0.25','0.35'])
    p5.add_argument('--small-areas',type=int,nargs='+',default=[8,16])
    p5.add_argument('--confidence-cutoffs',type=float,nargs='+',default=[.60,.80])
    p5.add_argument('--confidence-metrics',nargs='+',choices=['mean','p90','max'],default=['mean','p90','max'])
    p5.add_argument('--large-area',type=int,default=64)
    p5.add_argument('--max-iou-drop',type=float,default=.005)
    p5.add_argument('--small-recall-slack',type=int,default=0,help='Max number of small GT matches allowed below baseline for EACH seed')
    p5.add_argument('--small-max',type=float,default=100)
    p5.add_argument('--medium-max',type=float,default=1000)
    g5=sub.add_parser('v5-gallery',help='VALIDATION ONLY: small-building v3/v4/v5 before-after gallery')
    g5.add_argument('--model',default='augmentation_photometric')
    g5.add_argument('--seed',type=int,default=111)
    g5.add_argument('--count',type=int,default=6)
    g5.add_argument('--batch-size',type=int,default=2)
    g5.add_argument('--device',default='auto')
    g5.add_argument('--refresh-cache',action='store_true')
    v6=sub.add_parser('v6-study',help='VALIDATION ONLY: resumable 16/60 epoch, sampling and boundary studies')
    v6.add_argument('--phase',choices=['duration','sampling','boundary','full'],default='duration')
    v6.add_argument('--seeds',nargs='+',type=int,default=[111,222,333])
    v6.add_argument('--batch-size',type=int,default=2)
    v6.add_argument('--lr',type=float,default=1e-4)
    v6.add_argument('--workers',type=int,default=2)
    v6.add_argument('--stride',type=int,choices=[256,512],default=512)
    v6.add_argument('--val-every',type=int,default=2)
    v6.add_argument('--small-area',type=float,default=100.)
    v6.add_argument('--small-positive-fraction',type=float,default=.35)
    v6.add_argument('--boundary-weight',type=float,default=.5)
    v6.add_argument('--device',default='auto')
    v6.add_argument('--dry-run',action='store_true')
    v6rep=sub.add_parser('v6-report',help='VALIDATION ONLY: paired duration/sampling/size reports and curves')
    v6rep.add_argument('--phase',choices=['duration','sampling','boundary','full'],default='full')
    v6rep.add_argument('--seeds',nargs='+',type=int,default=[111,222,333])
    v6rep.add_argument('--pixel-only',action='store_true')
    v6rep.add_argument('--batch-size',type=int,default=2)
    v6rep.add_argument('--device',default='auto')
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
    elif args.action=='postprocess-study':
        from experiments.v4 import postprocess_study
        postprocess_study(args.run,args.batch_size,args.device,args.refresh_cache,
                          args.thresholds,args.areas,args.peak_distances,args.watershed_top,args.max_iou_drop)
    elif args.action=='instance-audit':
        from experiments.v4 import instance_audit
        instance_audit(args.run,args.selection,args.batch_size,args.device,args.refresh_cache,
                       args.count,args.small_max,args.medium_max)
    elif args.action=='compare-v4-external':
        from experiments.v4 import compare_external_instances
        compare_external_instances(args.run,args.pred_dir,args.selection,args.input_type,
                                   args.threshold,args.channel,args.batch_size,args.device,args.refresh_cache)
    elif args.action=='v5-study':
        from experiments.v5 import study
        study(args.models,args.seeds,args.batch_size,args.device,args.refresh_cache,
              args.thresholds,args.small_areas,args.confidence_cutoffs,
              args.confidence_metrics,args.large_area,args.max_iou_drop,
              args.small_recall_slack,args.small_max,args.medium_max)
    elif args.action=='v6-study':
        from experiments.v6 import run_study
        run_study(phase=args.phase,seeds=args.seeds,batch_size=args.batch_size,lr=args.lr,
                  workers=args.workers,stride=args.stride,val_every=args.val_every,
                  small_area=args.small_area,small_fraction=args.small_positive_fraction,
                  boundary_weight=args.boundary_weight,device=args.device,dry_run=args.dry_run)
    elif args.action=='v6-report':
        from experiments.v6 import validation_report
        validation_report(phase=args.phase,seeds=args.seeds,pixel_only=args.pixel_only,
                          batch_size=args.batch_size,device=args.device)
    elif args.action=='v5-gallery':
        from experiments.v5 import gallery
        gallery(args.model,args.seed,args.count,args.batch_size,args.device,args.refresh_cache)
    elif args.action=='export-official-csv':
        from experiments.compare import export_official_csv
        export_official_csv(args.input_csv)


if __name__ == '__main__':
    main()
