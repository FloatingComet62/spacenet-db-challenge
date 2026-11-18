"""v5 validation-only confidence-aware connected-component postprocessing.

A shared policy is chosen per model family across all requested random seeds.
No test tiles, training, checkpoints, split definitions or v4 reports are altered.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import box

from .common import OUT, SPLITS, csv_write, load_tile, read_json, sha256_file, write_json
from .engine import metrics_from_counts
from .v3 import _plot, _valid_run, match_objects, predicted_polygons, size_bin
from .v4 import Tile, counts, make_prediction, object_metrics, validation_tiles

OUTDIR = OUT / 'diagnostics_v5'
DEFAULT_MODELS = ('raw_bce_dice', 'augmentation_photometric')
DEFAULT_SEEDS = (111, 222, 333)
DEFAULT_THRESHOLDS = ('saved', '0.25', '0.35')
DEFAULT_SMALL_FLOORS = (8, 16)
DEFAULT_CONFIDENCES = (0.60, 0.80)
DEFAULT_CONFIDENCE_METRICS = ('mean', 'p90', 'max')


@dataclass(frozen=True)
class Policy:
    method: str = 'confidence'   # 'baseline_saved', 'filter' or 'confidence'
    threshold: str = 'saved'      # selected validation threshold or a fixed value
    large_area: int = 64
    small_area: int = 8
    confidence_metric: str = 'mean'
    confidence_cutoff: float = 0.70


def policy_dict(p):
    return asdict(p)


def candidate_policies(thresholds=DEFAULT_THRESHOLDS,
                       small_floors=DEFAULT_SMALL_FLOORS,
                       confidence_cutoffs=DEFAULT_CONFIDENCES,
                       confidence_metrics=DEFAULT_CONFIDENCE_METRICS,
                       large_area=64):
    """Deterministic grid including an always-eligible unchanged baseline fallback."""
    if large_area < 16:
        raise ValueError('large-area must be >= 16, the v3 minimum object polygon area')
    ts = tuple(dict.fromkeys(str(x) for x in thresholds))
    policies = [Policy(method='baseline_saved')]
    for t in ts:
        if t != 'saved' and not 0 < float(t) < 1:
            raise ValueError('All probability thresholds must be in (0,1)')
        policies.append(Policy(method='filter',threshold=t,large_area=large_area))
        for small in small_floors:
            small = int(small)
            if not (1 <= small < large_area):
                raise ValueError('small-area must be >=1 and < large-area')
            for metric in confidence_metrics:
                if metric not in ('mean','p90','max'):
                    raise ValueError('confidence-metric must be mean, p90, or max')
                for cutoff in confidence_cutoffs:
                    cutoff = float(cutoff)
                    if not (0 < cutoff <= 1):
                        raise ValueError('confidence cutoff must be in (0,1]')
                    policies.append(Policy('confidence',t,large_area,small,metric,cutoff))
    return policies


def actual_threshold(policy, tuned):
    return tuned if policy.threshold == 'saved' or policy.method == 'baseline_saved' else float(policy.threshold)


def _components(binary, probabilities):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(binary, np.uint8), connectivity=4)
    return n, labels, stats


def confidence_filter(probabilities, threshold, large_area=64, small_area=8,
                      confidence_metric='mean', confidence_cutoff=.70,
                      return_stats=False):
    """Keep large components; retain smaller ones only when sufficiently confident.

    Four-connectivity matches the v4 filter. Confidence is measured on the
    *original probability map*, not the thresholded mask. All pixels in a
    discarded component are removed; surviving components keep original shape.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2 or not np.isfinite(probs).all():
        raise ValueError('Expected finite HxW probability array')
    if not 0 < threshold < 1 or not (0 <= probs.min() <= probs.max() <= 1):
        raise ValueError('Probabilities and threshold must be in [0,1] and (0,1)')
    if not (1 <= small_area < large_area):
        raise ValueError('Expected 1 <= small_area < large_area')
    if confidence_metric not in ('mean','p90','max') or not 0 < confidence_cutoff <= 1:
        raise ValueError('Invalid confidence metric or cutoff')
    binary = probs >= threshold
    if not binary.any():
        return (binary.copy(), []) if return_stats else binary.copy()
    count, labels, stats = _components(binary, probs)
    keep = np.zeros(count, dtype=bool)
    rows = []
    for label in range(1,count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < small_area:
            continue
        if area >= large_area:
            keep[label] = True
            if return_stats:
                rows.append({'label':label,'area':area,'confidence':None,
                             'kept':True,'reason':'large'})
            continue
        # Restrict inspection to this component's bounding box, not the whole tile.
        x,y,w,h = map(int, stats[label, :4])
        component_roi = labels[y:y+h, x:x+w] == label
        vals = probs[y:y+h, x:x+w][component_roi]
        if confidence_metric == 'mean':
            confidence = float(vals.mean())
        elif confidence_metric == 'p90':
            confidence = float(np.percentile(vals,90))
        else:
            confidence = float(vals.max())
        keep[label] = confidence >= confidence_cutoff
        if return_stats:
            rows.append({'label':label,'area':area,'confidence':confidence,
                         'kept':bool(keep[label]),'reason':'confidence'})
    result = keep[labels]
    return (result,rows) if return_stats else result


def predict_policy(probs, policy, tuned):
    threshold = actual_threshold(policy,tuned)
    if policy.method == 'baseline_saved':
        return make_prediction(probs,'raw',threshold,min_area=16)
    if policy.method == 'filter':
        return make_prediction(probs,'filter',threshold,min_area=policy.large_area)
    if policy.method != 'confidence':
        raise ValueError(f'Invalid policy: {policy.method}')
    mask = confidence_filter(probs, threshold, large_area=policy.large_area,
                             small_area=policy.small_area,
                             confidence_metric=policy.confidence_metric,
                             confidence_cutoff=policy.confidence_cutoff)
    # v4's binary polygon extraction, with no further implicit area filter.
    return mask, predicted_polygons(mask, min_area=0)


def score_policy(tiles, policy, tuned, small_max=100, medium_max=1000,
                 collect_details=False):
    """Full-tile micro metrics; GT object matching uses v3's exact 0.50 convention."""
    px_totals = np.zeros(3, dtype=np.int64)
    ob_totals = np.zeros(3, dtype=np.int64)
    matched_size = {b:0 for b in ('small','medium','large')}
    size_totals = {b:0 for b in ('small','medium','large')}
    details = {}
    for tile in tiles:
        mask, objects = predict_policy(tile.probabilities,policy,tuned)
        pixel, ob = counts(mask,tile.truth,objects,tile.gt_polygons)
        px_totals += pixel
        ob_totals += ob
        matches = match_objects(objects,tile.gt_polygons,match_iou=.5)
        if int(ob[0]) != len(matches):
            raise AssertionError('Inconsistent matching')
        if collect_details:
            details[tile.image_id] = {'matches':set(matches), 'mask':mask, 'objects':objects}
        for idx, polygon in enumerate(tile.gt_polygons):
            b = size_bin(float(polygon.area),small_max,medium_max)
            size_totals[b] += 1
            matched_size[b] += int(idx in matches)
    pixel = metrics_from_counts(*map(int,px_totals))
    object_scores = object_metrics(*map(int,ob_totals))
    metrics = {'pixel_iou':float(pixel['iou']), 'pixel_dice':float(pixel['dice']),
               'instance_f1':float(object_scores['instance_f1']),
               'instance_precision':float(object_scores['instance_precision']),
               'instance_recall':float(object_scores['instance_recall']),
               'instance_tp':int(object_scores['instance_tp']),
               'instance_fp':int(object_scores['instance_fp']),
               'instance_fn':int(object_scores['instance_fn'])}
    for name in ('small','medium','large'):
        metrics[f'{name}_matched'] = matched_size[name]
        metrics[f'{name}_total'] = size_totals[name]
        metrics[f'{name}_recall'] = matched_size[name]/size_totals[name] if size_totals[name] else None
    return (metrics,details) if collect_details else metrics


def _average(rows,key):
    return float(np.mean([x[key] for x in rows]))


def _eligible(results, baselines, max_iou_drop, small_recall_slack):
    # Require constraints for EACH seed, not merely on the pooled average.
    return all((r['pixel_iou'] + 1e-12 >= b['pixel_iou']-max_iou_drop
                and r['small_matched'] + small_recall_slack >= b['small_matched'])
               for r,b in zip(results,baselines))


def _candidate_rank(row):
    # Higher mean F1; then stronger worst-seed F1; then better pixel IoU.
    return (row['mean_instance_f1'],row['min_instance_f1'],row['mean_pixel_iou'])


def select_shared_policy(policies, results, baseline_results,
                         max_iou_drop=.005, small_recall_slack=0):
    """Pure selection helper, suitable for synthetic tests and reproducible reports."""
    if max_iou_drop < 0 or small_recall_slack < 0:
        raise ValueError('Constraints must be nonnegative')
    if not baseline_results:
        raise ValueError('At least one seed required')
    rows = []
    for policy in policies:
        values = results[policy]
        if len(values) != len(baseline_results):
            raise ValueError('Seed count mismatch')
        rows.append({'policy':policy,'mean_instance_f1':_average(values,'instance_f1'),
                     'min_instance_f1':min(r['instance_f1'] for r in values),
                     'mean_pixel_iou':_average(values,'pixel_iou'),
                     'mean_small_recall':_average(values,'small_recall'),
                     'eligible':_eligible(values,baseline_results,max_iou_drop,small_recall_slack)})
    allowed = [r for r in rows if r['eligible']]
    if not allowed:
        raise RuntimeError('No policy satisfies all seed-level constraints')
    return max(allowed,key=_candidate_rank), rows


def _load_v4_selection(run_dir, split_hash, ckpt_hash):
    saved_path = run_dir/'diagnostics_v4'/'validation_selection.json'
    if not saved_path.exists():
        raise FileNotFoundError(f'Please run experiments_v4.sh first: {saved_path}')
    saved = read_json(saved_path)
    if saved.get('split_sha256') != split_hash or saved.get('checkpoint_sha256') != ckpt_hash:
        raise RuntimeError(f'v4 selection does not match current run/split: {saved_path}')
    config = saved['selected']
    return config


def _score_v4(tiles, config, small_max=100, medium_max=1000):
    px,objs=np.zeros(3,np.int64),np.zeros(3,np.int64)
    matched_small = 0
    for tile in tiles:
        mask, instances = make_prediction(tile.probabilities,config['method'],
                    float(config['threshold']),int(config['min_area']),config.get('peak_distance'))
        a,b = counts(mask,tile.truth,instances,tile.gt_polygons)
        px += a; objs += b
        matches = match_objects(instances,tile.gt_polygons)
        matched_small += sum(j in matches and size_bin(g.area,small_max,medium_max)=='small'
                             for j,g in enumerate(tile.gt_polygons))
    return {'pixel_iou':metrics_from_counts(*map(int,px))['iou'],
            'instance_f1':object_metrics(*map(int,objs))['instance_f1'],
            'small_matched':matched_small}


def study(model_names=DEFAULT_MODELS,seeds=DEFAULT_SEEDS,
          batch_size=2,device_name='auto',refresh=False,
          thresholds=DEFAULT_THRESHOLDS,small_floors=DEFAULT_SMALL_FLOORS,
          confidence_cutoffs=DEFAULT_CONFIDENCES,
          confidence_metrics=DEFAULT_CONFIDENCE_METRICS,
          large_area=64,max_iou_drop=.005,small_recall_slack=0,
          small_max=100,medium_max=1000):
    """Choose ONE configuration per model family, shared by all its seeds.

    Checkpoint-specific pre-tuned thresholds are permitted via `saved`.
    The filter logic and all confidence/area parameters are identical across
    the requested seeds. Missing runs cause an error, never silent exclusion.
    """
    if not 0 < small_max < medium_max:
        raise ValueError('Expected 0 < small_max < medium_max')
    policies=candidate_policies(thresholds,small_floors,confidence_cutoffs,
                                 confidence_metrics,large_area)
    all_seed_rows=[]
    for model in model_names:
        if not model.strip() or '/' in model or '\\' in model:
            raise ValueError(f'Invalid model prefix: {model!r}')
        model_dir=OUTDIR/model
        model_dir.mkdir(parents=True,exist_ok=True)
        runs=[]
        for seed in seeds:
            run_dir=OUT/'runs'/f'{model}_seed{seed}'
            run_dir,cfg,tuned,ids=_valid_run(run_dir)
            split_hash=hashlib.sha256(SPLITS.read_bytes()).hexdigest()
            ckpt_hash=sha256_file(run_dir/'best.pt')
            v4_config=_load_v4_selection(run_dir,split_hash,ckpt_hash)
            print(f'Loading validation cache: {model} seed={seed} ({len(ids)} tiles)',flush=True)
            tiles=validation_tiles(run_dir,batch_size,device_name,refresh)
            runs.append({'seed':int(seed),'name':run_dir.name,'dir':run_dir,'tuned':tuned,
                         'tiles':tiles,'split_hash':split_hash,'ckpt_hash':ckpt_hash,
                         'v4_config':v4_config})
        baselines=[]; v4_scores=[]
        for run in runs:
            b=score_policy(run['tiles'],Policy(method='baseline_saved'),run['tuned'],small_max,medium_max)
            v4=_score_v4(run['tiles'],run['v4_config'],small_max,medium_max)
            if b['small_total']==0:
                raise RuntimeError('No small (< small-max) GT buildings in validation split; cannot optimize small-building recall')
            baselines.append(b); v4_scores.append(v4)
        results={}
        for i,p in enumerate(policies,1):
            scores=[]
            for run in runs:
                if p.method == 'baseline_saved':
                    scores.append(baselines[len(scores)])
                else:
                    scores.append(score_policy(run['tiles'],p,run['tuned'],small_max,medium_max))
            results[p]=scores
            if i%10==0 or i==len(policies):
                print(f'  {model}: evaluated {i}/{len(policies)} shared policies',flush=True)
        selected, candidate_rows = select_shared_policy(policies,results,baselines,
                                                     max_iou_drop,small_recall_slack)
        chosen=selected['policy']
        grid=[]
        for row in candidate_rows:
            p=row['policy']
            grid.append({'policy_id':policies.index(p),**policy_dict(p),
                         **{k:v for k,v in row.items() if k!='policy'},
                         'per_seed_f1':';'.join(f'{x["instance_f1"]:.5f}' for x in results[p]),
                         'per_seed_small_matched':';'.join(str(x['small_matched']) for x in results[p])})
        csv_write(model_dir/'validation_shared_policy_grid.csv',grid,list(grid[0]))
        selected_rows=[]
        for idx,run in enumerate(runs):
            b=baselines[idx]; v4=v4_scores[idx]; new=results[chosen][idx]
            out={'model':model,'seed':run['seed'],'baseline_f1':b['instance_f1'],
                 'v4_f1':v4['instance_f1'],'v5_f1':new['instance_f1'],
                 'delta_v5_minus_v4':new['instance_f1']-v4['instance_f1'],
                 'baseline_pixel_iou':b['pixel_iou'],'v4_pixel_iou':v4['pixel_iou'],
                 'v5_pixel_iou':new['pixel_iou'],
                 'baseline_small_matched':b['small_matched'],
                 'v4_small_matched':v4['small_matched'],
                 'v5_small_matched':new['small_matched'],
                 'small_total':b['small_total']}
            selected_rows.append(out)
            all_seed_rows.append(out)
        csv_write(model_dir/'validation_seed_comparison.csv',selected_rows,list(selected_rows[0]))
        provenance={'split':'val','split_sha256':runs[0]['split_hash'],
                    'checkpoints_sha256':{r['name']:r['ckpt_hash'] for r in runs},
                    'tuned_thresholds':{r['name']:r['tuned'] for r in runs},
                    'v4_selection_sha256':{r['name']:sha256_file(r['dir']/'diagnostics_v4'/'validation_selection.json') for r in runs}}
        saved={'selection':'same v5 policy across listed seeds (saved uses each checkpoint validation-tuned threshold)',
               'objective':'maximize mean instance F1, then worst-seed instance F1',
               'constraints':'per seed: small matched >= v3 baseline - slack, pixel IoU >= v3 baseline - max_iou_drop',
               'max_iou_drop':float(max_iou_drop),'small_recall_slack':int(small_recall_slack),
               'small_max':small_max,'medium_max':medium_max,'seeds':list(map(int,seeds)),
               'selected_policy':policy_dict(chosen),'selected_mean_f1':selected['mean_instance_f1'],
               'selected_mean_small_recall':selected['mean_small_recall'],
               'baseline_mean_f1':_average(baselines,'instance_f1'),
               'v4_mean_f1':_average(v4_scores,'instance_f1'),
               'eligible_policies':sum(x['eligible'] for x in candidate_rows),
               'total_policies':len(candidate_rows),**provenance}
        write_json(model_dir/'validation_shared_policy.json',saved)
        _plot_grid(candidate_rows,selected,baselines,v4_scores,model_dir/'validation_shared_tradeoff.png',model)
        _write_building_deltas(model_dir,runs,chosen,small_max,medium_max)
        print(f'{model}: v3={saved["baseline_mean_f1"]:.4f}, '
              f'v4={saved["v4_mean_f1"]:.4f}, '
              f'v5={saved["selected_mean_f1"]:.4f}; '
              f'eligible={saved["eligible_policies"]}/{len(policies)}',flush=True)
        print(f'  selected: {policy_dict(chosen)}',flush=True)
        print(f'  outputs: {model_dir}',flush=True)
    if all_seed_rows:
        csv_write(OUTDIR/'validation_suite_summary.csv',all_seed_rows,list(all_seed_rows[0]))
        print('Saved:',OUTDIR/'validation_suite_summary.csv')


def _plot_grid(rows,selected,baseline,v4,outfile,model):
    plt=_plot()
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for method,mark in [('baseline_saved','o'),('filter','s'),('confidence','.')]:
        group=[r for r in rows if r['policy'].method==method]
        if group:
            axes[0].scatter([r['mean_pixel_iou'] for r in group],
                            [r['mean_instance_f1'] for r in group],
                            s=55 if method!='confidence' else 20,marker=mark,
                            label=method,alpha=.55)
    axes[0].scatter(selected['mean_pixel_iou'],selected['mean_instance_f1'],
                    c='red',s=160,marker='*',label='selected')
    axes[0].set_xlabel('Mean validation pixel IoU')
    axes[0].set_ylabel('Mean validation instance F1')
    axes[0].set_title(f'{model}: candidate policies')
    axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    baseline_small=_average(baseline,'small_recall')
    elig=[r for r in rows if r['eligible']]
    inelig=[r for r in rows if not r['eligible']]
    for group,label,color in [(inelig,'ineligible','gray'),(elig,'eligible','tab:blue')]:
        if group:
            axes[1].scatter([r['mean_small_recall'] for r in group],
                            [r['mean_instance_f1'] for r in group],
                            label=label,c=color,s=27,alpha=.55)
    axes[1].scatter(selected['mean_small_recall'],selected['mean_instance_f1'],
                    c='red',s=160,marker='*',label='selected')
    axes[1].axvline(baseline_small,c='gray',ls='--',label='Mean baseline small recall')
    axes[1].set_xlabel('Mean small-building recall (matched GT / GT objects)')
    axes[1].set_ylabel('Mean validation instance F1')
    axes[1].set_title('Small-building recall constraint (checked per seed)')
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(outfile,dpi=130);plt.close(fig)


def _write_building_deltas(model_dir,runs,chosen,small_max,medium_max):
    records=[]
    for run in runs:
        v4=run['v4_config']
        for tile in run['tiles']:
            base_mask,base_objs=predict_policy(tile.probabilities,Policy('baseline_saved'),run['tuned'])
            v4_mask,v4_objs=make_prediction(tile.probabilities,v4['method'],
                                float(v4['threshold']),int(v4['min_area']),v4.get('peak_distance'))
            v5_mask,v5_objs=predict_policy(tile.probabilities,chosen,run['tuned'])
            base=set(match_objects(base_objs,tile.gt_polygons))
            old=set(match_objects(v4_objs,tile.gt_polygons))
            new=set(match_objects(v5_objs,tile.gt_polygons))
            for j,g in enumerate(tile.gt_polygons):
                size=size_bin(float(g.area),small_max,medium_max)
                records.append({'run':run['name'],'seed':run['seed'],
                                'tile_id':tile.image_id,'gt_index':j,'size_bin':size,
                                'gt_area':round(float(g.area),2),
                                'baseline_matched':int(j in base),
                                'v4_matched':int(j in old),'v5_matched':int(j in new),
                                'v4_to_v5':('recovered' if j in new and j not in old else
                                            'lost' if j in old and j not in new else
                                            'both_matched' if j in old else 'both_missed')})
    if records:
        csv_write(model_dir/'validation_building_changes.csv',records,list(records[0]))
        small=[r for r in records if r['size_bin']=='small']
        recovered=sum(r['v4_to_v5']=='recovered' for r in small)
        lost=sum(r['v4_to_v5']=='lost' for r in small)
        print(f'  small-building v4 -> v5: recovered={recovered}, lost={lost} '
              f'across {len(runs)} seeds (same GT buildings repeated)',flush=True)


def gallery(model,seed=111,count=6,batch_size=2,device_name='auto',refresh=False):
    """Review same small GT building with v3, v4 and shared-v5 pixel masks."""
    if not 1 <= count <= 24:
        raise ValueError('count must be 1..24')
    run_dir=OUT/'runs'/f'{model}_seed{seed}'
    run_dir,_,tuned,_=_valid_run(run_dir)
    dest=OUTDIR/model
    policy_path=dest/'validation_shared_policy.json'
    if not policy_path.exists():
        raise FileNotFoundError(f'Run v5-study first: {policy_path}')
    stored=read_json(policy_path)
    if seed not in stored['seeds']:
        raise ValueError('Seed was not included in shared policy study')
    if stored['split_sha256']!=sha256_file(SPLITS) or stored['checkpoints_sha256'].get(run_dir.name)!=sha256_file(run_dir/'best.pt'):
        raise RuntimeError('v5 report does not match this split/checkpoint')
    policy=Policy(**stored['selected_policy'])
    v4=_load_v4_selection(run_dir,stored['split_sha256'],sha256_file(run_dir/'best.pt'))
    if stored['v4_selection_sha256'][run_dir.name]!=sha256_file(run_dir/'diagnostics_v4'/'validation_selection.json'):
        raise RuntimeError('v4 selection changed since v5 study')
    tiles=validation_tiles(run_dir,batch_size,device_name,refresh)
    items=[]
    for tile in tiles:
        bm,bo=predict_policy(tile.probabilities,Policy('baseline_saved'),tuned)
        vm,vo=make_prediction(tile.probabilities,v4['method'],v4['threshold'],v4['min_area'],v4.get('peak_distance'))
        nm,no=predict_policy(tile.probabilities,policy,tuned)
        bb=set(match_objects(bo,tile.gt_polygons))
        vb=set(match_objects(vo,tile.gt_polygons))
        nb=set(match_objects(no,tile.gt_polygons))
        for i,g in enumerate(tile.gt_polygons):
            if size_bin(g.area,stored['small_max'],stored['medium_max'])!='small':
                continue
            priority=(0 if i in nb and i not in vb else
                      1 if i not in vb and i in bb else
                      2 if i in vb and i not in nb else
                      3 if i not in vb and i not in nb else 4)
            items.append((priority,tile,i,g,(bm,vm,nm),(i in bb,i in vb,i in nb)))
    items.sort(key=lambda x:(x[0],x[1].image_id,x[2]))
    show=items[:count]
    if not show:
        print('No small GT buildings found for gallery')
        return None
    from rasterio.features import rasterize
    plt=_plot()
    fig,axes=plt.subplots(len(show),3,figsize=(11,3.5*len(show)),squeeze=False)
    image_cache={}
    for r,(_,tile,j,g,masks,hits) in enumerate(show):
        if tile.image_id not in image_cache:
            image_cache[tile.image_id]=load_tile(tile.image_id)[0]
        image=image_cache[tile.image_id]
        h,w=tile.truth.shape
        cx=(g.bounds[0]+g.bounds[2])/2;cy=(g.bounds[1]+g.bounds[3])/2
        half=max(48,(g.bounds[2]-g.bounds[0])/2+18,(g.bounds[3]-g.bounds[1])/2+18)
        x0,x1=max(0,int(cx-half)),min(w,int(cx+half))
        y0,y1=max(0,int(cy-half)),min(h,int(cy+half))
        target=rasterize([(g,1)],out_shape=(h,w),dtype=np.uint8)
        boundary=cv2.morphologyEx(target[y0:y1,x0:x1],cv2.MORPH_GRADIENT,
                                  np.ones((3,3),np.uint8))>0
        gt=tile.truth[y0:y1,x0:x1]
        for c,(mask,title,hit) in enumerate(zip(masks,('v3 raw','v4 selected','v5 shared'),hits)):
            view=image[y0:y1,x0:x1].astype(np.float32).copy()
            pred=mask[y0:y1,x0:x1]
            tp = pred & gt
            fp = pred & ~gt
            fn = ~pred & gt
            view[tp]=.65*view[tp]+.35*np.array([0,255,0])
            view[fp]=.60*view[fp]+.40*np.array([255,0,0])
            view[fn]=.60*view[fn]+.40*np.array([0,70,255])
            view[boundary]=[255,240,0]
            ax=axes[r,c]; ax.imshow(np.uint8(np.clip(view,0,255)))
            ax.set_title(f'{tile.image_id[-13:]} | GT {j} | {title}: {"MATCH" if hit else "MISS"}',fontsize=8)
            ax.axis('off')
    fig.suptitle(f'{run_dir.name}: small GT buildings | yellow target boundary | green TP, red FP, blue FN',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98))
    path=dest/f'validation_small_gallery_seed{seed}.png'
    fig.savefig(path,dpi=140);plt.close(fig)
    print('Saved:',path)
    return path
