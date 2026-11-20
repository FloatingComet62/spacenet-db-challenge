#!/usr/bin/env python3
"""Generate V7 SpaceNet LaTeX/PDF report from completed experiment outputs. No training.

Usage (run from repository root):
  python generate_v7_report.py
  python generate_v7_report.py --no-pdf
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).resolve().parent
LABELS = {'rgb':'Original RGB', 'rgb_clahe':'RGB + CLAHE', 'fusion':'RGB + CLAHE + Canny + unsharp'}
COLORS = {'rgb':'#475569', 'rgb_clahe':'#16807c', 'fusion':'#d49438'}

# Independent, external literature results (NOT V7 measurements).
# Source: Kshirsagar et al., Remote Sensing 18(10), 1529 (2026),
# Tables 7-9, "Pretrained model" versus "Pretrained model + FA" rows.
# These are localization F1 values on geographically held-out xBD subsets,
# not V7 SpaceNet building-footprint IoU or Dice scores.
REFERENCE_STUDY = {
    'authors': 'Shruti Kshirsagar, Bharath Chandra, Unaza Tallal, Rajiv Bagai, and Atri Dutta',
    'title': 'Geographic Bias Analysis and Cross-Domain Generalization in Deep Learning-Based Building Damage Assessment',
    'journal': 'Remote Sensing',
    'year': 2026, 'volume': 18, 'issue': 10, 'article': 1529,
    'doi': '10.3390/rs18101529',
    'url': 'https://doi.org/10.3390/rs18101529',
    'task': 'building localization and damage severity classification',
    'dataset': 'xBD/xView2, geographically held-out disaster regions',
    'reported_overall_f1_gain_percent': 7.1,
    'reported_overall_f1_gain_note': 'Paper-wide reported overall F1 improvement; not the localization-only Table 7-9 ablation, and not directly comparable to V7',
    'localization_f1': [
        {'region':'Texas', 'table':7, 'baseline':0.8236, 'fusion':0.8285},
        {'region':'Ayiti', 'table':8, 'baseline':0.6852, 'fusion':0.7043},
        {'region':'Portugal', 'table':9, 'baseline':0.7049, 'fusion':0.7251},
    ],
}
for _region in REFERENCE_STUDY['localization_f1']:
    _region['gain'] = round(_region['fusion'] - _region['baseline'], 4)



def escape_tex(s):
    replacements = {'\\':r'\textbackslash{}', '&':r'\&', '%':r'\%', '$':r'\$', '#':r'\#',
                    '_':r'\_', '{':r'\{', '}':r'\}', '~':r'\textasciitilde{}', '^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(s))


def load_runs(runs):
    items = []
    for file in sorted(runs.glob('*/seed_*/best_val.json')):
        d = json.loads(file.read_text())
        mode = file.parent.parent.name
        if mode not in LABELS: continue
        items.append({'mode':mode, 'seed':int(file.parent.name.split('_')[-1]),
                      'iou':float(d['iou']), 'dice':float(d['dice']),
                      'precision':float(d['precision']), 'recall':float(d['recall'])})
    if not items: raise RuntimeError(f'No valid */seed_*/best_val.json under {runs}')
    modes = {m:[r for r in items if r['mode']==m] for m in LABELS}
    if any(len(v)==0 for v in modes.values()):
        raise RuntimeError('Expected RGB, RGB+CLAHE, and fusion runs in ' + str(runs))
    summary = []
    for m, rows in modes.items():
        summary.append({'mode':m,'label':LABELS[m], 'n':len(rows),
          'iou':mean(r['iou'] for r in rows),
          'sd':stdev(r['iou'] for r in rows) if len(rows)>1 else 0,
          'dice':mean(r['dice'] for r in rows)})
    base = next(r['iou'] for r in summary if r['mode']=='rgb')
    for r in summary:
        r['delta_pp'] = (r['iou']-base)*100
        r['delta_relative'] = ((r['iou']/base)-1)*100 if base else 0
    return items,summary


def create_charts(items,summary,test,out):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.facecolor':'white'})
    # Means with standard deviations
    fig,ax=plt.subplots(figsize=(8.6,3.55))
    x=np.arange(len(summary)); bars=ax.bar(x,[r['iou'] for r in summary],
             yerr=[r['sd'] for r in summary],capsize=7,width=.56,
             color=[COLORS[r['mode']] for r in summary],edgecolor='none')
    ax.set_ylim(0,max(r['iou']+r['sd'] for r in summary)*1.22)
    ax.set_xticks(x,['Raw RGB','RGB + CLAHE','Feature fusion'])
    ax.set_ylabel('Full-tile validation IoU')
    ax.set_title('Matched three-seed validation comparison',weight='bold',loc='left')
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    for b,r in zip(bars,summary):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f'{r["iou"]:.4f}',ha='center',weight='bold')
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=175,bbox_inches='tight');plt.close(fig)
    # Per-seed results
    fig,ax=plt.subplots(figsize=(8.6,3.25))
    for m in LABELS:
        rows=sorted([r for r in items if r['mode']==m],key=lambda r:r['seed'])
        ax.plot([str(r['seed']) for r in rows],[r['iou'] for r in rows],label=LABELS[m],
                marker='o',linewidth=2,markersize=6,color=COLORS[m])
    ax.set_xlabel('Training seed');ax.set_ylabel('Best validation IoU')
    ax.set_title('Variation across training seeds',weight='bold',loc='left')
    ax.legend(loc='best',fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'seeds.png',dpi=175,bbox_inches='tight');plt.close(fig)
    # Published xBD localization F1 results, shown separately from V7 SpaceNet metrics.
    literature = REFERENCE_STUDY['localization_f1']
    fig, ax = plt.subplots(figsize=(8.6, 3.55))
    x=np.arange(len(literature))
    width=.33
    b1=ax.bar(x-width/2,[r['baseline'] for r in literature],width,
              color='#64748b',label='Published xBD baseline')
    b2=ax.bar(x+width/2,[r['fusion'] for r in literature],width,
              color='#d49438',label='Published xBD + fusion')
    ax.set_xticks(x,[r['region'] for r in literature])
    ax.set_ylim(0,1)
    ax.set_ylabel('Published localization F1 (xBD)')
    ax.set_title('Reference study: fusion effect by unseen region',weight='bold',loc='left')
    ax.grid(axis='y',alpha=.13)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right',fontsize=8)
    for bars in (b1,b2):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.018,
                    f'{bar.get_height():.4f}',fontsize=8,ha='center')
    fig.tight_layout()
    fig.savefig(out/'reference_paper_xbd_f1.png',dpi=175,bbox_inches='tight')
    plt.close(fig)
    # Row-normalized visualization, absolute pixel counts in text
    cm=np.array([[test['tn'],test['fp']],[test['fn'],test['tp']]],dtype=np.int64)
    normalized=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)*100
    fig,ax=plt.subplots(figsize=(6.6,3.65))
    img=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}\n{normalized[i,j]:.2f}% of row',
                    ha='center',va='center',fontsize=10,
                    color='white' if normalized[i,j]>65 else '#1e293b',weight='bold')
    ax.set_xticks([0,1],['Background','Building'])
    ax.set_yticks([0,1],['Background','Building'])
    ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
    ax.set_title('Test confusion matrix (row-normalized colors)',weight='bold',loc='left')
    fig.colorbar(img,ax=ax,label='Percent within actual class',fraction=.04,pad=.03)
    fig.tight_layout();fig.savefig(out/'confusion.png',dpi=175,bbox_inches='tight');plt.close(fig)


def make_workflow_diagram(out):
    """Self-contained vector-like PNG of the actual V7 experiment workflow."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12.8, 3.4))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 3.4)
    ax.axis('off')

    steps = [
        ('01  SOURCE DATA', 'SpaceNet 8 Germany', 'RGB TIFF + WKT'),
        ('02  ANNOTATIONS', 'Rasterize all polygons', 'Binary building masks'),
        ('03  IMAGE FEATURES', 'RGB + CLAHE', 'Canny + unsharp'),
        ('04  SPLIT + TRAIN', '142 / 30 / 30 tiles', '3/4/6-channel U-Net'),
        ('05  EVALUATION', 'Select by val IoU', 'Test + confusion'),
    ]
    colors = ['#f1f5f9','#e7f1f5','#e1f3f0','#fff1dd','#e7ecf8']
    left = .18
    box_w = 2.28
    box_h = 1.85
    gap = .29
    bottom = .92
    for i, (eyebrow, main, sub) in enumerate(steps):
        x = left + i * (box_w + gap)
        rect = FancyBboxPatch((x, bottom), box_w, box_h,
                              boxstyle='round,pad=.04,rounding_size=.16',
                              linewidth=1.15, edgecolor='#d0dae2', facecolor=colors[i])
        ax.add_patch(rect)
        ax.text(x+.15, bottom+box_h-.28, eyebrow, fontsize=8.7,
                weight='bold', color='#24736f', ha='left', va='center')
        ax.text(x+.15, bottom+box_h-.82, main, fontsize=10.0,
                weight='bold', color='#20334a', ha='left', va='center')
        ax.text(x+.15, bottom+box_h-1.24, sub, fontsize=9.1,
                color='#52657a', ha='left', va='center')
        if i < len(steps)-1:
            end = x + box_w + gap - .06
            arrow = FancyArrowPatch((x+box_w+.06, bottom+.91),
                      (end, bottom+.91), arrowstyle='-|>', mutation_scale=11,
                      linewidth=1.7, color='#6f889f')
            ax.add_patch(arrow)
    ax.text(6.4, .36,
            'Shared fixed tile split | three seeds per input mode | overlapping-logit full-tile reconstruction',
            fontsize=9.3, ha='center', color='#495d75')
    fig.savefig(out/'workflow.png', dpi=210, bbox_inches='tight', pad_inches=.14)
    plt.close(fig)


def make_real_preview(repo, test, out):
    """Derive the actual image-processing channels from one evaluated tile, no synthetic imagery."""
    try:
        from experiment_v7.features import auxiliary_channels
        per_tile = test.get('per_tile',[])
        ordered = sorted(per_tile,key=lambda r:r.get('tp',0)+r.get('fn',0),reverse=True)
        for rec in ordered:
            name=rec.get('tile','')
            path=repo/'data'/'building_dataset'/Path(name).name
            if not path.is_file():continue
            with np.load(path,allow_pickle=False) as d:
                rgb=d['image'];gt=d['mask']
            aux=auxiliary_channels(rgb)
            fig,axes=plt.subplots(1,5,figsize=(17,3.6))
            for ax,pic,cmap,label in zip(axes,
               (rgb,aux[:,:,0],aux[:,:,1],aux[:,:,2],gt),
               (None,'gray','gray','gray','gray'),
               ('Original RGB','CLAHE','Canny edges','Unsharp mask','Building ground truth')):
                ax.imshow(pic,cmap=cmap);ax.set_title(label,fontsize=11);ax.axis('off')
            fig.tight_layout();fig.savefig(out/'processing.png',dpi=150,bbox_inches='tight');plt.close(fig)
            return Path(name).stem
    except (ImportError, ValueError, KeyError) as err:
        print(f'Image-processing preview omitted: {err}')
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd(),help='Image processing repository root')
    p.add_argument('--runs',type=Path,default=Path('experiment_v7/runs'))
    p.add_argument('--test',type=Path,default=Path('experiment_v7/runs/rgb_clahe/seed_111/test_evaluation/metrics.json'))
    p.add_argument('--out',type=Path,default=Path('experiment_v7/final_report'))
    p.add_argument('--no-pdf',action='store_true')
    a=p.parse_args();repo=a.repo.resolve()
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    def full(x):return x if x.is_absolute() else (repo/x)
    runs=full(a.runs);testfile=full(a.test);out=full(a.out)
    if not testfile.is_file():p.error('Test metrics not found: '+str(testfile))
    test=json.loads(testfile.read_text())
    for key in ('iou','dice','precision','recall','tp','fp','fn','tn'):
        if key not in test:p.error(f'Missing {key} in {testfile}')
    if int(test['tp'])+int(test['fp'])+int(test['fn'])+int(test['tn']) <= 0:
        p.error('Empty test confusion matrix')
    items,summary=load_runs(runs)
    if any(r['n']!=3 for r in summary):
        print('WARNING: one or more modes have a non-three-seed sample; report shows actual N.')
    out.mkdir(parents=True,exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    create_charts(items,summary,test,fig)
    make_workflow_diagram(fig)
    preview=make_real_preview(repo,test,fig)
    examples=sorted((testfile.parent/'examples').glob('*.png'))
    copied=[]
    for i,source in enumerate(examples[:3],1):
        target=fig/f'example_{i}.png';shutil.copy2(source,target)
        copied.append({'path':f'figures/{target.name}','name':source.stem})
    # Write exact, user-editable figures and statistical tables
    with (out/'validation_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['mode','seed','iou','dice','precision','recall']);w.writeheader();w.writerows(items)
    (out/'metrics_used.json').write_text(json.dumps({'validation_runs':items,'validation_summary':summary,
                                    'test_checkpoint':str(testfile),'test_metrics':{k:test[k] for k in ['iou','dice','precision','recall','tp','fp','fn','tn','loss','tile_count','threshold'] if k in test},
                                    'external_literature':REFERENCE_STUDY,
                                    'dataset_source':{'page':'https://spacenet.ai/sn8-challenge/',
                                       's3':'s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz'}},indent=2))
    env=Environment(loader=FileSystemLoader(str(HERE/'report_v7')),undefined=StrictUndefined,autoescape=False,
           block_start_string='((*',block_end_string='*))',variable_start_string='((',variable_end_string='))')
    env.filters['tex']=escape_tex
    env.filters['numfmt']=lambda x:f'{int(x):,}'
    selected=next(r for r in summary if r['mode']=='rgb_clahe')
    content=env.get_template('report.tex.j2').render(items=items,summary=summary,test=test,
             selected=selected,copied=copied,preview=preview,has_preview=bool(preview),
             total_pixels=sum(int(test[k]) for k in ['tn','fp','fn','tp']),
             result_date='20 November 2026',reference=REFERENCE_STUDY)
    (out/'report.tex').write_text(content)
    if not a.no_pdf:
        latexmk=shutil.which('latexmk')
        if latexmk:
            res=subprocess.run([latexmk,'-pdf','-interaction=nonstopmode','-halt-on-error','report.tex'],
                        cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if res.returncode:
                (out/'latex_error.log').write_text(res.stdout)
                print('PDF compilation failed; inspect '+str(out/'latex_error.log'))
            else:print('PDF: '+str(out/'report.pdf'))
        else:print('latexmk is unavailable; generated LaTeX only. Install latexmk/TeX Live or compile report.tex on Overleaf.')
    print('LaTeX: '+str(out/'report.tex'))
    print('Charts and examples: '+str(fig))
    print('Validation CSV: '+str(out/'validation_per_seed.csv'))
    if not copied:print('NOTE: No prediction overlays found; skipped rather than inventing figures.')
    if not preview:print('NOTE: Could not generate real before/after processing preview.')

if __name__=='__main__':
    main()
