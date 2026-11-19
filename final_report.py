#!/usr/bin/env python3
"""Generate a reproducible SpaceNet8 Germany final report; no training.

Works inside the existing v6 project. Reads published local run artifacts and
uses original prepared masks to compute true TN. Never retunes thresholds or
performs model selection. With --summary-only, generates a clearly labeled
preliminary report using verified previously supplied aggregate metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                   TableStyle, Image as RLImage, PageBreak, KeepTogether)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import A4
except ImportError as exc:
    raise SystemExit('Missing reportlab: python -m pip install reportlab') from exc

RAW = 'v6_raw_bce_dice_balanced_e16_seed333'
PHOTO = 'v6_augmentation_photometric_balanced_e16_seed333'
OLD_RAW = 'raw_bce_dice_seed111'
CLAHE = 'clahe_seed111'
OLD_BCE = 'matched_bce_seed111'
COMBINED = 'augmentation_seed111'

DESCRIPTIONS = {
    RAW: 'Raw RGB, v6 (seed 333)',
    PHOTO: 'Photometric augmentation, v6 (seed 333)',
    OLD_RAW: 'Raw RGB, original study (seed 111)',
    CLAHE: 'CLAHE, original study (seed 111)',
    OLD_BCE: 'BCE only, original study (seed 111)',
    COMBINED: 'Combined augmentation, original study (seed 111)',
}
# Verified from user-shared output. Used ONLY for summary-only mode.
KNOWN_TEST = {
    RAW: dict(iou=.5229,dice=.6867,precision=.5869,recall=.8274),
    PHOTO: dict(iou=.5114,dice=.6767,precision=.5708,recall=.8309),
    OLD_RAW: dict(iou=.5134,dice=.6785,precision=.5675,recall=.8434),
    CLAHE: dict(iou=.5187,dice=.6831,precision=.5832,recall=.8242),
    OLD_BCE: dict(iou=.4998,dice=.6665,precision=.5540,recall=.8363),
    COMBINED: dict(iou=.4434,dice=.6144,precision=.4907,recall=.8213),
}
COL = {'raw':'#197b76', 'clahe':'#e29e44', 'photo':'#9055a2',
       'muted':'#64748b', 'ink':'#172a42', 'red':'#cc4f58', 'blue':'#3765af'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records_csv(path):
    if not Path(path).is_file():
        return []
    with open(path, newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, keys):
    with open(path, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def metrics_from_confusion(cm):
    # rows GT background, GT building; cols predicted background, predicted building.
    tn, fp, fn, tp = [int(x) for x in (cm[0,0], cm[0,1], cm[1,0], cm[1,1])]
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else float('nan'),
                recall=tp/(tp+fn) if tp+fn else float('nan'),
                accuracy=(tp+tn)/(tp+tn+fp+fn))


def read_test_metrics(base, run_id, summary_only=False):
    path=base/'runs'/run_id/'test_metrics.json'
    if path.exists():
        m=read_json(path)
        for key in ['iou','dice','precision','recall']:
            if key not in m:
                raise RuntimeError(f'{path}: missing {key}')
        return m
    if summary_only and run_id in KNOWN_TEST:
        return {'source':'previously supplied test console output', **KNOWN_TEST[run_id]}
    raise FileNotFoundError(f'{path} missing; run python exp.py finalize --run {base/"runs"/run_id}')


def validate_split(splits_path, run_path):
    if not splits_path.exists():
        return
    ckpt_config=run_path/'config.json'
    if ckpt_config.exists():
        cfg=read_json(ckpt_config)
        expected=cfg.get('split_sha256')
        if expected:
            actual=hashlib.sha256(splits_path.read_bytes()).hexdigest()
            if actual!=expected:
                raise RuntimeError(f'{run_path}: frozen split hash changed')


def confusion_exact(base, split_ids, run_id, metrics):
    """Compute exact 2x2 matrix from finalizer's saved TP, FP, FN and true mask size.

    This is more faithful than rethresholding float16 cached probabilities:
    float16 can quantize scores near threshold, slightly changing pixel counts.
    """
    missing=[k for k in ('tp','fp','fn') if k not in metrics]
    if missing:
        raise ValueError(f'{run_id} test_metrics lacks counts: {missing}')
    total_pixels=gt_positive=0
    for image_id in split_ids:
        npz=base/'dataset'/f'{image_id}.npz'
        if not npz.exists():
            raise FileNotFoundError(npz)
        with np.load(npz) as d:
            gt=d['mask']
            total_pixels += gt.size
            gt_positive += int((gt>0).sum())
    tp,fp,fn = (int(metrics['tp']),int(metrics['fp']),int(metrics['fn']))
    tn=total_pixels-tp-fp-fn
    if tn<0 or tp+fn!=gt_positive:
        raise RuntimeError(f'{run_id}: saved counts do not match frozen test masks: '+
                           f'GT positives={gt_positive}, TP+FN={tp+fn}, total={total_pixels}')
    return np.array([[tn,fp],[fn,tp]],dtype=np.int64)


def preproc_clahe(image):
    """Same image enhancement as experiments.engine.preprocess(...,'clahe')."""
    if image.dtype != np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('CLAHE requires uint8 HWC RGB, exactly as in training')
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2RGB)


def tile(base, image_id):
    with np.load(base/'dataset'/f'{image_id}.npz') as d:
        return d['image'].copy(), d['mask'].astype(bool).copy()


def pred(base, run_id, image_id, shape):
    r=base/'runs'/run_id
    prob_path=r/'test_predictions'/f'{image_id}.npy'
    if not prob_path.exists():
        raise FileNotFoundError('Cached prediction missing: '+str(prob_path)+
                                '. Run python exp.py finalize --run '+str(r))
    prob=np.load(prob_path).astype(np.float32)
    if prob.shape!=shape:
        raise ValueError(f'{prob_path}: expected {shape}, got {prob.shape}')
    th_path=r/'threshold.json'
    if not th_path.exists():
        raise FileNotFoundError(th_path)
    threshold=float(read_json(th_path)['threshold'])
    return prob>=threshold


def viz_overlay(rgb, gt, predicted):
    out=rgb.astype(np.float32).copy()
    for m,c in ((gt&predicted,[22,184,99]),(~gt&predicted,[245,70,84]),(gt&~predicted,[35,107,239])):
        out[m]=.52*out[m]+.48*np.array(c,dtype=np.float32)
    return np.clip(out,0,255).astype(np.uint8)


def plt_base():
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'figure.facecolor':'white','axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})


def chart_test(metrics, out):
    ordered=[RAW,CLAHE,OLD_RAW,PHOTO,OLD_BCE,COMBINED]
    labels=['Raw v6','CLAHE v1','Raw v1','Photo v6','BCE only','Combined aug.']
    palette=[COL['raw'],COL['clahe'],COL['muted'],COL['photo'],'#8494a7','#c26f77']
    x=np.arange(len(ordered))
    fig,axs=plt.subplots(1,2,figsize=(12.6,4.1),gridspec_kw={'wspace':.25})
    for ax,key in zip(axs,['iou','dice']):
        vals=[metrics[k][key] for k in ordered]
        bars=ax.barh(x,vals,color=palette,height=.63)
        ax.invert_yaxis(); ax.set_yticks(x,labels)
        ax.set_xlim(0,max(vals)+.14); ax.set_xlabel(key.upper()+' (test)')
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        for b,v in zip(bars,vals): ax.text(v+.006,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=9)
    fig.suptitle('Fixed-split test comparison | separate training cohorts are labeled',fontweight='bold',y=1.02)
    fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig)


def chart_confusion(matrices,out):
    keys=[k for k in [RAW,CLAHE,PHOTO] if k in matrices]
    if not keys:return None
    fig,axes=plt.subplots(1,len(keys),figsize=(4.7*len(keys),4.3),squeeze=False)
    for ax,key in zip(axes[0],keys):
        cm=matrices[key].astype(float)
        fractions=cm / np.maximum(cm.sum(axis=1,keepdims=True),1)
        pal=LinearSegmentedColormap.from_list('cm',['#eff6f9','#cee9e5','#19867a'])
        ax.imshow(fractions,vmin=0,vmax=1,cmap=pal)
        for row in range(2):
            for col in range(2):
                v=fractions[row,col]
                ax.text(col,row,f'{int(cm[row,col]):,}\n({v*100:.1f}%)',ha='center',va='center',
                        fontsize=9,color=('white' if v>.65 else '#14263b'),fontweight='bold')
        ax.set_yticks([0,1],['Background','Building']);ax.set_xticks([0,1],['Background','Building'])
        ax.set_xlabel('Prediction'); ax.set_ylabel('Ground truth')
        ax.set_title({'v6_raw_bce_dice_balanced_e16_seed333':'Raw v6',
                      'clahe_seed111':'CLAHE v1',
                      'v6_augmentation_photometric_balanced_e16_seed333':'Photo v6'}[key])
    fig.suptitle('Exact test confusion matrices | cell counts and GT-row percentages',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.91]); fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig)
    return out


def chart_before_after(image,out):
    after=preproc_clahe(image)
    orig_l=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)[:,:,0]
    proc_l=cv2.cvtColor(after,cv2.COLOR_RGB2LAB)[:,:,0]
    fig,axs=plt.subplots(2,2,figsize=(10.6,8.0),gridspec_kw={'height_ratios':[3,1]})
    axs[0,0].imshow(image);axs[0,0].set_title('Before | Original RGB (no enhancement)')
    axs[0,1].imshow(after);axs[0,1].set_title('After | CLAHE on LAB luminance')
    for ax in axs[0]:ax.axis('off')
    axs[1,0].hist(orig_l.ravel(),bins=48,range=(0,256),color=COL['raw'],alpha=.9)
    axs[1,1].hist(proc_l.ravel(),bins=48,range=(0,256),color=COL['clahe'],alpha=.9)
    for ax in axs[1]: ax.set_xlabel('Luminance (L channel, 8-bit)');ax.set_ylabel('Pixel count');ax.grid(alpha=.13)
    fig.tight_layout();fig.savefig(out,dpi=155,bbox_inches='tight');plt.close(fig)
    return {'raw_luminance_std':float(orig_l.std()),'clahe_luminance_std':float(proc_l.std())}


def chart_predictions(base, ids, out, a, b, titles, count=3):
    records=[]
    for image_id in ids:
        im,gt=tile(base,image_id)
        records.append((int(gt.sum()),image_id))
    selected=[t[1] for t in sorted(records)]
    picks=np.linspace(0,len(selected)-1,min(count,len(selected)),dtype=int)
    selected=list(dict.fromkeys(selected[int(i)] for i in picks))
    fig,ax=plt.subplots(len(selected),4,figsize=(15.0,max(4,3.4*len(selected))),squeeze=False)
    for row,image_id in enumerate(selected):
        im,gt=tile(base,image_id)
        pa=pred(base,a,image_id,gt.shape)
        pb=pred(base,b,image_id,gt.shape)
        images=[im,gt.astype(np.uint8),viz_overlay(im,gt,pa),viz_overlay(im,gt,pb)]
        labels=['Original RGB','Reference mask',titles[0],titles[1]]
        for j in range(4):
            ax[row,j].imshow(images[j],cmap='gray' if j==1 else None,vmin=0 if j==1 else None,vmax=1 if j==1 else None)
            ax[row,j].set_title(labels[j],fontsize=10);ax[row,j].axis('off')
        ax[row,0].set_ylabel(image_id.split('_')[-3]+'_'+image_id.split('_')[-2]+'_'+image_id.split('_')[-1],fontsize=10)
    fig.suptitle('Pixel error overlays | green TP, red FP, blue FN',fontweight='bold',y=.997)
    fig.tight_layout(rect=[0,0,1,.975]);fig.savefig(out,dpi=140,bbox_inches='tight');plt.close(fig)
    return selected


def chart_validation_summary(rows,out):
    rows=[x for x in rows if x.get('sampling')=='balanced' and x.get('objective')=='bce_dice']
    if not rows:return None
    labels=[]; vals=[];errs=[]; cs=[]
    for r in rows:
        labels.append(('Photo' if r['model']=='augmentation_photometric' else 'Raw')+' / '+r['epochs']+' ep')
        vals.append(float(r['mean_val_iou']));errs.append(float(r['std_val_iou']))
        cs.append(COL['photo'] if r['model']=='augmentation_photometric' else COL['raw'])
    fig,ax=plt.subplots(figsize=(9.4,3.4))
    ax.barh(labels,vals,xerr=errs,height=.57,color=cs,ecolor='#26354a',capsize=4)
    ax.invert_yaxis();ax.set_xlim(0,.69);ax.set_xlabel('Mean validation IoU (3 seeds; error bars = SD)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for y,v in enumerate(vals):ax.text(v+.018,y,f'{v:.4f}',va='center')
    fig.tight_layout();fig.savefig(out,dpi=160,bbox_inches='tight');plt.close(fig)
    return out


def make_report(out, metrics, matrices, charts, val_rows, metadata, summary_only):
    dest=out/'SpaceNet8_Germany_Final_Report.pdf'
    W,H=A4
    navy=colors.HexColor('#172a42');muted=colors.HexColor('#64748b')
    teal=colors.HexColor(COL['raw']);gold=colors.HexColor(COL['clahe'])
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitleCustom',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=navy,spaceAfter=12))
    styles.add(ParagraphStyle(name='SubtitleCustom',fontName='Helvetica',fontSize=11.5,leading=17,textColor=muted,spaceAfter=15))
    styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=14,leading=18,textColor=navy,spaceBefore=10,spaceAfter=8))
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.4,leading=14.5,textColor=navy,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.1,leading=12,textColor=muted,spaceAfter=7))
    styles.add(ParagraphStyle(name='CaptionCustom',fontName='Helvetica',fontSize=8.2,leading=12,textColor=muted,spaceAfter=8))
    body=lambda s:Paragraph(s,styles['BodyCustom'])
    small=lambda s:Paragraph(s,styles['SmallCustom'])
    sect=lambda s:Paragraph(s,styles['SectionCustom'])
    story=[]
    def img(path,width=510):
        from PIL import Image as PILImage
        with PILImage.open(path) as image:
            w,h=image.size
        return RLImage(str(path),width=width,height=width*h/w,kind='proportional')
    def table(rows,widths=None,first_header=True):
        nested=[]
        for i,row in enumerate(rows):
            nested.append([Paragraph(str(x),styles['SmallCustom'] if i else styles['BodyCustom']) for x in row])
        t=Table(nested,colWidths=widths,repeatRows=1 if first_header else 0,hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f2f4')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
            ('LINEBELOW',(0,0),(-1,0),.9,teal),
            ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('VALIGN',(0,0),(-1,-1),'TOP')]))
        return t
    def footer(canvas,doc):
        canvas.saveState();canvas.setFillColor(muted);canvas.setFont('Helvetica',8)
        canvas.drawString(42,24,'SpaceNet 8 | Germany_Training_Public | exploratory study')
        canvas.drawRightString(W-42,24,f'Page {doc.page}')
        canvas.restoreState()

    story += [Spacer(1,20),Paragraph('Building Footprint Segmentation',styles['CoverTitleCustom']),
              Paragraph('SpaceNet 8 - Germany_Training_Public | Final experimental report',styles['SubtitleCustom']),
              body('<b>Study objective.</b> Evaluate whether deterministic image enhancement and training-time photometric augmentation improve building footprints, using a ResNet-34 U-Net and a common test metric.'),
              sect('Executive summary')]
    raw=metrics[RAW];photo=metrics[PHOTO];clahe=metrics[CLAHE];old=metrics[OLD_RAW]
    story += [table([
        ['Key outcome','Value / interpretation'],
        ['Best observed test run',f'Raw RGB v6 seed333: IoU {raw["iou"]:.4f}, Dice {raw["dice"]:.4f}'],
        ['CLAHE vs matching raw v1',f'{clahe["iou"]:.4f} vs {old["iou"]:.4f}: {clahe["iou"]-old["iou"]:+.4f} IoU'],
        ['Photo augmentation vs matching raw v6',f'{photo["iou"]:.4f} vs {raw["iou"]:.4f}: {photo["iou"]-raw["iou"]:+.4f} IoU'],
        ['Training duration','60 epochs did not materially improve the 3-seed validation result'],
        ['Task scope','Pre-event RGB building masks only; no roads or flood classification'],
    ],[192,317]),Spacer(1,14),
    body('<b>Primary conclusion:</b> Image processing was genuinely investigated. CLAHE enhanced image luminance and showed a small positive result in the original matched seed-111 test. Photometric augmentation was used during training, not as a fixed inference-time transform. Neither showed a robust final test improvement over the strongest raw-RGB run.'),
    small('Methodology caveat: the test split has been accessed repeatedly during this exploratory project, and adjacent tiles exist across split boundaries. These results are not geographically independent and are not directly comparable to the official SpaceNet leaderboard.'),
    small('Metric convention: IoU and Dice are global binary foreground pixel scores over complete, reconstructed tiles. The model checkpoint and probability threshold were chosen on validation data.'),
    PageBreak()]

    story += [sect('1. What image processing was performed?'),
    body('<b>Deterministic preprocessing:</b> CLAHE enhances local contrast by modifying the L channel in LAB color space (clip limit 2.0, 8x8 tile grid). Experiments also covered gamma correction, sharpening, bilateral filtering and processing chains. These change the model input images.'),
    body('<b>Training-time processing:</b> photometric augmentation applies random brightness/contrast changes during model training. The final photometric model still receives raw RGB at inference. Separate geometric flips and rotations were also studied and generally reduced validation IoU.'),
    body('<b>Core pipeline:</b> TIFF RGB + WKT building polygons -&gt; corrected binary raster masks -&gt; full-tile training/validation/test split -&gt; 512x512 patches -&gt; ResNet-34 U-Net -&gt; overlap-averaged full-tile prediction -&gt; sigmoid + validation-selected threshold.'),
    small('The original mask-preparation script had a polygon indentation error that omitted most buildings. Ground truth was rebuilt correctly before the controlled experiments.'),
    sect('2. Before and after CLAHE')]
    if 'before_after' in charts:
        story += [img(charts['before_after'],480),small('A real test tile. Left: original raw RGB; right: the same tile after CLAHE. Histograms show luminance redistribution. This is a visual transformation, not an accuracy measurement.')]
        d=metadata.get('luminance',{})
        if d:story += [small(f'Luminance standard deviation on the illustrated tile: raw {d["raw_luminance_std"]:.1f}, CLAHE {d["clahe_luminance_std"]:.1f} (higher does not by itself mean better segmentation).')]
    else:
        story += [small('Raw Germany TIFF pixels were not uploaded. The exact paired before/after CLAHE image is generated locally when the full report script is run.')]
    story += [PageBreak(),sect('3. Test performance: without vs with processing')]
    if 'test_bars' in charts:story += [img(charts['test_bars'],490),Spacer(1,7)]
    rows=[['Run','IoU','Dice','Precision','Recall']]
    for run in [RAW,CLAHE,OLD_RAW,PHOTO,OLD_BCE,COMBINED]:
        m=metrics[run]
        rows.append([DESCRIPTIONS[run],*(f'{m[k]:.4f}' for k in ['iou','dice','precision','recall'])])
    story += [table(rows,[235,67,67,67,73]),Spacer(1,10),
    body(f'<b>Controlled comparison A - deterministic CLAHE:</b> raw v1 seed111 IoU {old["iou"]:.4f} versus CLAHE seed111 IoU {clahe["iou"]:.4f}, a {clahe["iou"]-old["iou"]:+.4f} absolute difference ({100*(clahe["iou"]-old["iou"]):+.2f} percentage points). Across three validation seeds, CLAHE averaged 0.5698 versus 0.5735 for raw BCE + Dice, so this advantage was not consistently reproduced.'),
    body(f'<b>Controlled comparison B - training photometric augmentation:</b> raw v6 seed333 IoU {raw["iou"]:.4f} versus photometric v6 seed333 IoU {photo["iou"]:.4f}, a {photo["iou"]-raw["iou"]:+.4f} absolute difference. Photometric augmentation was slightly better on mean v6 validation IoU at 16 epochs, but worse in this matched test comparison.'),
    small('Cross-cohort run rankings are descriptive only: the CLAHE experiment and v6 experiments used different experimental cohorts. The paired comparisons above are the defensible before/after contrasts.'),
    PageBreak(),sect('4. Confusion matrices and error interpretation')]
    if 'confusion' in charts:
        story += [img(charts['confusion'],505),small('Exact confusion matrices use stored integer TP/FP/FN from full-tile test inference and ground-truth mask sizes to obtain TN. Rows are true classes; columns are predictions. Cell percentages are normalized within each true class.')]
        cr=metrics_from_confusion(matrices[RAW]);cp=metrics_from_confusion(matrices[PHOTO])
        story += [small(f'Raw v6: TP {cr["tp"]:,}, FP {cr["fp"]:,}, FN {cr["fn"]:,}, TN {cr["tn"]:,}. '+
                        f'Photometric v6: TP {cp["tp"]:,}, FP {cp["fp"]:,}, FN {cp["fn"]:,}, TN {cp["tn"]:,}.')]
    else:
        story += [body('<b>Exact 2x2 confusion matrices are pending local execution.</b> Aggregate IoU, Dice, precision and recall do not uniquely determine all four integer counts, especially true negatives. The finalizer will calculate the matrix from the original binary masks and saved test counts; it will not fabricate TN.'),
                  table([['Selected raw v6 error rate','Value'],
                        ['Building sensitivity (TP / GT building pixels)',f'{raw["recall"]*100:.2f}%'],
                        ['Missed building-pixel rate (FN / GT building pixels)',f'{(1-raw["recall"])*100:.2f}%'],
                        ['Predicted building precision (TP / predicted building pixels)',f'{raw["precision"]*100:.2f}%'],
                        ['False discovery rate (FP / predicted building pixels)',f'{(1-raw["precision"])*100:.2f}%']],
                       [290,218])]
    story += [body('<b>Interpretation.</b> Recall exceeds precision: the model recovers many annotated building pixels but produces substantial extra building area. TN is usually large because background dominates satellite imagery; pixel accuracy should never be used alone to assess building segmentation.'),
    PageBreak(),sect('5. Representative test prediction overlays')]
    if 'pred_raw_clahe' in charts:
        story += [img(charts['pred_raw_clahe'],505),small('Raw v1 vs CLAHE v1 on matched test tiles. Green = correct building pixels; red = predicted buildings absent from reference; blue = missed building pixels.')]
    if 'pred_raw_photo' in charts:
        story += [Spacer(1,5),sect('Raw RGB versus photometric training'),img(charts['pred_raw_photo'],505),
                  small('The input image itself is raw RGB for both v6 models. Photometric perturbations were applied only during training.')]
    if 'pred_raw_clahe' not in charts:
        story += [body('Representative test overlays are generated by the local finalizer from cached probability maps. No test tiles were supplied to this report-building environment.'),
                  small('This avoids substituting unrelated satellite imagery for actual SpaceNet Germany examples.')]
    story += [PageBreak(),sect('6. Three-seed validation, training duration, and object detection')]
    if 'validation_bars' in charts:story += [img(charts['validation_bars'],465)]
    if val_rows:
        rows=[['Model','Epochs','Mean IoU','Mean F1 (instances)','Small recall']]
        for r in sorted(val_rows,key=lambda r:(r['model'],int(r['epochs']))):
            if r.get('sampling')!='balanced' or r.get('objective')!='bce_dice':continue
            rows.append([('Photo' if r['model']=='augmentation_photometric' else 'Raw'),r['epochs'],
                         f'{float(r["mean_val_iou"]):.4f}',f'{float(r["mean_instance_f1"]):.4f}',
                         f'{100*float(r["mean_small_recall"]):.1f}%'])
        story += [table(rows,[110,65,100,120,114])]
    story += [Spacer(1,8),
       body('The optimal checkpoints occurred early. Increasing the maximum budget to 60 epochs did not meaningfully improve validation IoU, and could reduce building-instance F1. The 16-epoch budget was adequate for the best saved checkpoints.'),
       body('Instance-level analysis exposed weak small-building recall. A fixed 64-pixel component filter increased validation instance F1 but removed some genuine small buildings. Confidence-aware filtering preserved the baseline small-building matches but yielded a smaller F1 gain. None of these postprocessing configurations is included in the v6 test metrics above.'),
       body('Image processing conclusion: CLAHE is a visually demonstrable enhancement, but the final highest observed test IoU came from unenhanced raw RGB. This is a meaningful negative result: preprocessing is not automatically beneficial, and image enhancement must be evaluated against the task metric.'),
       sect('7. Limitations and reproducibility'),
       small('Dataset: SpaceNet 8 Germany_Training_Public, pre-event RGB only. Mask reference: Germany_Training_Public_reference.csv, building Wkt_Pix. Five to six iterations of ablation and postprocessing, single-region split.'),
       small('Architecture: segmentation_models_pytorch U-Net with ImageNet-pretrained ResNet-34 encoder; BCE + soft Dice; Adam; 512px patches; raw inference normalization to ImageNet mean and standard deviation.'),
       small('Limitations: geographic adjacency across train/validation/test boundaries, prior test-set inspection, a limited set of seeds for test comparisons, and no same-protocol official SpaceNet baseline reproduction. This report makes no leaderboard claim.')]
    if metadata.get('split_sha256'):
        story += [small('Frozen split SHA-256: '+metadata['split_sha256'])]
    if summary_only:
        story += [small('Summary-only PDF: quantitative test metrics are sourced from the user-supplied console results and uploaded validation CSVs. Exact confusion matrices and before/after TIFF imagery require local source tiles.')]

    doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=45,
                          title='SpaceNet 8 Germany - Building Footprint Segmentation: Final Report',
                          author='Experimental results synthesis')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return dest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('.'))
    ap.add_argument('--output',type=Path,default=Path('experiments_output/final_report'))
    ap.add_argument('--summary-only',action='store_true',help='Use previously reported aggregate scores; no fake confusion matrix')
    args=ap.parse_args()
    project=args.project.resolve()
    base=project/'experiments_output'
    dest=args.output if args.output.is_absolute() else project/args.output
    dest.mkdir(parents=True,exist_ok=True)
    plt_base()
    metrics={k:read_test_metrics(base,k,args.summary_only) for k in KNOWN_TEST}
    if args.summary_only:
        ref=Path(__file__).resolve().parent/'reference_data'
        vr=records_csv(ref/'validation_summary.csv')
    else:
        vr=records_csv(base/'diagnostics_v6'/'validation_summary.csv')
        if not vr:
            print('WARNING: v6 validation summary not present, only test statistics will be included',file=sys.stderr)
    matrices={};charts={};meta={}
    chart_test(metrics,dest/'test_metric_comparison.png');charts['test_bars']=dest/'test_metric_comparison.png'
    chart_validation_summary(vr,dest/'v6_validation_comparison.png')
    if (dest/'v6_validation_comparison.png').exists():charts['validation_bars']=dest/'v6_validation_comparison.png'
    if not args.summary_only:
        splits_path=base/'splits.json'
        splits=read_json(splits_path)
        test_ids=splits['test']
        if not test_ids:raise RuntimeError('Frozen test split contains no tiles')
        meta['split_sha256']=hashlib.sha256(splits_path.read_bytes()).hexdigest()
        for key in [RAW,CLAHE,PHOTO,OLD_RAW]:
            validate_split(splits_path,base/'runs'/key)
            matrices[key]=confusion_exact(base,test_ids,key,metrics[key])
            cm_metrics=metrics_from_confusion(matrices[key])
            # Running metrics are computed from original model float32 probabilities;
            # caches are float16 and are for visuals only, never for official counts.
            for score in ('iou','dice','precision','recall'):
                if abs(cm_metrics[score]-float(metrics[key][score]))>5e-5:
                    raise RuntimeError(f'{key}: confusion counts disagree with saved {score}')
        chart_confusion(matrices,dest/'test_confusion_matrices.png')
        charts['confusion']=dest/'test_confusion_matrices.png'
        confusion_rows=[]
        for key,cm in matrices.items():
            scores=metrics_from_confusion(cm)
            confusion_rows.append({'run':key,**scores})
        write_csv(dest/'test_confusion_counts.csv',confusion_rows,list(confusion_rows[0]))
        # Ground-truth-defined median building coverage, not test-performance-based selection.
        covers=[]
        for image_id in test_ids:
            im,gt=tile(base,image_id)
            covers.append((int(gt.sum()),image_id))
        candidates=[x[1] for x in sorted(covers) if x[0]>0]
        image_id=candidates[len(candidates)//2] if candidates else test_ids[0]
        im,gt=tile(base,image_id)
        meta['example_tile_id']=image_id
        meta['luminance']=chart_before_after(im,dest/'rgb_before_after_clahe.png')
        charts['before_after']=dest/'rgb_before_after_clahe.png'
        selected1=chart_predictions(base,test_ids,dest/'raw_vs_clahe_overlays.png',OLD_RAW,CLAHE,
                                    ('Raw RGB v1','CLAHE v1'))
        selected2=chart_predictions(base,test_ids,dest/'raw_vs_photometric_overlays.png',RAW,PHOTO,
                                    ('Raw RGB v6','Photometric v6'))
        charts['pred_raw_clahe']=dest/'raw_vs_clahe_overlays.png'
        charts['pred_raw_photo']=dest/'raw_vs_photometric_overlays.png'
        meta['comparison_tile_ids']={'raw_vs_clahe':selected1,'raw_vs_photo':selected2}
    pdf=make_report(dest,metrics,matrices,charts,vr,meta,args.summary_only)
    write_csv(dest/'test_performance.csv',
              [{'run':name,'description':DESCRIPTIONS[name],**{k:metrics[name][k] for k in ['iou','dice','precision','recall']}} for name in KNOWN_TEST],
              ['run','description','iou','dice','precision','recall'])
    (dest/'report_provenance.json').write_text(json.dumps({
        'mode':'summary_only' if args.summary_only else 'full_from_local_test_artifacts',
        'split_sha256':meta.get('split_sha256'),
        'runs':list(metrics),
        'comparison_cohorts':{
            'deterministic_preprocessing':[OLD_RAW,CLAHE],
            'training_photometric_augmentation':[RAW,PHOTO]},
        'metrics':'original stored float32 full-tile results',
        'confusion':'exact integer TP/FP/FN plus TN from frozen test mask sizes' if not args.summary_only else 'unavailable in aggregate data',
        'selected_tiles':meta.get('comparison_tile_ids',{})
    },indent=2)+'\n')
    print('Final PDF:',pdf)
    print('Output directory:',dest)
    if args.summary_only:
        print('NOTE: Run without --summary-only inside the real project for exact confusion matrices and before/after imagery.')
    else:
        print('Exact confusion matrix CSV:',dest/'test_confusion_counts.csv')
        print('No training and no new test-time tuning were performed.')


if __name__=='__main__':
    main()
