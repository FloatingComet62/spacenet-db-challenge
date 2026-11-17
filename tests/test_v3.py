"""Synthetic v3 tests; never require actual SpaceNet data or pretrained weights."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from experiments.v3 import (threshold_table, match_objects, size_bin, geometry_audit,
                            threshold_curves, building_sizes, annotation_audit,
                            compare_official_validation)
from experiments.engine import apply_augment
from experiments.study import planned_tasks
from experiments.data import prepare
from experiments.common import read_json, SPLITS, load_tile


def test_threshold_sweep_counts_original_pixels():
    probs=np.array([[.1,.4],[.8,.9]])
    gt=np.array([[0,1],[1,0]],dtype=bool)
    rows=threshold_table([(probs,gt)],[.2,.5,.85])
    assert [r['tp'] for r in rows]==[2,1,0]
    assert [r['fp'] for r in rows]==[1,1,1]
    assert [r['fn'] for r in rows]==[0,1,2]
    assert rows[0]['iou']==pytest.approx(2/3)


def test_object_matching_and_size_bin_boundaries():
    gt=[box(0,0,5,5),box(10,10,40,40)]
    preds=[box(0,0,5,5),box(10,10,40,40),box(50,50,60,60)]
    matches=match_objects(preds,gt)
    assert set(matches)=={0,1}
    assert matches[0][1]==1
    assert matches[1][1]==1
    assert size_bin(99)=='small'
    assert size_bin(100)=='medium'
    assert size_bin(1000)=='large'


def test_new_geometric_modes_synchronize_mask():
    mask=np.zeros((24,24),np.uint8)
    mask[3:7,12:21]=1
    image=np.zeros((24,24,3),np.uint8)
    image[:,:,0]=mask*255
    for mode in ('flip_only','rotation_only'):
        for seed in range(12):
            random.seed(seed)
            aug,gt=apply_augment(image,mask,mode)
            assert np.array_equal(aug[:,:,0]>0,gt.astype(bool))
            assert gt.sum()==mask.sum()
    assert len(planned_tasks('geometry',(111,222,333)))==6


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(3,1,1)

    def forward(self,x):
        return self.conv(x)


def test_v3_end_to_end(tmp_path,monkeypatch):
    """Train one small checkpoint and exercise all v3 public diagnostics."""
    from experiments import engine
    monkeypatch.chdir(tmp_path)
    Path('data/PRE-event').mkdir(parents=True)
    ids=[f'scene_0_{r}_{c}' for r in range(3) for c in range(3)]
    with Path('data/Germany_Training_Public_reference.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['ImageId','Object','Wkt_Pix'])
        writer.writeheader()
        for image_id in ids:
            writer.writerow({'ImageId':image_id,'Object':'Building',
                             'Wkt_Pix':'POLYGON ((10 10, 60 10, 60 60, 10 60, 10 10))'})
    for image_id in ids:
        with rasterio.open(Path('data/PRE-event')/f'{image_id}.tif','w',driver='GTiff',
                           height=512,width=512,count=3,dtype='uint8',
                           transform=from_origin(0,512,1,1)) as ds:
            data=np.zeros((3,512,512),np.uint8)
            data[0,:,:]=128
            data[1,10:60,10:60]=220
            ds.write(data)
    prepare(seed=111,block_size=1)
    monkeypatch.setattr(engine,'new_model',TinyModel)
    monkeypatch.setattr(engine,'load_architecture',TinyModel)
    run=engine.train_one('raw_bce_dice',seed=111,epochs=1,batch_size=2,workers=0,device_name='cpu')
    engine.optimize_threshold(run,batch_size=2,device_name='cpu')

    curve=threshold_curves(run,batch_size=2,device_name='cpu')
    assert len(curve)>=19
    assert (run/'diagnostics_v3/validation_threshold_curves.png').is_file()
    sizes,overall=building_sizes(run,batch_size=2,device_name='cpu')
    assert sum(row['objects'] for row in sizes)==len(read_json(SPLITS)['val'])
    assert 0<=overall['object_f1']<=1
    records=annotation_audit(run,batch_size=2,device_name='cpu',count=2)
    assert all(x['mask_vs_reference_different_pixels']==0 for x in records)
    assert (run/'diagnostics_v3/annotation_review_candidates.png').is_file()
    stats=geometry_audit(samples=8,count=2)
    assert all(x['failures']==0 for x in stats.values())
    assert Path('experiments_output/diagnostics_v3/geometry_alignment_audit.png').is_file()

    external=Path('external')
    external.mkdir()
    for image_id in read_json(SPLITS)['val']:
        _,gt=load_tile(image_id)
        with rasterio.open(external/f'{image_id}_buildingpred.tif','w',
                           driver='GTiff',height=512,width=512,count=1,
                           dtype='uint8',transform=from_origin(0,512,1,1)) as ds:
            ds.write((gt*255).astype(np.uint8),1)
    report=compare_official_validation(run,external,batch_size=2,device_name='cpu')
    assert report['external_global_pixel']['iou']==1
    assert report['split']=='val'
    assert (run/'diagnostics_v3/official_comparison/validation_paired_delta.png').exists()
    # The test split is not read, evaluated or finalized by any v3 command.
    assert not (run/'test_metrics.json').exists()
    assert not (run/'test_predictions').exists()


def test_split_neighbors_only_cross_split(tmp_path,monkeypatch):
    from experiments.v3 import split_neighbor_audit
    monkeypatch.chdir(tmp_path)
    root=Path('experiments_output')
    data=root/'dataset'
    data.mkdir(parents=True)
    # One neighboring train/val tile, and another non-neighboring test tile.
    splits={'train':['scene_0_2_3'],'val':['scene_0_2_4'],'test':['scene_0_8_8']}
    (root/'splits.json').write_text(json.dumps(splits))
    from experiments.common import sha256_file
    Path('data').mkdir()
    ref=Path('data/Germany_Training_Public_reference.csv')
    ref.write_text('ImageId,Object,Wkt_Pix\n')
    (root/'manifest.json').write_text(json.dumps({'reference_sha256':sha256_file(ref)}))
    for image_id in sum((splits[k] for k in ('train','val','test')),[]):
        np.savez_compressed(data/f'{image_id}.npz',image=np.zeros((8,8,3),np.uint8),
                            mask=np.zeros((8,8),np.uint8))
    edges=split_neighbor_audit()
    assert len(edges)==1
    assert set([edges[0]['split_a'],edges[0]['split_b']])=={'train','val'}
