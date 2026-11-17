"""Small synthetic tests; no user dataset, SMP, or GPU required."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from experiments.common import patch_starts
from experiments.data import prepare, read_annotations, make_splits
from experiments.compare import instance_counts, load_prediction, export_official_csv
from experiments.engine import preprocess, metrics_from_counts, predict_full
import torch


def test_patch_coverage():
    assert patch_starts(1300,512,512) == [0,512,788]
    assert patch_starts(400,512,512) == [0]
    assert patch_starts(1024,512,512) == [0,512]


def test_spatial_split():
    ids = [f'scene_0_{r}_{c}' for r in range(5) for c in range(5)]
    splits, groups = make_splits(ids, seed=111,block_size=2,mode='spatial')
    assert sum((splits[s] for s in ('train','val','test')),[]) != []
    assert len(set(sum([splits[k] for k in ('train','val','test')],[]))) == len(ids)
    assert not (set(groups['test']) & set(groups['train']))
    assert not (set(groups['val']) & set(groups['train']))


def test_metrics():
    m = metrics_from_counts(10,5,5)
    assert m['iou']==.5
    assert m['dice']==2/3
    assert m['precision']==2/3
    assert m['recall']==2/3


def test_preprocessors_do_not_mutate():
    image = np.random.default_rng(2).integers(0,255,(80,80,3),dtype=np.uint8)
    old = image.copy()
    for name in ('raw','clahe','gamma','bilateral','sharpen','chainA','chainB','chainC'):
        out = preprocess(image,name)
        assert out.shape==image.shape and out.dtype==np.uint8
        assert np.array_equal(old,image)


class IdentityLogits(torch.nn.Module):
    def forward(self, x):
        return x[:,0:1]


def test_full_tile_overlap_scores_each_pixel_once():
    image = np.zeros((580,617,3),dtype=np.uint8)
    image[:,:,0] = 255
    pred = predict_full(IdentityLogits(),image,torch.device('cpu'),batch_size=2,stride=256)
    assert pred.shape==(580,617)
    expected = float(torch.sigmoid(torch.tensor((1-.485)/.229)))
    assert np.allclose(pred,expected,atol=1e-6)


def test_corrected_rasterization_all_polygons(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (Path('data/PRE-event')).mkdir(parents=True)
    ref = Path('data/Germany_Training_Public_reference.csv')
    ids=[f'scene_0_{r}_{c}' for r in range(3) for c in range(3)]
    for image_id in ids:
        with rasterio.open(Path('data/PRE-event')/f'{image_id}.tif','w',driver='GTiff',
                           height=64,width=64,count=3,dtype='uint8',transform=from_origin(0,64,1,1)) as ds:
            ds.write(np.zeros((3,64,64),dtype=np.uint8))
    with ref.open('w',newline='') as fh:
        writer = csv.DictWriter(fh,fieldnames=['ImageId','Object','Wkt_Pix'])
        writer.writeheader()
        writer.writerow({'ImageId':ids[0],'Object':'Building','Wkt_Pix':'POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))'})
        writer.writerow({'ImageId':ids[0],'Object':'Building','Wkt_Pix':'POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))'})
        writer.writerow({'ImageId':ids[1],'Object':'Road','Wkt_Pix':'POLYGON ((0 0, 60 0, 60 60, 0 60, 0 0))'})
        writer.writerow({'ImageId':ids[2],'Object':'Building','Wkt_Pix':'POLYGON EMPTY'})
    manifest=prepare(seed=111,block_size=1)
    from experiments.common import load_tile
    img,mask=load_tile(ids[0])
    assert mask[5,5]==1 and mask[25,25]==1
    assert mask[15,15]==0
    assert manifest['n_tiles']==9
    assert manifest['annotation_stats']['building_rows']==3
    assert manifest['n_empty_tiles']==8
    assert len(json.loads(Path('experiments_output/splits.json').read_text())['test'])>0
    ann,_=read_annotations()
    assert len(ann[ids[0]])==2
    result=instance_counts(mask.astype(bool),ann[ids[0]],min_area=1)
    assert result==(2,0,0)
    # Official output convention: 255 indicates building.
    external=Path('external')
    external.mkdir()
    for image_id in ids:
        _,mask=load_tile(image_id)
        with rasterio.open(external/f'{image_id}_buildingpred.tif','w',driver='GTiff',
                           height=64,width=64,count=1,dtype='uint8',transform=from_origin(0,64,1,1)) as ds:
            ds.write((mask*255).astype(np.uint8),1)
    assert load_prediction(external,ids[0]).sum()==164
    # Export official CSV, respecting the same fixed split.
    src_csv=Path('official.csv')
    with src_csv.open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['preimg','postimg','flood','building','road','roadspeed'])
        writer.writeheader()
        for image_id in ids:
            writer.writerow({'preimg':f'/data/{image_id}.tif','postimg':'a','flood':'b','building':'c','road':'d','roadspeed':'e'})
    export_official_csv([src_csv])
    assert sum(len(list(csv.DictReader(open(Path('experiments_output/official_csv')/f'{s}.csv')))) for s in ['train','val','test'])==9


def test_training_tuning_finalization_smoke(tmp_path,monkeypatch):
    """Full experiment flow with a tiny fake model (does not download pretrained weights)."""
    from experiments import engine
    monkeypatch.chdir(tmp_path)
    Path('data/PRE-event').mkdir(parents=True)
    ids=[f'scene_0_{r}_{c}' for r in range(3) for c in range(3)]
    for image_id in ids:
        with rasterio.open(Path('data/PRE-event')/f'{image_id}.tif','w',driver='GTiff',
                           height=32,width=32,count=3,dtype='uint8',transform=from_origin(0,32,1,1)) as ds:
            image=np.zeros((3,32,32),dtype=np.uint8)
            image[0] = 180
            ds.write(image)
    with Path('data/Germany_Training_Public_reference.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['ImageId','Object','Wkt_Pix'])
        writer.writeheader()
        for image_id in ids:
            writer.writerow({'ImageId':image_id,'Object':'Building',
                             'Wkt_Pix':'POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))'})
    prepare(seed=111,block_size=1)

    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=torch.nn.Conv2d(3,1,1)
        def forward(self,x):
            return self.conv(x)

    monkeypatch.setattr(engine,'new_model',FakeUNet)
    monkeypatch.setattr(engine,'load_architecture',FakeUNet)
    run=engine.train_one('matched_bce',seed=111,epochs=1,batch_size=2,workers=0,device_name='cpu')
    assert (run/'best.pt').exists()
    thresh=engine.optimize_threshold(run,batch_size=2,device_name='cpu')
    assert .2<=thresh['threshold']<=.8
    score=engine.finalize(run,batch_size=2,device_name='cpu')
    assert 0<=score['iou']<=1
    from experiments.compare import evaluate_external
    result=evaluate_external(run/'test_predictions',kind='probabilities',
                             threshold=thresh['threshold'],name='synthetic_test')
    assert result['pixel']['iou']==score['iou']
    assert 'instance' in result


def test_validation_visual_diagnostics_smoke(tmp_path, monkeypatch):
    from experiments.diagnostics import save_augmentation_preview
    from experiments.error_analysis import save_validation_errors
    from experiments import engine
    monkeypatch.chdir(tmp_path)
    Path('data/PRE-event').mkdir(parents=True)
    ids=[f'scene_0_{r}_{c}' for r in range(3) for c in range(3)]
    with Path('data/Germany_Training_Public_reference.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['ImageId','Object','Wkt_Pix'])
        writer.writeheader()
        for image_id in ids:
            writer.writerow({'ImageId':image_id,'Object':'Building',
                             'Wkt_Pix':'POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))'})
    for image_id in ids:
        with rasterio.open(Path('data/PRE-event')/f'{image_id}.tif','w',driver='GTiff',
                           height=32,width=32,count=3,dtype='uint8',transform=from_origin(0,32,1,1)) as ds:
            ds.write(np.full((3,32,32),180,dtype=np.uint8))
    prepare(seed=111,block_size=1)
    assert save_augmentation_preview(count=2,seed=111).is_file()

    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=torch.nn.Conv2d(3,1,1)
        def forward(self,x):
            return self.conv(x)
    monkeypatch.setattr(engine,'new_model',FakeUNet)
    monkeypatch.setattr(engine,'load_architecture',FakeUNet)
    run=engine.train_one('matched_bce',seed=111,epochs=1,batch_size=2,workers=0,device_name='cpu')
    engine.optimize_threshold(run,batch_size=2,device_name='cpu')
    rows=save_validation_errors(run,count=2,batch_size=2,device_name='cpu')
    assert rows
    assert (run/'validation_errors.png').exists()
    assert (run/'validation_error_report.csv').exists()
