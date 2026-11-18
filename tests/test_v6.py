"""V6 unit and tiny end-to-end tests; never read the real user's dataset."""
import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box

from experiments.common import OUT, SPLITS, read_json, write_json, sha256_file
from experiments.data import prepare
from experiments.v6 import (Task, boundary_weighted_loss, config_for_task,
    planned_tasks, sampling_weights, small_patch_flags, train_task,
    validation_report, _inspect_run, run_study)
from experiments.engine import PatchDataset


def test_unique_run_ids_and_contrasts():
    ids={t.name for t in planned_tasks('full')}
    assert len(ids)==6
    assert len(planned_tasks('duration'))==4
    assert len(planned_tasks('sampling'))==2
    assert Task('augmentation_photometric',60,'small_aware').name in ids
    assert Task('augmentation_photometric',60,'balanced','boundary').name in ids
    assert Task('raw_bce_dice',16).name != Task('raw_bce_dice',60).name
    with pytest.raises(ValueError):
        Task('raw_bce_dice',0).name


def test_balanced_positive_total_and_small_bias():
    positive=[True]*6+[False]*4
    small=[True]*2+[False]*8
    weights,meta=sampling_weights(positive,small,.35)
    assert np.isclose(weights.sum(),1)
    assert np.isclose(weights[:2].sum(),.35)
    assert np.isclose(weights[2:6].sum(),.15)
    assert np.isclose(weights[6:].sum(),.5)
    assert meta['patch_counts']=={'small':2,'other_positive':4,'empty':4}
    assert meta['sampling_quotas']['small']==.35


def test_sampler_absent_categories_and_rejects_invalid_flags():
    w,meta=sampling_weights([True,True,True],[True,True,True])
    assert np.isclose(w.sum(),1)
    assert meta['sampling_quotas']['small']==1
    with pytest.raises(ValueError,match='No small'):
        sampling_weights([True,False],[False,False])
    with pytest.raises(ValueError,match='must be positive'):
        sampling_weights([True,False],[False,True])


def test_boundary_loss_gradient_and_zero_weight_equivalence():
    x=torch.randn(2,1,16,16,requires_grad=True)
    y=torch.zeros_like(x)
    y[:,:,4:10,5:9]=1
    from experiments.engine import bce_dice
    assert torch.allclose(boundary_weighted_loss(x,y,0),bce_dice(x,y,'bce_dice'))
    value=boundary_weighted_loss(x,y,.5)
    assert value > bce_dice(x,y,'bce_dice')
    value.backward()
    assert torch.isfinite(x.grad).all()


def tiny_dataset(root):
    image_dir=root/'data/PRE-event'
    image_dir.mkdir(parents=True)
    ref=root/'data/Germany_Training_Public_reference.csv'
    ids=[f'scene_0_{i}_0' for i in range(12)]
    with ref.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['ImageId','Object','Wkt_Pix'])
        w.writeheader()
        for i,name in enumerate(ids):
            if i%3!=0:
                w.writerow({'ImageId':name,'Object':'Building',
                   'Wkt_Pix':'POLYGON ((10 10, 16 10, 16 16, 10 16, 10 10))'})
                w.writerow({'ImageId':name,'Object':'Building',
                   'Wkt_Pix':'POLYGON ((24 25, 39 25, 39 42, 24 42, 24 25))'})
    for i,name in enumerate(ids):
        im=np.full((3,64,64),120+i,dtype=np.uint8)
        with rasterio.open(image_dir/f'{name}.tif','w',driver='GTiff',width=64,height=64,
            count=3,dtype='uint8',transform=from_origin(0,64,1,1)) as raster:
            raster.write(im)
    prepare(seed=111,block_size=1)


def test_small_patch_flags_only_training_tiles(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_dataset(tmp_path)
    train_ids=read_json(SPLITS)['train']
    ds=PatchDataset(train_ids)
    flags=small_patch_flags(ds,100)
    assert len(flags)==len(ds)
    assert np.any(flags)
    assert not np.any(flags & ~np.asarray(ds.positive))


def test_dry_run_is_read_only_and_mismatched_config_fails(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_dataset(tmp_path)
    plan=run_study(phase='duration',seeds=[111],dry_run=True)
    assert len(plan)==4
    assert not (OUT/'runs').exists()
    cfg=config_for_task(Task('raw_bce_dice',16),111,2,1e-4,512,0,2,100,.35,.5)
    run=Task('raw_bce_dice',16).run_dir(111)
    run.mkdir(parents=True)
    write_json(run/'config.json',cfg)
    assert _inspect_run(run,cfg)=='TRAIN'
    with pytest.raises(RuntimeError,match='configuration mismatch'):
        _inspect_run(run,dict(cfg,epochs=99))


def test_v6_two_epoch_training_resume_and_full_report(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_dataset(tmp_path)
    from experiments import engine
    from experiments import v6
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=torch.nn.Conv2d(3,1,kernel_size=1)
        def forward(self,x):
            return self.conv(x)
    monkeypatch.setattr(v6,'new_model',Tiny)
    monkeypatch.setattr(engine,'load_architecture',Tiny)
    tasks=[Task('augmentation_photometric',2,'balanced'),
           Task('augmentation_photometric',2,'small_aware')]
    for task in tasks:
        path=train_task(task,seed=111,batch_size=2,lr=.0001,workers=0,
                   stride=512,val_every=1,device='cpu')
        assert (path/'best.pt').is_file()
        assert (path/'latest.pt').is_file()
        assert (path/'completed.json').is_file()
        assert (path/'threshold.json').is_file()
        assert read_json(path/'completed.json')['completed_epochs']==2
        # Idempotent complete run and threshold reuse.
        assert train_task(task,seed=111,batch_size=2,lr=.0001,workers=0,
                         stride=512,val_every=1,device='cpu')==path
    monkeypatch.setattr(v6,'planned_tasks',lambda phase:tasks)
    reports=validation_report('sampling',seeds=[111],device='cpu')
    assert len(reports)==2
    assert all(r['small_recall'] is not None for r in reports)
    assert (OUT/'diagnostics_v6'/'validation_learning_curves.png').is_file()
    assert (OUT/'diagnostics_v6'/'validation_summary.csv').is_file()
    assert not list((OUT/'runs').glob('*/test_metrics.json'))
    # Genuine end-of-last-epoch crash: completed.json wasn't yet persisted.
    path=tasks[0].run_dir(111)
    (path/'completed.json').unlink()
    assert _inspect_run(path,read_json(path/'config.json'))=='RESUME'
    assert train_task(tasks[0],seed=111,batch_size=2,lr=.0001,workers=0,
                      stride=512,val_every=1,device='cpu')==path
    assert (path/'completed.json').exists()
