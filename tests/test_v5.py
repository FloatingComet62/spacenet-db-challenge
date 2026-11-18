"""v5: confidence-preserving small-building rules and fully cached end-to-end study."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from experiments.common import OUT, SPLITS, read_json, write_json, sha256_file, load_tile
from experiments.data import prepare
from experiments.v4 import Tile
from experiments.v5 import (Policy, candidate_policies, confidence_filter,
                            score_policy, select_shared_policy, study, gallery)


def test_confidence_filter_preserves_high_conf_small_building():
    a=np.zeros((48,48),np.float32)
    a[2:8,2:8]=.92         # 36 pixel genuine building
    a[20:26,20:26]=.35     # 36 pixel false positive
    a[30:47,2:20]=.78     # 306 pixel large building
    mask,component_stats=confidence_filter(a,.25,large_area=64,small_area=8,
                                             confidence_metric='mean',confidence_cutoff=.7,
                                             return_stats=True)
    assert mask[3,3]
    assert not mask[21,21]
    assert mask[31,3]
    assert len(component_stats)==3
    assert sum(row['kept'] for row in component_stats)==2
    assert np.array_equal(confidence_filter(a,.25,64,8,'p90',.7),mask)
    assert np.array_equal(confidence_filter(a,.25,64,8,'max',.7),mask)


def test_scoring_and_per_seed_constraints():
    gt=np.zeros((48,48),bool)
    gt[2:8,2:8]=True
    gt[30:47,2:20]=True
    p=np.zeros((48,48),np.float32)
    p[gt]=.92
    p[20:26,20:26]=.35
    tile=Tile('example',gt,p,[box(2,2,8,8),box(2,30,20,47)])
    baseline=score_policy([tile],Policy('baseline_saved'),.25)
    filtered=score_policy([tile],Policy('filter','.25',64),.25)
    good=score_policy([tile],Policy('confidence','.25',64,8,'mean',.70),.25)
    assert baseline['small_matched']==1
    assert filtered['small_matched']==0
    assert good['small_matched']==1
    assert good['instance_f1'] > baseline['instance_f1']
    policies=[Policy('baseline_saved'),Policy('filter','.25',64),
              Policy('confidence','.25',64,8,'mean',.7)]
    results={policies[0]:[baseline,baseline],policies[1]:[filtered,filtered],
             policies[2]:[good,good]}
    chosen,grid=select_shared_policy(policies,results,[baseline,baseline])
    assert chosen['policy']==policies[2]
    assert not grid[1]['eligible']
    assert grid[2]['eligible']
    # The same policy must preserve small recall for every seed; mean alone is insufficient.
    damaged=dict(good,small_matched=0,small_recall=0)
    results[policies[2]]=[good,damaged]
    chosen,_=select_shared_policy(policies,results,[baseline,baseline])
    assert chosen['policy']==policies[0]


@pytest.mark.parametrize('bad',[{'confidence_metric':'median'}, {'confidence_cutoff':1.01}, {'small_area':64}])
def test_invalid_config_rejected(bad):
    kw=dict(threshold=.25,large_area=64,small_area=8,
            confidence_metric='mean',confidence_cutoff=.7)
    kw.update(bad)
    with pytest.raises(ValueError):
        confidence_filter(np.zeros((10,10),np.float32),**kw)


def test_v5_end_to_end_with_existing_validation_caches(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_dir=Path('data/PRE-event')
    image_dir.mkdir(parents=True)
    ids=[f'scene_0_{r}_{c}' for r in range(3) for c in range(4)]
    ref=Path('data/Germany_Training_Public_reference.csv')
    with ref.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['ImageId','Object','Wkt_Pix'])
        writer.writeheader()
        for name in ids:
            for poly in ['POLYGON ((5 5, 11 5, 11 11, 5 11, 5 5))',
                         'POLYGON ((28 25, 52 25, 52 52, 28 52, 28 25))']:
                writer.writerow({'ImageId':name,'Object':'Building','Wkt_Pix':poly})
    for name in ids:
        with rasterio.open(image_dir/f'{name}.tif','w',driver='GTiff',width=64,height=64,
                           count=3,dtype='uint8',transform=from_origin(0,64,1,1)) as image:
            image.write(np.full((3,64,64),127,np.uint8))
    prepare(seed=111,block_size=1)
    split_sha=sha256_file(SPLITS)
    val=read_json(SPLITS)['val']
    assert val
    for name in ('raw_bce_dice','augmentation_photometric'):
        for seed in (111,222,333):
            run=OUT/'runs'/f'{name}_seed{seed}'
            run.mkdir(parents=True)
            (run/'best.pt').write_bytes(f'fake_{name}_{seed}'.encode())
            conf={'split_sha256':split_sha,'preprocessor':'raw','stride':512}
            write_json(run/'config.json',conf)
            write_json(run/'threshold.json',{'threshold':.25})
            cache=run/'diagnostics_v3'/'validation_probs'
            cache.mkdir(parents=True)
            write_json(cache/'metadata.json',{'checkpoint_sha256':sha256_file(run/'best.pt'),
                       'split_sha256':split_sha,'preprocessor':'raw', 'stride':512,
                       'ids':val,'format':'float32_probability_v1'})
            for tile_id in val:
                _,gt=load_tile(tile_id)
                probs=np.where(gt>0,.90,.01).astype(np.float32)
                probs[16:22,16:22]=.35 # conf-low false-positive component
                np.save(cache/f'{tile_id}.npy',probs)
            write_json(run/'diagnostics_v4'/'validation_selection.json',{
                'split_sha256':split_sha,'checkpoint_sha256':sha256_file(run/'best.pt'),
                'selected':{'method':'filter','threshold':.25,'min_area':64,'peak_distance':None}})
    study(model_names=['raw_bce_dice','augmentation_photometric'],
          seeds=[111,222,333],device_name='cpu',thresholds=['saved','.25'],
          small_floors=[8,16],confidence_cutoffs=[.5,.7],
          confidence_metrics=['mean','p90'],max_iou_drop=.005)
    for model in ['raw_bce_dice','augmentation_photometric']:
        root=OUT/'diagnostics_v5'/model
        sel=read_json(root/'validation_shared_policy.json')
        assert sel['selected_policy']['method']=='confidence'
        assert len(sel['checkpoints_sha256'])==3
        assert (root/'validation_building_changes.csv').is_file()
        assert (root/'validation_shared_tradeoff.png').is_file()
        assert sel['selected_mean_small_recall']==1.0
        assert sel['selected_mean_f1']>sel['v4_mean_f1']
    result=gallery('augmentation_photometric',seed=111,count=2,device_name='cpu')
    assert result.is_file()
    assert not (OUT/'runs'/'raw_bce_dice_seed111'/'test_metrics.json').exists()
    assert not (OUT/'runs'/'raw_bce_dice_seed111'/'test_predictions').exists()
