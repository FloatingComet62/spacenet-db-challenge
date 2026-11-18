"""v4 smoke tests use only synthetic TIFFs, masks and cached probabilities."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from experiments.common import OUT, SPLITS, load_tile, read_json, write_json, sha256_file
from experiments.data import prepare
from experiments.v3 import building_sizes
from experiments.v4 import (Tile, filter_components, watershed_labels, instance_polygons,
                            make_prediction, counts, object_metrics, select_best,
                            failure_records, score_configuration, postprocess_study,
                            instance_audit, compare_external_instances)


def test_filter_and_watershed_preserve_expected_pixels():
    mask = np.zeros((96, 96), np.uint8)
    mask[10:30, 10:31] = 1
    mask[10:30, 35:55] = 1
    mask[15:25, 31:35] = 1  # thin bridge between two roofs
    mask[80, 80] = 1       # isolated false positive
    filtered = filter_components(mask, 10)
    assert filtered.sum() == mask.sum() - 1
    labels = watershed_labels(filtered, min_distance=8)
    assert np.array_equal(labels > 0, filtered)
    assert len(instance_polygons(labels)) >= 2


def test_baseline_uses_raw_pixel_mask_but_object_minimum():
    truth = np.zeros((40, 40), bool)
    truth[4:18, 4:18] = True
    probs = np.full(truth.shape, .01, dtype=np.float32)
    probs[truth] = .9
    probs[37,37] = .8  # tiny FP does not count as polygon under 16px
    tile = Tile('fake',truth,probs,[box(4,4,18,18)])
    raw = score_configuration([tile], {'method':'raw','threshold':.25,'min_area':16})
    filtered = score_configuration([tile], {'method':'filter','threshold':.25,'min_area':16})
    assert raw['instance_f1'] == 1.0
    assert filtered['instance_f1'] == 1.0
    assert raw['pixel_iou'] < filtered['pixel_iou']
    assert filtered['pixel_iou'] == 1.0


def test_best_respects_pixel_constraint():
    rows=[{'method':'raw','instance_f1':.50,'pixel_iou':.57},
          {'method':'filter','instance_f1':.80,'pixel_iou':.56},
          {'method':'watershed','instance_f1':.60,'pixel_iou':.57}]
    assert select_best(rows,.57,0)['method']=='watershed'
    assert select_best(rows,.57,.02)['method']=='filter'
    assert rows[1]['eligible']


def test_failure_categories_are_diagnostic():
    tile=Tile('fake',np.zeros((32,32),bool),np.zeros((32,32)),
              [box(1,1,5,5),box(10,10,20,20)])
    rows=failure_records(tile,[box(10,10,15,15)])
    assert rows[0]['reason']=='no_predicted_overlap'
    assert rows[1]['reason']=='below_iou_0_50'


def test_v4_commands_on_existing_validation_cache(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=Path('data/PRE-event')
    root.mkdir(parents=True)
    image_ids=[f'scene_0_{r}_{c}' for r in range(3) for c in range(3)]
    with Path('data/Germany_Training_Public_reference.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['ImageId','Object','Wkt_Pix'])
        writer.writeheader()
        for tile_id in image_ids:
            writer.writerow({'ImageId':tile_id,'Object':'Building',
                             'Wkt_Pix':'POLYGON ((10 10, 24 10, 24 24, 10 24, 10 10))'})
            writer.writerow({'ImageId':tile_id,'Object':'Building',
                             'Wkt_Pix':'POLYGON ((30 32, 53 32, 53 54, 30 54, 30 32))'})
    for tile_id in image_ids:
        with rasterio.open(root/f'{tile_id}.tif','w',driver='GTiff',width=64,height=64,
                           count=3,dtype='uint8',transform=from_origin(0,64,1,1)) as t:
            t.write(np.full((3,64,64),127,np.uint8))
    prepare(seed=111,block_size=1)
    run=OUT/'runs'/'raw_bce_dice_seed111'
    run.mkdir(parents=True)
    # A complete cached run does not require importing SMP or loading model weights.
    (run/'best.pt').write_bytes(b'fake checkpoint bytes, never executed')
    split_sha=hashlib.sha256(SPLITS.read_bytes()).hexdigest()
    config={'split_sha256':split_sha,'preprocessor':'raw','stride':512}
    write_json(run/'config.json',config)
    write_json(run/'threshold.json',{'threshold':.25})
    ids=read_json(SPLITS)['val']
    cache=run/'diagnostics_v3'/'validation_probs'
    cache.mkdir(parents=True)
    write_json(cache/'metadata.json',{
        'checkpoint_sha256':sha256_file(run/'best.pt'),
        'split_sha256':split_sha,'preprocessor':'raw',
        'stride':512,'ids':ids,'format':'float32_probability_v1'})
    ext=Path('external')
    ext.mkdir()
    for image_id in ids:
        _,gt=load_tile(image_id)
        prob=np.where(gt > 0,.9,.02).astype(np.float32)
        prob[4:6,4:6]=.7   # false positive
        np.save(cache/f'{image_id}.npy',prob)
        np.save(ext/f'{image_id}.npy',gt.astype(np.uint8))
    _,existing=building_sizes(run,batch_size=1,device_name='cpu')
    report=postprocess_study(run,device_name='cpu',thresholds=[.25,.5],
                             areas=[0,8,16],peak_distances=[6],watershed_top=1)
    assert report['baseline']['instance_f1']==pytest.approx(existing['object_f1'])
    assert report['baseline']['pixel_iou']<=report['selected']['pixel_iou'] + 1e-12
    assert (run/'diagnostics_v4/validation_postprocess_grid.csv').exists()
    assert (run/'diagnostics_v4/validation_postprocess_tradeoff.png').exists()
    summary, reasons=instance_audit(run,'baseline',device_name='cpu',count=2)
    assert sum(r['objects'] for r in summary)==2*len(ids)
    assert (run/'diagnostics_v4/baseline_instance_audit/validation_per_building.csv').exists()
    cmp=compare_external_instances(run,ext,selection='baseline',device_name='cpu')
    assert cmp['external_pixel']['iou']==1.0
    assert cmp['external_instance']['instance_f1']==1.0
    assert not (run/'test_metrics.json').exists()
    assert not (run/'test_predictions').exists()
