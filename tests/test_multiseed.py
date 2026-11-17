"""New study checks (no GPU, no external data or model downloads required)."""
import hashlib
import json
import random

import numpy as np
import pytest

from experiments.common import write_json
from experiments.engine import apply_augment, config_for
from experiments.study import (planned_tasks, _validate_existing,
                               validation_report)


def test_augmented_masks_stay_aligned_and_inputs_immutable():
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[2:11, 5:18] = 1
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[..., 0] = mask * 255
    image_before, mask_before = image.copy(), mask.copy()
    for seed in range(10):
        for mode in ('none', 'geometry', 'photometric', 'combined'):
            random.seed(seed)
            im, gt = apply_augment(image, mask, mode)
            assert im.dtype == np.uint8 and gt.dtype == np.uint8
            assert im.flags.c_contiguous and gt.flags.c_contiguous
            assert set(np.unique(gt)).issubset({0, 1})
            assert np.array_equal(im[..., 0] > 128, gt.astype(bool))
            assert gt.sum() == mask.sum()
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_policy_compatibility():
    image = np.ones((8, 8, 3), np.uint8) * 128
    mask = np.ones((8, 8), np.uint8)
    random.seed(7)
    a = apply_augment(image, mask, 'combined')
    random.seed(7)
    b = apply_augment(image, mask, True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert config_for('augmentation_geometry')['augment_mode'] == 'geometry'
    assert config_for('augmentation_photometric')['augment_mode'] == 'photometric'


def test_planned_tasks():
    initial = planned_tasks('initial', (111, 222, 333))
    assert len(initial) == 9
    assert ('clahe', 222) in initial
    assert ('augmentation_geometry', 111) in initial
    assert ('augmentation_geometry', 222) not in initial
    assert len(planned_tasks('full', (111, 222, 333))) == 15
    with pytest.raises(ValueError, match='distinct'):
        planned_tasks('full', (111, 111))


def test_existing_run_validation(tmp_path):
    run = tmp_path / 'runs' / 'clahe_seed111'
    run.mkdir(parents=True)
    (run/'best.pt').write_bytes(b'dummy')
    (run/'completed.json').write_text('{"completed_epochs": 16}')
    expected = config_for('clahe')
    expected.update({'seed':111, 'epochs':16, 'batch_size':2,
                     'lr':1e-4, 'stride':512, 'workers':2, 'split_sha256':'abc'})
    write_json(run/'config.json',expected)
    assert _validate_existing(run,'clahe',111,16,2,1e-4,512,'abc')=='REUSE'
    with pytest.raises(RuntimeError, match='Cannot reuse'):
        _validate_existing(run,'clahe',111,50,2,1e-4,512,'abc')
    # Older configs lack augment_mode, but must still be reusable.
    original = config_for('augmentation')
    original.pop('augment_mode')
    original.update({'seed':111,'epochs':16,'batch_size':2,'lr':1e-4,
                     'stride':512,'workers':2,'split_sha256':'abc'})
    write_json(run/'config.json',original)
    assert _validate_existing(run,'augmentation',111,16,2,1e-4,512,'abc')=='REUSE'


def test_validation_report_excludes_test_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'experiments_output').mkdir()
    split = tmp_path/'experiments_output'/'splits.json'
    split.write_text('{"train": [], "val": [], "test": []}')
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    for seed, raw_iou, clahe_iou in ((111,.50,.52),(222,.55,.54)):
        for name, val_iou in (('raw_bce_dice',raw_iou),('clahe',clahe_iou)):
            run = tmp_path/'experiments_output'/'runs'/f'{name}_seed{seed}'
            cfg=config_for(name)
            cfg.update({'seed':seed,'epochs':16,'batch_size':2,'lr':1e-4,
                        'stride':512,'workers':2,'split_sha256':digest})
            write_json(run/'config.json', cfg)
            write_json(run/'threshold.json', {'threshold':.5,
                         'validation': {'iou':val_iou,'dice':.6,
                                        'precision':.5,'recall':.8}})
            write_json(run/'test_metrics.json', {'iou':1.0})
    rows=validation_report(names=['raw_bce_dice','clahe'],seeds=[111,222])
    assert len(rows)==4
    assert all(r['val_iou'] < 1 for r in rows)
    result=(tmp_path/'experiments_output'/'validation_summary.csv').read_text()
    assert '0.53' in result
