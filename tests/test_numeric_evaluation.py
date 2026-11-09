"""CPU synthetic fixtures exercise actual coordinate, routing and aggregation math."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from plane_conditioned_segmentation.common import sha256_file, write_json
from plane_conditioned_segmentation.evaluation import classification, final
from plane_conditioned_segmentation.evaluation import metrics as evaluation


@pytest.fixture
def cpu_cuda_calls(monkeypatch):
    # Only adapt device transfers; production crop/interpolation/threshold math stays real.
    original = torch.tensor

    def tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', tensor)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)


@pytest.mark.parametrize('binary,offtarget', [(False, False), (True, False), (False, True)])
def test_common_original_removes_padding_and_uses_only_target_polygon(
    monkeypatch, tmp_path, cpu_cuda_calls, binary, offtarget
):
    (tmp_path / 'audit.json').write_text(
        json.dumps(dict(classes=['target', 'other'], class_map={'source.zip': ['target', 'other']}))
    )
    (tmp_path / 'manifest.json').write_text(
        json.dumps([dict(archive='source.zip', name='source/images/a.png', label='source/labels/explicit.txt')])
    )
    (tmp_path / 'geometry.csv').write_text(
        'index,top,left,content_width,content_height,original_width,original_height\n0,3,2,4,4,4,4\n'
    )
    with zipfile.ZipFile(tmp_path / 'source.zip', 'w') as z:
        z.writestr('source/labels/explicit.txt', '0 0 0 .5 0 .5 1 0 1\n1 0 0 1 0 1 1 0 1\n')

    class Images:
        def __init__(self, ids):
            self.ids = ids

        def __getitem__(self, key):
            return torch.zeros(1, 8, 8), torch.zeros(7, 8, 8), torch.ones(1, 8, 8), 0, 0

    class Model:
        target = 0 if binary else None

        def __call__(self, a, c):
            z = torch.full((1, 1 if binary else 7, 8, 8), -30.0)
            z[:, 0, 3:7, 2:4] = 30
            if offtarget:
                z[:, 1, 3:7, 2:6] = 30
            return z

    monkeypatch.setattr(final, 'DATA', tmp_path)
    monkeypatch.setattr(final, 'SOURCE_ARCHIVES', tmp_path)
    monkeypatch.setattr(final, 'SPEC', dict(targets=[dict(source_plane='source.zip')]))
    monkeypatch.setattr(final, 'OLD_CLASSES', list(range(7)))
    monkeypatch.setattr(final, 'Images', Images)
    score = final.common_original(Model(), np.array([0]))
    assert score['dice'] == pytest.approx(1) and score['macro_dice'] == pytest.approx(1)
    assert score['global_dice'] == pytest.approx(0.5 if offtarget else 1, abs=1e-6)


@pytest.mark.parametrize('estimated', [False, True])
def test_evaluation_routing_metrics_and_row_order(monkeypatch, cpu_cuda_calls, estimated):
    p = torch.arange(7)
    idx = torch.arange(6, -1, -1)
    a = torch.zeros(7, 1, 4, 4)
    v = torch.ones(7, 1, 4, 4)
    y = torch.zeros(7, 7, 4, 4)
    y[torch.arange(7), p] = 1
    monkeypatch.setattr(evaluation, 'loader', lambda ids: ([(a, y, v, p, idx)], None))

    class Model:
        target = None

        def eval(self):
            return self

        def __call__(self, a, c):
            self.conditions = c
            z = torch.full((7, 7, 4, 4), -30.0)
            z[torch.arange(7), p] = 30
            return z

    m = Model()
    conditions = {int(i): (torch.eye(7)[int(t)] * 0.8 + 0.2 / 7).numpy() for i, t in zip(idx, p)} if estimated else None
    result = evaluation.evaluate(m, np.arange(7), conditions)
    assert [r['index'] for r in result['rows']] == list(range(7))
    assert result['summary']['macro_dice'] == pytest.approx(1)
    assert result['summary']['offtarget_fp_pixels'] == 0
    expected = torch.stack([torch.from_numpy(conditions[int(i)]) for i in idx]) if estimated else torch.eye(7)
    torch.testing.assert_close(m.conditions, expected)


def test_classifier_confusion_macro_and_sorted_predictions(monkeypatch, cpu_cuda_calls):
    p = torch.arange(7)
    idx = torch.arange(6, -1, -1)
    a = torch.zeros(7, 1, 4, 4)
    monkeypatch.setattr(classification, 'loader', lambda ids: ([(a, None, None, p, idx)], None))

    class Classifier:
        def eval(self):
            return self

        def __call__(self, a):
            return torch.eye(7) * 30

    result = classification.evaluate(Classifier(), np.arange(7))
    assert result['accuracy'] == 1 and result['macro_accuracy'] == 1
    assert result['confusion'] == np.eye(7, dtype=int).tolist()
    assert [r['index'] for r in result['rows']] == list(range(7))


@pytest.mark.parametrize(
    'rows',
    [
        [],
        [
            dict(
                plane=0,
                **{
                    k: float('nan')
                    for k in [
                        'dice',
                        'global_dice',
                        'offtarget_fp_pixels',
                        'offtarget_fp_rate',
                        'target_fp',
                        'target_fn',
                        'target_predicted_pixels',
                        'target_gt_pixels',
                    ]
                },
            )
        ],
    ],
)
def test_aggregation_refuses_empty_or_nonfinite_metrics(rows):
    with pytest.raises(ValueError):
        evaluation.summarize(rows)


def test_original_polygon_accepts_same_blank_lines_as_preparation():
    from plane_conditioned_segmentation.data.preparation import _rasterize

    text = '\n0 0 0 1 0 1 1 0 1\n  \n'
    original = final._original_truth(text, 0, dict(original_width=4, original_height=4))
    cached = _rasterize(text.splitlines(), ['target'], ['target'], (4, 4), 4, 4, 0, 0)[0]
    np.testing.assert_array_equal(original, cached)
    assert original.sum() == 16


@pytest.mark.parametrize('changed', [None, 'study_plan.json', 'cohort.json'])
def test_final_evaluation_rejects_changed_frozen_inputs(tmp_path, monkeypatch, changed):
    for filename in ['study_plan.json', 'cohort.json']:
        write_json(tmp_path / filename, {'value': 1})
    write_json(
        tmp_path / 'FINAL_MODEL_LOCK.json',
        dict(
            full_training_models=11,
            checkpoints={},
            study_plan_sha256=sha256_file(tmp_path / 'study_plan.json'),
            cohort_sha256=sha256_file(tmp_path / 'cohort.json'),
        ),
    )
    monkeypatch.setattr(final, 'ROOT', tmp_path)
    if changed:
        write_json(tmp_path / changed, {'value': 2})
        with pytest.raises(ValueError, match='Frozen evaluation input changed'):
            final._verify_lock()
    else:
        final._verify_lock()
