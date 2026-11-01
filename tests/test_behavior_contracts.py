"""Hand-counted numerical oracles and real temporary-file state transitions."""

import copy
import datetime
import hashlib
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import record_behavior
from plane_conditioned_segmentation.config import LEGACY_V1_SECTIONS, SufficiencyGate
from plane_conditioned_segmentation.evaluation import metrics as evaluation
from plane_conditioned_segmentation.models import classifier as classification
from plane_conditioned_segmentation.models import segmentation as model
from plane_conditioned_segmentation.pipeline import registry as run_registry
from plane_conditioned_segmentation.pipeline import stages
from plane_conditioned_segmentation.training import losses, runtime, sufficiency

GATE = SufficiencyGate(**LEGACY_V1_SECTIONS['sufficiency_gate'])
from plane_conditioned_segmentation.evaluation import metrics as metric_module


def score(*args, **kwargs):
    return metric_module.metrics(*args, threshold=0.5, **kwargs)


import test_registry as registry_fixtures


def tensors():
    y = torch.zeros(1, 7, 1, 4)
    y[0, 0, 0, :2] = 1
    return torch.full((1, 7, 1, 4), -20.0), y, torch.ones(1, 1, 1, 4), torch.tensor([0])


@pytest.mark.parametrize(
    'positive,tp,fp,fn',
    [((), 0, 0, 2), ((0,), 1, 0, 1), ((0, 1), 2, 0, 0), ((0, 1, 2), 2, 1, 0), ((2, 3), 0, 2, 2)],
    ids=['all_FN', 'one_FN', 'perfect', 'one_FP', 'disjoint_FP_FN'],
)
def test_hand_counted_fp_fn_dice(positive, tp, fp, fn):
    z, y, v, p = tensors()
    for i in positive:
        z[0, 0, 0, i] = 20
    r = score(z, y, v, p)
    assert int(r['tp'][0]) == tp and int(r['target_fp'][0]) == fp and int(r['target_fn'][0]) == fn
    assert int(r['target_predicted_pixels'][0]) == tp + fp and int(r['target_gt_pixels'][0]) == tp + fn
    # Dice definition applied to independently enumerated counts, not a second mask algorithm.
    expected = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    assert r['dice'][0] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    'logit,expected_pc', [(-0.001, 0), (0.0, 0), (0.001, 4)], ids=['below_half', 'exact_half_is_negative', 'above_half']
)
def test_strict_half_probability_threshold(logit, expected_pc):
    z, y, v, p = tensors()
    z[:, 0] = logit
    r = score(z, y, v, p)
    assert r['target_predicted_pixels'][0] == expected_pc


def test_padding_never_counts_as_fp_or_fn():
    z, y, v, p = tensors()
    v[:, :, :, -1] = 0
    y[:, 0, 0, 3] = 1
    z[:, 0, 0, [0, 3]] = 20
    r = score(z, y, v, p)
    assert (
        r['target_predicted_pixels'][0] == 1
        and r['target_gt_pixels'][0] == 2
        and r['target_fp'][0] == 0
        and r['target_fn'][0] == 1
    )


def test_offtask_false_positive_does_not_change_target_dice():
    z, y, v, p = tensors()
    z[:, 0, 0, 0] = 20
    z[:, 1, 0, 2:] = 20
    r = score(z, y, v, p)
    assert r['dice'][0] == pytest.approx((2 + 1e-6) / (3 + 1e-6))
    assert r['global_dice'][0] == pytest.approx((2 + 1e-6) / (5 + 1e-6))
    assert r['offtarget_fp_pixels'][0] == 2 and r['offtarget_fp_rate'][0] == pytest.approx(2 / 24)


@pytest.mark.parametrize('has_prediction', [False, True])
def test_empty_ground_truth_has_explicit_set_semantics(has_prediction):
    z, y, v, p = tensors()
    y.zero_()
    if has_prediction:
        z[0, 0, 0, 2] = 20
    r = score(z, y, v, p)
    assert r['target_gt_pixels'][0] == 0 and r['target_fn'][0] == 0
    assert r['target_fp'][0] == int(has_prediction)
    assert r['dice'][0] == pytest.approx(1e-6 / (int(has_prediction) + 1e-6))
    # This is empty-set arithmetic, not a claim of a clinically negative cohort.


def test_zero_logit_loss_has_closed_form_oracle():
    z, y, v, p = tensors()
    z.zero_()
    y[:, 0] = 1
    # Four GT pixels: target BCE=ln2, target Dice loss=2/7;
    # six equally weighted negative channels contribute another ln2.
    assert losses.loss(z, y, v, p, 1).item() == pytest.approx(2 * math.log(2) + 2 / 7, abs=1e-6)
    assert losses.loss(z[:, :1], y, v, p, 1).item() == pytest.approx(math.log(2) + 2 / 7, abs=1e-6)


@pytest.mark.parametrize('field', ['z', 'y', 'v'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')], ids=['nan', 'posinf', 'neginf'])
def test_nonfinite_tensors_are_rejected_not_silently_scored_as_fn(field, bad):
    z, y, v, p = tensors()
    {'z': z, 'y': y, 'v': v}[field].view(-1)[0] = bad
    with pytest.raises(ValueError):
        score(z, y, v, p)
    with pytest.raises(ValueError):
        losses.loss_parts(z, y, v, p)


@pytest.mark.parametrize(
    'fault',
    ['zero_support', 'fractional_mask', 'negative_mask', 'fractional_plane', 'negative_plane', 'large_plane', 'shape'],
)
def test_invalid_metric_and_loss_contracts(fault):
    z, y, v, p = tensors()
    if fault == 'zero_support':
        v.zero_()
    elif fault == 'fractional_mask':
        y[0, 0, 0, 0] = 0.25
    elif fault == 'negative_mask':
        v[0, 0, 0, 0] = -1
    elif fault == 'fractional_plane':
        p = torch.tensor([0.5])
    elif fault == 'negative_plane':
        p = torch.tensor([-1])
    elif fault == 'large_plane':
        p = torch.tensor([7])
    else:
        y = y[:, :, :, :3]
    with pytest.raises(ValueError):
        score(z, y, v, p)
    with pytest.raises(ValueError):
        losses.loss_parts(z, y, v, p)


@pytest.mark.parametrize('target', [None, -1, 7, True])
def test_binary_expert_requires_explicit_valid_target(target):
    z, y, v, p = tensors()
    with pytest.raises(ValueError):
        score(z[:, :1], y, v, p, target=target)


@pytest.mark.parametrize(
    'key,value', [('dice', 1.1), ('global_dice', -0.1), ('offtarget_fp_rate', 1.1), ('target_fp', -1)]
)
def test_aggregate_rejects_impossible_metrics(key, value):
    row = dict(
        plane=0,
        dice=0.5,
        global_dice=0.5,
        offtarget_fp_pixels=0,
        offtarget_fp_rate=0,
        target_fp=0,
        target_fn=0,
        target_predicted_pixels=1,
        target_gt_pixels=1,
    )
    row[key] = value
    with pytest.raises(ValueError):
        evaluation.summarize([row])


def history():
    def row(epoch):
        return dict(
            epoch=epoch,
            train=dict(macro_dice=0.8, macro_global_dice=0.7, per_target={'0': dict(dice=0.8), '1': dict(dice=0.8)}),
            validation=dict(selection=0.0),
        )

    return [row(1), row(10)]


@pytest.mark.parametrize('field,threshold', [('train', 0.70), ('global', 0.60), ('minimum', 0.50), ('gain', 0.02)])
@pytest.mark.parametrize('side', ['below', 'exact', 'above'])
def test_gate_ieee_boundary_contract(field, threshold, side):
    h = history()
    value = float(np.nextafter(threshold, -np.inf if side == 'below' else np.inf)) if side != 'exact' else threshold
    if field == 'train':
        h[-1]['train']['macro_dice'] = value
        h[-1]['train']['macro_global_dice'] = 0.65
        for target in h[-1]['train']['per_target'].values():
            target['dice'] = value
    elif field == 'global':
        h[-1]['train']['macro_global_dice'] = value
    elif field == 'minimum':
        h[-1]['train']['per_target']['0']['dice'] = value
        h[-1]['train']['per_target']['1']['dice'] = 1.0
        h[-1]['train']['macro_dice'] = (value + 1) / 2
    else:
        h[-1]['validation']['selection'] = value
    expected = (side != 'above') if field == 'gain' else (side != 'below')
    result = sufficiency.assess(h, GATE)
    assert result['pass_gate'] is expected
    record_behavior(
        'gate:' + field + ':' + side, dict(pass_gate=result['pass_gate'], gain=result['last20percent_selection_gain'])
    )


@pytest.mark.parametrize(
    'fault',
    [
        'empty',
        'nan_target',
        'infinite_validation',
        'duplicate_epoch',
        'reversed_epoch',
        'missing_terminal_metrics',
        'changed_target_set',
    ],
)
def test_bad_history_cannot_pass_or_choose_a_different_terminal(fault):
    h = history()
    if fault == 'empty':
        h = []
    elif fault == 'nan_target':
        h[-1]['train']['per_target']['1']['dice'] = float('nan')
    elif fault == 'infinite_validation':
        h[0]['validation']['selection'] = float('inf')
    elif fault == 'duplicate_epoch':
        h[-1]['epoch'] = h[0]['epoch']
    elif fault == 'reversed_epoch':
        h.reverse()
    elif fault == 'missing_terminal_metrics':
        h.append(dict(epoch=11))
    else:
        del h[-1]['train']['per_target']['1']
    with pytest.raises(ValueError):
        sufficiency.assess(h, GATE)


def test_registration_plan_register_replay_tamper_state_machine(monkeypatch, tmp_path):
    registry_fixtures.RegistryTests().fixture(tmp_path)

    class FixedClock:
        value = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.value

    monkeypatch.setattr(
        run_registry, 'datetime', types.SimpleNamespace(datetime=FixedClock, timezone=datetime.timezone)
    )
    assert run_registry.registration(tmp_path)['read_only']
    assert not (tmp_path / 'run_manifest.json').exists()
    assert run_registry.registration(tmp_path, True)['action'] == 'registered_existing_completed_run'
    manifest = tmp_path / 'run_manifest.json'
    before = manifest.read_bytes()
    assert json.loads(before)['registered_utc'] == '2020-01-01T00:00:00+00:00'
    projected = json.loads(before)
    projected.pop('experiment')
    record_behavior(
        'registration_projection', hashlib.sha256(json.dumps(projected, sort_keys=True).encode()).hexdigest()
    )
    FixedClock.value = datetime.datetime(2040, 1, 1, tzinfo=datetime.timezone.utc)
    for _ in range(2):
        assert run_registry.registration(tmp_path, True)['action'] == 'reuse_registration'
        assert manifest.read_bytes() == before
    (tmp_path / 'full/plain/history.json').write_text('[{"altered":true}]')
    with pytest.raises(ValueError):
        run_registry.registration(tmp_path, True)
    assert manifest.read_bytes() == before


def test_real_execution_lock_state_transitions_and_exception_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, 'REPO_ROOT', tmp_path)
    lock = tmp_path / '.local/execution.lock'
    with pytest.raises(RuntimeError):
        with stages.execution_lock():
            first = lock.read_bytes()
            with pytest.raises(ValueError):
                with stages.execution_lock():
                    raise AssertionError('Must not enter second owner')
            assert lock.read_bytes() == first
            raise RuntimeError('deterministic interruption')
    assert not lock.exists()
    with stages.execution_lock():
        assert lock.exists()
    assert not lock.exists()


@pytest.mark.parametrize('mode', ['plain', 'expert', 'film', 'condconv'])
def test_actual_checkpoint_roundtrip_and_eval_repeatability(tmp_path, mode):
    torch.random.default_generator.manual_seed(11)
    m = model.Model(mode, 0 if mode == 'expert' else None, initialize=False).eval()
    a = torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64)
    c = torch.eye(7)[:1]
    with torch.inference_mode():
        expected = m(a, c)
    record_behavior('model:' + mode, dict(logits_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest()))
    path = tmp_path / 'synthetic.pt'
    runtime.checkpoint(path, m, epoch=0)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = model.load(path)
    second = tmp_path / 'second/synthetic.pt'
    second.parent.mkdir()
    runtime.checkpoint(second, m, epoch=0)
    assert hashlib.sha256(second.read_bytes()).hexdigest() == before
    with torch.inference_mode():
        torch.testing.assert_close(restored(a, c), expected, rtol=0, atol=0)
        torch.testing.assert_close(restored(a, c), expected, rtol=0, atol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    record_behavior('checkpoint:' + mode, before)


def test_actual_classifier_roundtrip_has_no_running_stat_side_effect(tmp_path):
    torch.random.default_generator.manual_seed(12)
    m = classification.Classifier(initialize=False).eval()
    a = torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64)
    with torch.inference_mode():
        expected = m(a)
    record_behavior('classifier_logits', hashlib.sha256(expected.numpy().tobytes()).hexdigest())
    path = tmp_path / 'classifier.pt'
    torch.save(dict(state=m.state_dict()), path)
    restored = classification.load_classifier(path)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with torch.inference_mode():
        torch.testing.assert_close(restored(a), expected, rtol=0, atol=0)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in before.items())
