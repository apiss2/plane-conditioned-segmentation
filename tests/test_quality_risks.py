"""Synthetic CPU tests for numerical semantics, malformed input and artifact protection."""

import contextlib
import hashlib
import io
import json
import types
import zipfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from plane_conditioned_segmentation import common
from plane_conditioned_segmentation.data import dataset as data
from plane_conditioned_segmentation.data import preparation
from plane_conditioned_segmentation.evaluation import metrics as evaluation
from plane_conditioned_segmentation.evaluation import metrics as metric_module
from plane_conditioned_segmentation.pipeline import stages as cli
from plane_conditioned_segmentation.training import classifier as classification
from plane_conditioned_segmentation.training import losses
from plane_conditioned_segmentation.training import runtime as core
from plane_conditioned_segmentation.training import segmentation as training


def score(*args, **kwargs):
    return metric_module.metrics(*args, threshold=0.5, **kwargs)


import test_portability as portability_fixtures
import test_registry as registry_fixtures
from plane_conditioned_segmentation.paths import REPO_ROOT


@pytest.fixture(scope='module')
def source_fixture(tmp_path_factory):
    base = tmp_path_factory.mktemp('polygon_source')
    manifest, archives = portability_fixtures.PortableDataTests().fixtures(base)
    return base, manifest, archives


@pytest.fixture(scope='module')
def prepared_fixture(source_fixture, tmp_path_factory):
    base, manifest, archives = source_fixture
    out = tmp_path_factory.mktemp('prepared')
    preparation.prepare(
        manifest,
        REPO_ROOT / 'configs/data_definition.json',
        out / 'run',
        out / 'cache',
        archives,
        REPO_ROOT / 'configs/sevenplane_protocol.json',
    )
    return out


def bind_data(monkeypatch, prepared_fixture):
    monkeypatch.setattr(data, 'DATA', prepared_fixture / 'cache')
    monkeypatch.setattr(data, 'COHORT', json.loads((prepared_fixture / 'run/cohort.json').read_text()))


def test_loss_padding_and_effective_batch_gradients():
    torch.manual_seed(2)
    z = torch.randn(4, 7, 4, 6, requires_grad=True)
    y = torch.zeros_like(z)
    p = torch.tensor([0, 1, 2, 3])
    y[torch.arange(4), p, 1:3, 1:4] = 1
    v = torch.ones(4, 1, 4, 6)
    v[:, :, :, -2:] = 0
    full = losses.loss(z, y, v, p, 4)
    g = torch.autograd.grad(full, z, retain_graph=True)[0]
    accumulated = sum(losses.loss(z[i : i + 1], y[i : i + 1], v[i : i + 1], p[i : i + 1], 4) for i in range(4))
    h = torch.autograd.grad(accumulated, z)[0]
    torch.testing.assert_close(g, h)
    assert not g[:, :, :, -2:].any()
    assert torch.isfinite(full)


def test_expert_target_loss_matches_shared_and_offtask_penalty_is_separate():
    z = torch.zeros(2, 7, 3, 3)
    y = torch.zeros_like(z)
    p = torch.tensor([1, 1])
    y[:, 1] = 1
    v = torch.ones(2, 1, 3, 3)
    a, negative = losses.loss_parts(z, y, v, p)
    b, none = losses.loss_parts(z[:, 1:2], y, v, p)
    torch.testing.assert_close(a, b)
    assert negative > 0 and none == 0
    changed = z.clone()
    changed[:, 2] = 10
    torch.testing.assert_close(losses.loss_parts(changed, y, v, p)[0], a)
    assert losses.loss_parts(changed, y, v, p)[1] > negative


@pytest.mark.parametrize('expert', [False, True])
def test_metrics_wrong_channel_and_invalid_pixels_get_no_semantic_credit(expert):
    y = torch.zeros(1, 7, 2, 3)
    y[:, 0, :, :2] = 1
    v = torch.ones(1, 1, 2, 3)
    v[:, :, :, -1] = 0
    z = torch.full((1, 1 if expert else 7, 2, 3), -30.0)
    z[:, 0 if expert else 1] = 30
    result = score(z, y, v, torch.tensor([0]), target=1 if expert else None)
    assert result['dice'][0] < 1e-5 and result['offtarget_fp_pixels'][0] == 4 and result['target_fn'][0] == 4
    correct = torch.full_like(z, -30)
    correct[:, 0] = 30
    if expert:
        result = score(correct, y, v, torch.tensor([0]), target=0)
    else:
        result = score(correct, y, v, torch.tensor([0]))
    assert result['dice'][0] == pytest.approx(1) and result['global_dice'][0] == pytest.approx(1)


def test_macro_scores_weight_targets_equally_not_by_image_count():
    def row(plane, dice):
        return dict(
            plane=plane,
            **{
                k: dice
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

    score = evaluation.summarize([row(0, 1), row(0, 1), row(0, 1), row(1, 0)])
    assert score['dice'] == 0.75 and score['macro_dice'] == 0.5 and score['selection'] == 0.5


def test_bucket_sampler_is_complete_homogeneous_and_deterministic(monkeypatch, prepared_fixture):
    bind_data(monkeypatch, prepared_fixture)
    ids = data.ids_for(0)
    sampler = data.Batches(ids, 3, True)
    groups = sampler.groups()
    assert groups == sampler.groups() and len(sampler) == len(groups)
    assert sorted(i for group in groups for i, e in group) == list(range(len(ids)))
    widths = np.load(data.DATA / 'widths.npy')[ids]
    assert all(len(set(widths[i] for i, e in group)) == 1 for group in groups)
    sampler.epoch = 4
    assert all(e == 4 for group in sampler for i, e in group)


def test_data_cohort_missing_target_and_classification_policy(monkeypatch, prepared_fixture):
    bind_data(monkeypatch, prepared_fixture)
    ids = data.ids_for(0)
    first = int(ids[0])
    cohort = dict(data.COHORT)
    cohort['allowed_indices'] = {k: list(v) for k, v in cohort['allowed_indices'].items()}
    cohort['allowed_indices']['train'].remove(first)
    monkeypatch.setattr(data, 'COHORT', cohort)
    assert first not in data.ids_for(0)
    assert np.array_equal(data.plane_ids(ids), np.arange(7))
    original_load = np.load

    def load(path, *args, **kw):
        array = original_load(path, *args, **kw)
        if Path(path).name == 'areas.npy':
            array = array.copy()
            array[first] = 0
        return array

    monkeypatch.setattr(data, 'COHORT', json.loads((prepared_fixture / 'run/cohort.json').read_text()))
    monkeypatch.setattr(data.np, 'load', load)
    assert first not in data.ids_for(0) and first in data.ids_for(0, classification=True)
    assert len(data.ids_for(0, [1])) == 1


def test_dataset_augmentation_is_joint_and_epoch_seeded(monkeypatch, prepared_fixture):
    bind_data(monkeypatch, prepared_fixture)
    idx = data.ids_for(0)[:1]
    dataset = data.Images(idx, aug=True)
    a, y, v, p, i = dataset[(0, 2)]
    again = dataset[(0, 2)]
    assert torch.equal(a, again[0]) and torch.equal(y, again[1]) and torch.equal(v, again[2])
    assert not (y.bool() & ~v.bool()).any()
    assert p == 0
    dl, sampler = data.loader(idx, batch=1, aug=False)
    assert len(list(dl)) == 1


@pytest.mark.parametrize('fault', ['index', 'split', 'archive', 'geometry', 'dimensions', 'hash', 'allowed_type'])
def test_prepare_bad_manifest_refuses_before_any_output(source_fixture, tmp_path, fault):
    base, manifest, archives = source_fixture
    rows = json.loads(manifest.read_text())
    r = rows[0]
    if fault == 'index':
        r['index'] = 4
    elif fault == 'split':
        r['split'] = 'validation'
    elif fault == 'archive':
        r['archive'] = 'different.zip'
    elif fault == 'geometry':
        r['geometry']['left'] = 0.5
    elif fault == 'dimensions':
        r['geometry']['original_width'] = 100
    elif fault == 'hash':
        r['sha256'] = '0' * 64
    else:
        r['allowed'] = 'false'
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        preparation.prepare(
            path,
            REPO_ROOT / 'configs/data_definition.json',
            tmp_path / 'run',
            tmp_path / 'cache',
            archives,
            REPO_ROOT / 'configs/sevenplane_protocol.json',
        )
    assert not (tmp_path / 'run').exists() and not (tmp_path / 'cache').exists()


@pytest.mark.parametrize(
    'polygon', ['0 .2 .2 .8', '0 nan .2 .8 .2 .8 .8', '-1 .2 .2 .8 .2 .8 .8', '0 .2 .2 1.1 .2 .8 .8']
)
def test_invalid_polygon_does_not_publish_cache(source_fixture, tmp_path, polygon):
    base, manifest, archives = source_fixture
    rows = json.loads(manifest.read_text())
    row = rows[0]
    custom = tmp_path / 'archives'
    custom.mkdir()
    with zipfile.ZipFile(archives / row['archive']) as src, zipfile.ZipFile(custom / row['archive'], 'w') as dst:
        dst.writestr(row['name'], src.read(row['name']))
        dst.writestr(row['label'], polygon)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        preparation.prepare(
            path,
            REPO_ROOT / 'configs/data_definition.json',
            tmp_path / 'run',
            tmp_path / 'cache',
            custom,
            REPO_ROOT / 'configs/sevenplane_protocol.json',
        )
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize(
    'metadata',
    [
        {},
        dict(gray_mean=0.4, gray_std=0),
        dict(gray_mean=float('nan'), gray_std=0.2),
        dict(gray_mean=[0.4, 0.5], gray_std=0.2),
    ],
)
def test_bad_initialization_metadata_never_writes(tmp_path, metadata):
    path = tmp_path / 'meta.json'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        preparation.initialize(tmp_path / 'init', tmp_path / 'absent_encoder.pt', path)
    assert not (tmp_path / 'init').exists()


@pytest.mark.parametrize('score', [True, 1.2])
def test_classifier_quality_rejects_non_probability_values(tmp_path, score):
    folder = tmp_path / 'full/classifier'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'x')
    (folder / 'result.json').write_text(json.dumps(dict(seed=17, macro_accuracy=score)))
    assert cli.existing_model(tmp_path, 'classifier')['action'] == 'blocked_quality_review'


def test_segmentation_quality_requires_boolean_true(tmp_path):
    folder = tmp_path / 'full/plain'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'x')
    (folder / 'result.json').write_text(json.dumps(dict(seed=17, learning_sufficiency_pass='false')))
    assert cli.existing_model(tmp_path, 'plain')['action'] == 'blocked_quality_review'


def test_classifier_stop_marker_precedes_resources_or_output(monkeypatch, tmp_path):
    (tmp_path / 'STOP_REQUESTED.json').write_text('{}')
    monkeypatch.setattr(classification, 'ROOT', tmp_path)
    resources = Mock()
    monkeypatch.setattr(classification, 'resources', resources)
    with pytest.raises(AssertionError):
        classification.train_classifier()
    resources.assert_not_called()
    assert not (tmp_path / 'full').exists()


def test_segmentation_stop_marker_precedes_resources(monkeypatch, tmp_path):
    (tmp_path / 'STOP_REQUESTED.json').write_text('{}')
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    resources = Mock()
    monkeypatch.setattr(training, 'resources', resources)
    with pytest.raises(AssertionError):
        training.train_seg('plain', 'plain')
    resources.assert_not_called()


@pytest.mark.parametrize('scenario', ['ok', 'memory', 'duplicate', 'process_query_error'])
def test_resource_guards_without_launching_any_process(monkeypatch, tmp_path, scenario):
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    pid = core.os.getpid()
    ps = [
        dict(ProcessId=pid, ParentProcessId=1, CommandLine='this test'),
        dict(ProcessId=1, ParentProcessId=0, CommandLine='parent'),
    ]
    if scenario == 'duplicate':
        ps.append(dict(ProcessId=pid + 999, ParentProcessId=1, CommandLine='plane-seg stage3-baseline --execute'))
    query = Mock(
        side_effect=lambda args, **kw: (
            ''
            if '--query-compute-apps=pid' in args
            else ('6000, 12000, 20' if scenario == 'memory' else '1000, 12000, 20')
        )
    )
    monkeypatch.setattr(core.subprocess, 'check_output', query)
    monkeypatch.setattr(
        core.subprocess,
        'run',
        lambda *a, **k: types.SimpleNamespace(
            returncode=1 if scenario == 'process_query_error' else 0, stdout=json.dumps(ps), stderr='query failed'
        ),
    )
    if scenario == 'ok':
        assert core.resources()['gpu_used_mib'] == 1000
    else:
        with pytest.raises(AssertionError):
            core.resources()
        assert not (tmp_path / 'last_resource_check.json').exists()
    if scenario in ['duplicate', 'process_query_error']:
        query.assert_not_called()


def test_atomic_dump_event_and_seed_are_isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    common.write_json(tmp_path / 'result.json', dict(v=2))
    assert json.loads((tmp_path / 'result.json').read_text()) == dict(v=2)
    assert not (tmp_path / 'result.json.tmp').exists()
    core.event('SYNTHETIC_QA', count=1)
    assert json.loads((tmp_path / 'events.jsonl').read_text())['kind'] == 'SYNTHETIC_QA'
    core.seed()
    first = torch.rand(3)
    core.seed()
    torch.testing.assert_close(first, torch.rand(3))


def test_evaluation_requires_lock_for_existing_results(monkeypatch, tmp_path):
    registry_fixtures.RegistryTests().fixture(tmp_path)
    (tmp_path / 'FINAL_MODEL_LOCK.json').unlink()
    monkeypatch.setattr(cli, 'REPO_ROOT', tmp_path)
    with pytest.raises(ValueError):
        cli.evaluate_stage(tmp_path, True)


def test_fresh_evaluation_freezes_first_and_partial_replay_is_blocked(monkeypatch, tmp_path):
    registry_fixtures.RegistryTests().fixture(tmp_path)
    (tmp_path / 'final_evaluation/summary.json').unlink()
    (tmp_path / 'final_evaluation/REPORT.md').unlink()
    (tmp_path / 'final_evaluation').rmdir()
    (tmp_path / 'FINAL_MODEL_LOCK.json').unlink()
    monkeypatch.setattr(cli, 'REPO_ROOT', tmp_path)

    def fake_eval():
        lock = json.loads((tmp_path / 'FINAL_MODEL_LOCK.json').read_text())
        assert len(lock['checkpoints']) == 11
        assert all(
            digest == hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
            for path, digest in lock['checkpoints'].items()
        )
        assert lock['study_plan_sha256'] == hashlib.sha256((tmp_path / 'study_plan.json').read_bytes()).hexdigest()
        assert lock['cohort_sha256'] == hashlib.sha256((tmp_path / 'cohort.json').read_bytes()).hexdigest()
        assert 'repository_source_hashes_sha256' not in lock
        assert not (tmp_path / 'configs').exists()
        (tmp_path / 'final_evaluation').mkdir()
        raise RuntimeError('synthetic evaluator stopped')

    module = types.ModuleType('plane_conditioned_segmentation.evaluation.final')
    module.main = fake_eval
    monkeypatch.setitem(__import__('sys').modules, 'plane_conditioned_segmentation.evaluation.final', module)
    assert cli.evaluate_stage(tmp_path)['read_only']
    with pytest.raises(RuntimeError):
        cli.evaluate_stage(tmp_path, True)
    assert not (tmp_path / '.local/execution.lock').exists()
    with pytest.raises(ValueError):
        cli.evaluate_stage(tmp_path, True)


def test_legacy_lock_source_record_is_compatible_without_source_list(monkeypatch, tmp_path):
    registry_fixtures.RegistryTests().fixture(tmp_path)
    path = tmp_path / 'FINAL_MODEL_LOCK.json'
    lock = json.loads(path.read_text())
    lock['repository_source_hashes_sha256'] = 'historical-record-only'
    path.write_text(json.dumps(lock))
    before = (tmp_path / 'final_evaluation/summary.json').read_bytes()
    monkeypatch.setattr(cli, 'REPO_ROOT', tmp_path)
    assert cli.evaluate_stage(tmp_path, True)['action'] == 'reuse_frozen_results'
    assert (tmp_path / 'final_evaluation/summary.json').read_bytes() == before
    assert not (tmp_path / 'configs').exists()
