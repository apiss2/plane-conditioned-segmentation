"""Real optimizer serialization, next-update equivalence, and isolated copy/overwrite guards."""

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from plane_conditioned_segmentation.common import read_json, sha256_file, write_json
from plane_conditioned_segmentation.config import load_protocol
from plane_conditioned_segmentation.paths import REPO_ROOT
from plane_conditioned_segmentation.pipeline.continuation import (
    INHERITED,
    SHARED,
    _assert_equal,
    load_policy_dict,
    prepare,
    transition_checkpoint,
)
from plane_conditioned_segmentation.training.control import EarlyStopping, load_policy
from plane_conditioned_segmentation.training.schedulefree_state import (
    make_optimizer,
    restore_state,
    save_state,
    set_mode,
)


def checkpoint(tmp_path, label='plain', source_epoch=200):
    path = tmp_path / 'study_plan.json'
    plan = read_json(REPO_ROOT / 'configs/sixplane_radam_protocol.json')
    plan['schedulefree'].update(
        max_epochs=source_epoch,
        min_epochs=60 if source_epoch == 200 else 1,
        min_applied_updates=1500 if source_epoch == 200 else 0,
    )
    write_json(path, plan)
    old = load_policy(path)
    new = dataclasses.replace(old, max_epochs=300 if source_epoch == 200 else 500, min_epochs=1, min_applied_updates=0)
    torch.manual_seed(17)
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2))
    groups = [dict(params=m[0].parameters(), lr=1e-5), dict(params=m[2].parameters(), lr=1e-3)]
    opt = make_optimizer(groups, load_protocol(path), old)
    for _ in range(8):
        step(m, opt)
    tracker = EarlyStopping(old)
    history = []
    for e in range(1, source_epoch + 1):
        score = 0.2 + min(e, source_epoch - 2) / 500
        history.append(dict(epoch=e, validation=dict(selection=score), early_stopping=tracker.observe(score, e, 8)))
    config = dict(label=label, protocol_sha256='old', policy=plan['schedulefree'], encoder_lr=1e-5, decoder_lr=1e-3)
    save_state(
        tmp_path / 'origin.pt',
        m,
        opt,
        label=label,
        mode=label,
        target=None,
        target_count=6,
        resumable=True,
        optimizer_name='RAdamScheduleFree',
        protocol_sha256='old',
        epoch=source_epoch,
        next_epoch=source_epoch,
        completed_batches=0,
        history=history,
        early_stopping=tracker.state_dict(),
        attempted=8,
        applied=8,
        scaler={},
        seconds=1.0,
    )
    ck = torch.load(tmp_path / 'origin.pt', map_location='cpu', weights_only=True)
    return ck, config, history, new, m, opt


def step(m, opt):
    set_mode(m, opt, True)
    opt.zero_grad()
    loss = m(torch.ones(2, 4)).square().mean()
    loss.backward()
    opt.step()
    return loss.detach().item()


@pytest.mark.parametrize('label', SHARED)
@pytest.mark.parametrize('source_epoch', [200, 300])
def test_transition_preserves_all_numeric_state_and_next_actual_updates(tmp_path, label, source_epoch):
    ck, config, h, new, m, opt = checkpoint(tmp_path, label, source_epoch)
    original = copy.deepcopy(ck)
    out = transition_checkpoint(ck, config, h, label=label, old_sha='old', new_sha='new', policy=new)
    _assert_equal(ck, original)
    assert out['early_stopping']['bad_checks'] == 2 and out['early_stopping']['best_epoch'] == source_epoch - 2
    torch.save(out, tmp_path / 'new.pt')
    # Same resume tensors and RNG must yield the same stochastic dropout updates.
    losses = []
    states = []
    for file, sha in [('origin.pt', 'old'), ('new.pt', 'new')]:
        restore_state(tmp_path / file, m, opt, protocol_sha256=sha, label=label)
        losses.append([step(m, opt) for _ in range(4)])
        set_mode(m, opt, False)
        states.append(copy.deepcopy(m.state_dict()))
    assert losses[0] == losses[1]
    _assert_equal(states[0], states[1])
    tracker = EarlyStopping(new)
    tracker.load_state_dict(out['early_stopping'])
    assert tracker.stop_reason() is None
    for e in range(source_epoch + 1, source_epoch + 28):
        assert not tracker.observe(tracker.best, e, 8)['stop']
    assert tracker.observe(tracker.best, source_epoch + 28, 8)['reason'] == 'validation_patience'
    assert tracker.best_epoch == source_epoch - 2


@pytest.mark.parametrize(
    'mutation',
    ['label', 'optimizer', 'train_mode', 'partial', 'epoch', 'history', 'lr', 'policy', 'nonfinite', 'counter'],
)
def test_transition_rejects_incompatible_or_damaged_inputs(tmp_path, mutation):
    ck, config, h, new, _, _ = checkpoint(tmp_path)
    if mutation == 'label':
        ck['label'] = 'film'
    elif mutation == 'optimizer':
        ck['optimizer_name'] = 'AdamW'
    elif mutation == 'train_mode':
        ck['optimizer']['param_groups'][0]['train_mode'] = True
    elif mutation == 'partial':
        ck['completed_batches'] = 1
    elif mutation == 'epoch':
        ck['next_epoch'] = 199
    elif mutation == 'history':
        h = copy.deepcopy(h)
        h[-1]['epoch'] = 199
    elif mutation == 'lr':
        ck['optimizer']['param_groups'][0]['lr'] = 0.1
    elif mutation == 'policy':
        new = dataclasses.replace(new, patience=31)
    elif mutation == 'counter':
        ck['early_stopping']['bad_checks'] = 0
    else:
        next(iter(ck['optimizer']['state'].values()))['z'].fill_(float('nan'))
    with pytest.raises(ValueError):
        transition_checkpoint(ck, config, h, label='plain', old_sha='old', new_sha='new', policy=new)


@pytest.mark.parametrize('source_epoch', [200, 300])
def test_prepare_copies_independently_and_refuses_existing_destination(tmp_path, monkeypatch, source_epoch):
    source = tmp_path / 'origin'
    source.mkdir()
    ck, config, h, new_policy, _, _ = checkpoint(source, source_epoch=source_epoch)
    plan = read_json(source / 'study_plan.json')
    oldsha = sha256_file(source / 'study_plan.json')
    write_json(source / 'confirmed_targets.json', dict(targets=[dict(id=i) for i in range(6)]))
    write_json(source / 'cohort.json', {'allowed_indices': dict(train=[0], val=[1], test=[2])})
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text('[]')
    (cache / 'split.npy').write_bytes(b'fixture')
    (tmp_path / 'archives').mkdir()
    (tmp_path / 'init').mkdir()
    write_json(
        source / 'data_references.json',
        dict(
            immutable_data_source='../cache',
            source_archives='../archives',
            initialization_source='../init',
            manifest_sha256=sha256_file(cache / 'manifest.json'),
            split_sha256=sha256_file(cache / 'split.npy'),
            cohort_sha256=sha256_file(source / 'cohort.json'),
        ),
    )
    locks = {}
    for name in [*INHERITED, *SHARED]:
        folder = source / 'full' / name
        folder.mkdir(parents=True)
        (folder / 'best.pt').write_bytes(('immutable-' + name).encode())
        write_json(folder / 'result.json', dict(epochs=source_epoch, best_epoch=source_epoch - 2))
        locks[str((folder / 'best.pt').relative_to(source))] = sha256_file(folder / 'best.pt')
        if name in SHARED:
            c = copy.deepcopy(ck)
            c.update(label=name, mode=name, protocol_sha256=oldsha)
            cfg = {**config, 'label': name, 'protocol_sha256': oldsha, 'epochs': source_epoch}
            torch.save(c, folder / 'resume.pt')
            write_json(folder / 'configuration.json', cfg)
            write_json(folder / 'history.json', h)
    write_json(source / 'FINAL_MODEL_LOCK.json', dict(checkpoints=locks))
    write_json(source / 'COMPARISON_COMPLETE.json', dict(done=True))
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    dest = tmp_path / 'continued'
    assert prepare(dest, source, max_epochs=new_policy.max_epochs)['continued_models'] == list(SHARED)
    assert not dest.exists()
    # Real files and optimizer states are used; no train/evaluate call is involved.
    prepare(dest, source, True, max_epochs=new_policy.max_epochs)
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    for name in SHARED:
        assert not (dest / f'full/{name}/result.json').exists()
        state = torch.load(dest / f'full/{name}/resume.pt', weights_only=True)
        assert (
            state['next_epoch'] == source_epoch
            and state['early_stopping']['policy']['max_epochs'] == new_policy.max_epochs
        )
        assert (dest / f'full/{name}/best.pt').read_bytes() == before[source / f'full/{name}/best.pt']
    changed = dest / 'full/classifier/best.pt'
    changed.write_bytes(b'changed-copy')
    assert (source / 'full/classifier/best.pt').read_bytes() == before[source / 'full/classifier/best.pt']
    with pytest.raises(ValueError, match='exists'):
        prepare(dest, source, True, max_epochs=new_policy.max_epochs)


@pytest.mark.parametrize('bad_checks,remaining', [(1, 29), (17, 13), (0, 30)])
def test_500_continuation_carries_real_patience_state_without_reset(tmp_path, bad_checks, remaining):
    ck, config, _, policy, _, _ = checkpoint(tmp_path, source_epoch=300)
    old_tracker = EarlyStopping(load_policy_dict(config['policy']))
    history = []
    for epoch in range(1, 301):
        score = 0.2 + min(epoch, 300 - bad_checks) / 500
        history.append(
            dict(epoch=epoch, validation=dict(selection=score), early_stopping=old_tracker.observe(score, epoch, 8))
        )
    ck['history'] = history
    ck['early_stopping'] = old_tracker.state_dict()
    converted = transition_checkpoint(ck, config, history, label='plain', old_sha='old', new_sha='new', policy=policy)
    tracker = EarlyStopping(policy)
    tracker.load_state_dict(converted['early_stopping'])
    assert tracker.bad_checks == bad_checks and tracker.stop_reason() is None
    for epoch in range(301, 300 + remaining):
        assert not tracker.observe(tracker.best, epoch, 8)['stop']
    assert tracker.observe(tracker.best, 300 + remaining, 8)['reason'] == 'validation_patience'
    assert tracker.bad_checks == 30


def test_500_improvement_resets_patience_and_hard_cap_still_stops(tmp_path):
    ck, config, history, policy, _, _ = checkpoint(tmp_path, source_epoch=300)
    converted = transition_checkpoint(ck, config, history, label='plain', old_sha='old', new_sha='new', policy=policy)
    tracker = EarlyStopping(policy)
    tracker.load_state_dict(converted['early_stopping'])
    assert not tracker.observe(tracker.best + 0.001, 301, 8)['stop']
    assert tracker.bad_checks == 0 and tracker.best_epoch == 301
    for epoch in range(302, 331):
        assert not tracker.observe(tracker.best, epoch, 8)['stop']
    assert tracker.observe(tracker.best, 331, 8)['reason'] == 'validation_patience'
    # An improving validation sequence can reach, but never exceed, the fixed 500 cap.
    tracker = EarlyStopping(policy)
    tracker.load_state_dict(converted['early_stopping'])
    initial_best = tracker.best
    for epoch in range(301, 500):
        assert not tracker.observe(initial_best + (epoch - 300) / 2000, epoch, 8)['stop']
    assert tracker.observe(0.9, 500, 8)['reason'] == 'max_epochs'
    with pytest.raises(ValueError):
        tracker.observe(0.91, 501, 8)


def test_unapproved_source_to_500_or_other_cap_is_rejected(tmp_path):
    ck, config, history, policy, _, _ = checkpoint(tmp_path)
    for cap in [500, 600]:
        with pytest.raises(ValueError):
            transition_checkpoint(
                ck,
                config,
                history,
                label='plain',
                old_sha='old',
                new_sha='new',
                policy=dataclasses.replace(policy, max_epochs=cap),
            )
