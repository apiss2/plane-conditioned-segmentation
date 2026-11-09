"""Real Schedule-Free state transitions, checkpoint/restart and validation-only stop boundaries."""

import dataclasses
import io
import json

import pytest
import torch

from plane_conditioned_segmentation.config import load_protocol
from plane_conditioned_segmentation.paths import REPO_ROOT
from plane_conditioned_segmentation.training.control import EarlyStopping, load_policy
from plane_conditioned_segmentation.training.schedulefree_state import (
    make_optimizer,
    restore_state,
    save_state,
    set_mode,
)


def policy(**kw):
    return dataclasses.replace(load_policy(REPO_ROOT / 'configs/sixplane_radam_protocol.json'), **kw)


def protocol():
    return load_protocol(REPO_ROOT / 'configs/sixplane_radam_protocol.json')


def test_frozen_radam_protocol_matches_requested_conditions():
    p = policy()
    pr = protocol()
    assert (p.optimizer, p.package_version, p.max_epochs, p.patience, p.min_delta) == (
        'RAdamScheduleFree',
        '1.4.1',
        200,
        30,
        0.0,
    )
    assert pr.optimization.validation_every == 1
    assert pr.learning_rates.encoder == 1e-5 and pr.learning_rates.decoder == 1e-3
    assert p.min_epochs == 1 and p.min_applied_updates == 0
    assert load_policy(REPO_ROOT / 'configs/sevenplane_protocol.json') is None
    assert pr.augmentation == load_protocol(REPO_ROOT / 'configs/sixplane_protocol.json').augmentation


def test_patience_exact_thirtieth_tie_and_finite_improvement_reset():
    stop = EarlyStopping(policy(min_epochs=1, min_applied_updates=0))
    assert stop.observe(0.5, 1, 1)['selected']
    for epoch in range(2, 31):
        r = stop.observe(0.5, epoch, epoch)
        assert not r['stop'] and not r['selected']
    r = stop.observe(0.5, 31, 31)
    assert r['stop'] and r['bad_checks'] == 30 and r['reason'] == 'validation_patience'
    stop = EarlyStopping(policy(min_epochs=1, min_applied_updates=0))
    stop.observe(0.5, 1, 1)
    for e in range(2, 30):
        stop.observe(0.49, e, e)
    assert stop.observe(0.500001, 30, 30)['selected'] and stop.bad_checks == 0
    assert stop.best_epoch == 30


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.1, 1.1, True, None, '0.8'])
def test_nonfinite_or_invalid_score_aborts_without_counting(bad):
    tracker = EarlyStopping(policy())
    tracker.observe(0.5, 1, 50)
    before = tracker.state_dict()
    with pytest.raises(ValueError):
        tracker.observe(bad, 2, 100)
    assert tracker.state_dict() == before


def test_minimum_epoch_and_applied_updates_and_hard_cap():
    tracker = EarlyStopping(policy(min_epochs=60, min_applied_updates=1500))
    for e in range(1, 61):
        r = tracker.observe(0.7, e, e)
        assert not r['stop']
    assert tracker.observe(0.7, 61, 1500)['stop']
    tracker = EarlyStopping(policy(min_applied_updates=1000000))
    for e in range(1, 201):
        r = tracker.observe(0.7, e, e)
    assert r['stop'] and r['reason'] == 'max_epochs' and not r['eligible']


def test_min_delta_reference_separate_from_exact_checkpoint_selection():
    tracker = EarlyStopping(policy(min_delta=0.01, min_epochs=1, min_applied_updates=0))
    tracker.observe(0.5, 1, 1)
    r = tracker.observe(0.505, 2, 2)
    assert r['selected'] and tracker.reference == 0.5 and r['bad_checks'] == 1
    r = tracker.observe(0.52, 3, 3)
    assert r['selected'] and tracker.reference == 0.52 and r['bad_checks'] == 0


def test_tracker_torch_serialization_and_policy_mismatch(tmp_path):
    p = policy()
    a = EarlyStopping(p)
    for e in range(1, 12):
        a.observe(0.5, e, 100 * e)
    file = tmp_path / 'tracker.pt'
    torch.save(a.state_dict(), file)
    b = EarlyStopping(p)
    b.load_state_dict(torch.load(file, weights_only=True))
    assert a.observe(0.5, 12, 1200) == b.observe(0.5, 12, 1200)
    with pytest.raises(ValueError):
        EarlyStopping(policy(patience=31)).load_state_dict(a.state_dict())


def tiny():
    # Real stochastic forward exercises saved RNG, not a placeholder optimizer mock.
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(8, 2))


def step(model, opt):
    set_mode(model, opt, True)
    opt.zero_grad(set_to_none=True)
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0, 1]))
    loss.backward()
    opt.step()
    return float(loss.detach())


def new_opt(model):
    return make_optimizer([dict(params=model.parameters(), lr=0.001)], protocol(), policy())


def test_real_optimizer_requires_train_and_eval_checkpoint_exact_reload(tmp_path):
    m = tiny()
    opt = new_opt(m)
    with pytest.raises(Exception, match='train mode'):
        opt.step()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    for _ in range(8):
        step(m, opt)
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())
    set_mode(m, opt, False)
    x = torch.ones(3, 4)
    with torch.no_grad():
        expected = m(x).clone()
    file = tmp_path / 'best.pt'
    save_state(file, m, opt, resumable=False, epoch=8)
    ck = torch.load(file, weights_only=True)
    loaded = tiny().eval()
    loaded.load_state_dict(ck['state'])
    assert ck['weight_sequence'] == 'schedulefree_eval_x' and 'optimizer' not in ck
    assert not any(g['train_mode'] for g in opt.param_groups)
    with torch.no_grad():
        assert torch.equal(expected, loaded(x))
    # Repeating eval is idempotent and cannot silently re-transform the saved parameter sequence.
    set_mode(m, opt, False)
    with torch.no_grad():
        assert torch.equal(expected, m(x))


def test_real_optimizer_resume_matches_uninterrupted_next_updates_and_rng(tmp_path):
    torch.manual_seed(17)
    a = tiny()
    oa = new_opt(a)
    for _ in range(8):
        step(a, oa)
    path = tmp_path / 'resume.pt'
    save_state(path, a, oa, resumable=True, optimizer_name='RAdamScheduleFree', protocol_sha256='fixed', label='plain')
    expected_losses = [step(a, oa) for _ in range(5)]
    set_mode(a, oa, False)
    expected = {k: v.detach().clone() for k, v in a.state_dict().items()}
    b = tiny()
    ob = new_opt(b)
    state = restore_state(path, b, ob, protocol_sha256='fixed', label='plain')
    actual_losses = [step(b, ob) for _ in range(5)]
    set_mode(b, ob, False)
    assert expected_losses == actual_losses
    assert all(torch.equal(expected[k], v) for k, v in b.state_dict().items())
    assert [g['k'] for g in oa.param_groups] == [g['k'] for g in ob.param_groups]
    assert all(not g['train_mode'] for g in state['optimizer']['param_groups'])
    for sha, label in [('changed', 'plain'), ('fixed', 'expert_0')]:
        with pytest.raises(ValueError):
            restore_state(path, b, ob, protocol_sha256=sha, label=label)


@pytest.mark.parametrize(
    'change',
    [
        dict(max_epochs=0),
        dict(patience=0),
        dict(min_epochs=201),
        dict(min_delta=-1),
        dict(betas=(1.0, 0.9)),
        dict(eps=float('nan')),
    ],
)
def test_invalid_policy_fails_before_training(change):
    with pytest.raises(ValueError):
        policy(**change)


def test_adam_and_training_weight_checkpoint_cannot_masquerade_as_resume(tmp_path):
    model = tiny()
    opt = new_opt(model)
    p = tmp_path / 'bad.pt'
    torch.save(
        dict(state=model.state_dict(), optimizer=torch.optim.AdamW(model.parameters()).state_dict(), resumable=True), p
    )
    with pytest.raises(ValueError):
        restore_state(p, model, opt, protocol_sha256='fixed', label='plain')
