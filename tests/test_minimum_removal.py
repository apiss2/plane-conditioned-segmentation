import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from plane_conditioned_segmentation.paths import REPO_ROOT
from plane_conditioned_segmentation.training.control import EarlyStopping, load_policy


def fixture(tmp_path):
    path = tmp_path / 'study_plan.json'
    legacy = json.loads((REPO_ROOT / 'configs/sixplane_radam_protocol.json').read_text())
    legacy['schedulefree'].update(min_epochs=60, min_applied_updates=1500)
    path.write_text(json.dumps(legacy))
    review = dict(
        base_protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        models=['expert_3'],
        changes=dict(min_epochs=1, min_applied_updates=0),
    )
    return path, review


def test_override_is_scoped_and_does_not_change_base_or_optimizer(tmp_path):
    path, review = fixture(tmp_path)
    base = load_policy(path)
    (tmp_path / 'stopping_policy_override.json').write_text(json.dumps(review))
    effective = load_policy(path, 'expert_3')
    assert effective == dataclasses.replace(base, min_epochs=1, min_applied_updates=0)
    assert load_policy(path) == load_policy(path, 'expert_0') == base
    assert effective.patience == 30 and effective.max_epochs == 200


@pytest.mark.parametrize('kind', ['protocol', 'lr_or_patience'])
def test_override_cannot_change_other_conditions(tmp_path, kind):
    path, review = fixture(tmp_path)
    if kind == 'protocol':
        review['base_protocol_sha256'] = 'different'
    else:
        review['changes']['patience'] = 31
    (tmp_path / 'stopping_policy_override.json').write_text(json.dumps(review))
    with pytest.raises(ValueError):
        load_policy(path, 'expert_3')


def test_exact30_nonimprovements_with_no_extra_epoch_or_update_budget(tmp_path):
    path, review = fixture(tmp_path)
    (tmp_path / 'stopping_policy_override.json').write_text(json.dumps(review))
    tracker = EarlyStopping(load_policy(path, 'expert_3'))
    for epoch in range(1, 31):
        assert tracker.observe(0.5, epoch, 0)['stop'] is False
    decision = tracker.observe(0.5, 31, 0)
    assert decision['stop'] and decision['reason'] == 'validation_patience' and decision['bad_checks'] == 30
    assert tracker.stop_reason() == 'validation_patience'


def test_transition_keeps_counter_and_rejects_unapproved_changes(tmp_path):
    path, review = fixture(tmp_path)
    base = load_policy(path)
    old = EarlyStopping(base)
    for epoch in range(1, 35):
        old.observe(0.5, epoch, 100)
    state = old.state_dict()
    (tmp_path / 'stopping_policy_override.json').write_text(json.dumps(review))
    new = EarlyStopping(load_policy(path, 'expert_3'))
    with pytest.raises(ValueError):
        new.load_state_dict(state)
    new.load_state_dict(state, allow_minimum_removal=True)
    assert new.bad_checks == 33 and new.best_epoch == 1 and new.last_epoch == 34
    assert new.stop_reason() == 'validation_patience'
    bad = {**state, 'policy': {**state['policy'], 'patience': 31}}
    with pytest.raises(ValueError):
        new.load_state_dict(bad, allow_minimum_removal=True)


def test_restored_maximum_is_terminal_without_observing_an_extra_epoch(tmp_path):
    path, _ = fixture(tmp_path)
    tracker = EarlyStopping(load_policy(path))
    tracker.last_epoch = 200
    tracker.last_updates = 10
    assert tracker.stop_reason() == 'max_epochs'
