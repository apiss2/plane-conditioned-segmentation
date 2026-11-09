"""Unchanged sufficiency diagnostics at authorized continuation stop boundaries."""

import copy
import json

import pytest

from plane_conditioned_segmentation.config import run_protocol
from plane_conditioned_segmentation.pipeline import stages
from plane_conditioned_segmentation.training.control import EarlyStopping, load_policy
from plane_conditioned_segmentation.training.sufficiency import assess
from test_budget_cap import approve, model, setup, sha, write


@pytest.mark.parametrize(
    'case', ['accepted', 'counter29', 'no_manifest', 'not_authorized', 'low_train', 'low_global', 'low_target', 'nan']
)
def test_continuation_patience_diagnostic_keeps_false_gate_and_train_safety(tmp_path, case):
    root = setup(tmp_path / 'continued')
    plan = json.loads((root / 'study_plan.json').read_text())
    plan['schedulefree'].update(max_epochs=300, min_epochs=1, min_applied_updates=0)
    write(root / 'study_plan.json', plan)
    model(root, 'condconv', True)
    folder = root / 'full/condconv'
    value = json.loads((folder / 'result.json').read_text())
    tracker = EarlyStopping(load_policy(root / 'study_plan.json'))
    history = []
    for e in range(1, 241):
        selection = 0.2 + min(e, 210) / 1000
        history.append(
            dict(
                epoch=e,
                train=dict(
                    macro_dice=0.9, macro_global_dice=0.9, per_target={str(i): dict(dice=0.9) for i in range(6)}
                ),
                validation=dict(selection=selection),
                early_stopping=tracker.observe(selection, e, e * 10),
            )
        )
    assert tracker.bad_checks == 30 and tracker.best_epoch == 210
    value.update(epochs=240, best_epoch=210, stop_reason='validation_patience', early_stopping=tracker.state_dict())
    if case == 'counter29':
        value['early_stopping']['bad_checks'] = 29
    if case == 'low_train':
        history[-1]['train']['macro_dice'] = 0.69
    if case == 'low_global':
        history[-1]['train']['macro_global_dice'] = 0.59
    if case == 'low_target':
        history[-1]['train']['per_target']['0']['dice'] = 0.49
    if case == 'nan':
        history[-1]['validation']['selection'] = float('nan')
    if case != 'nan':
        value['terminal_learning_assessment'] = assess(history, run_protocol(root).sufficiency_gate)
    write(folder / 'history.json', history)
    write(folder / 'result.json', value)
    approve(root)
    review = json.loads((root / 'budget_cap_review.json').read_text())
    review.update(maximum_epochs=300, allow_patience_nonconvergence=case != 'not_authorized')
    write(root / 'budget_cap_review.json', review)
    if case != 'no_manifest':
        write(root / 'continuation_preparation.json', dict(new_protocol_sha256=sha(root / 'study_plan.json')))
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    if case == 'nan':
        with pytest.raises(ValueError):
            stages.existing_model(root, 'condconv')
    else:
        result = stages.existing_model(root, 'condconv')
        assert result['action'] == ('reuse' if case == 'accepted' else 'blocked_quality_review')
        if case == 'accepted':
            assert 'remains FAILED' in result['quality_note'] and 'Validation-patience' in result['quality_note']
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert json.loads((folder / 'result.json').read_text())['learning_sufficiency_pass'] is False
