"""Real saved-state I/O contracts; no medical data, network or optimizer execution."""

import hashlib
import json
from pathlib import Path

import pytest

from plane_conditioned_segmentation.config import run_protocol
from plane_conditioned_segmentation.paths import REPO_ROOT
from plane_conditioned_segmentation.pipeline import stages
from plane_conditioned_segmentation.pipeline.provisional import prepare
from plane_conditioned_segmentation.training.sufficiency import assess


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(root):
    root.mkdir()
    plan = json.loads((REPO_ROOT / 'configs/sixplane_radam_protocol.json').read_text())
    plan.update(provisional_reference_experts='old', segmentation_counts=[6] * 3, classification_counts=[6] * 3)
    write(root / 'study_plan.json', plan)
    write(root / 'confirmed_targets.json', {'targets': [{'id': i} for i in range(6)]})
    write(root / 'cohort.json', {'allowed_indices': {'train': [1], 'val': [2], 'test': [3]}})
    write(
        root / 'data_references.json',
        dict(
            manifest_sha256='m',
            split_sha256='s',
            cohort_sha256='c',
            immutable_data_source='../cache',
            source_archives='../archives',
            initialization_source='../init',
        ),
    )
    return root


def model(root, name, failed=False, **overrides):
    plan = json.loads((root / 'study_plan.json').read_text())
    history = [
        dict(
            epoch=e,
            train=dict(macro_dice=0.8, macro_global_dice=0.8, per_target={str(i): {'dice': 0.8} for i in range(6)}),
            validation={'selection': 0.5 if e <= 140 else 0.53},
        )
        for e in range(1, 201)
    ]
    value = dict(
        seed=17,
        macro_accuracy=0.8,
        learning_sufficiency_pass=not failed,
        optimizer_name='RAdamScheduleFree',
        epochs=200,
        best_epoch=200,
        stop_reason='max_epochs',
        policy=plan['schedulefree'],
        protocol_sha256=sha(root / 'study_plan.json'),
        terminal_learning_assessment=assess(history, run_protocol(root).sufficiency_gate),
        **overrides,
    )
    folder = root / 'full' / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'best.pt').write_bytes(('fixture-weight-' + name).encode())
    write(folder / 'history.json', history)
    write(folder / 'result.json', value)


def approve(root):
    write(
        root / 'budget_cap_review.json',
        dict(
            allow_budget_cap_nonconvergence=True,
            models=['condconv', *[f'expert_{i}' for i in range(6)]],
            maximum_epochs=200,
            run_protocol_sha256=sha(root / 'study_plan.json'),
            training_protocol_sha256=sha(root / 'study_plan.json'),
            fixed_existing_checkpoints={'condconv': sha(root / 'full/condconv/best.pt')},
        ),
    )


def test_explicit_review_reuses_fixed_model_but_preserves_failed_gate_and_bytes(tmp_path, monkeypatch):
    root = setup(tmp_path / 'new')
    for name in ['classifier', 'plain', 'film']:
        model(root, name)
    model(root, 'condconv', True)
    assert stages.existing_model(root, 'condconv')['action'] == 'blocked_quality_review'
    approve(root)
    protected = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    state = stages.existing_model(root, 'condconv')
    assert state['action'] == 'reuse' and 'remains FAILED' in state['quality_note']
    stages.prerequisites(root, 'stage2-experts')
    monkeypatch.setattr(stages, 'REPO_ROOT', root)

    def must_not_train(_):
        raise AssertionError('An existing model must not be trained')

    monkeypatch.setattr(stages, '_train', must_not_train)
    stages.execute_stage(root, 'stage5-condconv')
    assert all(p.read_bytes() == data for p, data in protected.items())
    assert json.loads((root / 'full/condconv/result.json').read_text())['learning_sufficiency_pass'] is False


@pytest.mark.parametrize(
    'kind',
    [
        'not_approved',
        '199epochs',
        'patience',
        'wrong_optimizer',
        'low_train',
        'low_global',
        'low_target',
        'nan',
        'incomplete',
        'tampered_checkpoint',
        'different_protocol',
    ],
)
def test_review_does_not_admit_ineligible_states(tmp_path, kind):
    root = setup(tmp_path / 'new')
    model(root, 'condconv', True)
    approve(root)
    path = root / 'full/condconv/result.json'
    value = json.loads(path.read_text())
    history_path = root / 'full/condconv/history.json'
    history = json.loads(history_path.read_text())
    if kind == 'not_approved':
        review = json.loads((root / 'budget_cap_review.json').read_text())
        review['allow_budget_cap_nonconvergence'] = False
        write(root / 'budget_cap_review.json', review)
    elif kind == '199epochs':
        value['epochs'] = 199
    elif kind == 'patience':
        value['stop_reason'] = 'validation_patience'
    elif kind == 'wrong_optimizer':
        value['optimizer_name'] = 'AdamW'
    elif kind == 'tampered_checkpoint':
        (root / 'full/condconv/best.pt').write_bytes(b'changed')
    elif kind == 'different_protocol':
        review = json.loads((root / 'budget_cap_review.json').read_text())
        review['run_protocol_sha256'] = 'changed'
        write(root / 'budget_cap_review.json', review)
    elif kind == 'incomplete':
        history.pop(0)
    else:
        if kind == 'low_train':
            history[-1]['train']['macro_dice'] = 0.69
        elif kind == 'low_global':
            history[-1]['train']['macro_global_dice'] = 0.59
        elif kind == 'low_target':
            history[-1]['train']['per_target']['0']['dice'] = 0.49
        elif kind == 'nan':
            history[-1]['validation']['selection'] = float('nan')
        if kind != 'nan':
            value['terminal_learning_assessment'] = assess(history, run_protocol(root).sufficiency_gate)
    write(path, value)
    write(history_path, history)
    if kind in ['nan', 'incomplete', 'tampered_checkpoint', 'different_protocol']:
        with pytest.raises(ValueError):
            stages.existing_model(root, 'condconv')
    else:
        assert stages.existing_model(root, 'condconv')['action'] == 'blocked_quality_review'


def test_provisional_copy_keeps_review_and_false_gate_idempotently(tmp_path, monkeypatch):
    monkeypatch.setenv('PCS_RUNS_DIR', str(tmp_path))
    root = setup(tmp_path / 'new')
    old = setup(tmp_path / 'old')
    for name in ['classifier', 'plain', 'film']:
        model(root, name)
    model(root, 'condconv', True)
    approve(root)
    for i in range(6):
        model(old, f'expert_{i}')
    child = prepare(root)
    assert stages.existing_model(child, 'condconv')['action'] == 'reuse'
    assert json.loads((child / 'full/condconv/result.json').read_text())['learning_sufficiency_pass'] is False
    assert sha(root / 'full/condconv/best.pt') == sha(child / 'full/condconv/best.pt')
    assert prepare(root) == child
    review = json.loads((child / 'budget_cap_review.json').read_text())
    review['maximum_epochs'] = 201
    write(child / 'budget_cap_review.json', review)
    with pytest.raises(ValueError, match='review changed'):
        prepare(root)


def test_experts_continue_after_expected_saved_budget_cap_only(tmp_path, monkeypatch):
    root = setup(tmp_path / 'new')
    for name in ['classifier', 'plain', 'film']:
        model(root, name)
    model(root, 'condconv', True)
    approve(root)
    monkeypatch.setattr(stages, 'REPO_ROOT', root)
    calls = []

    def saved_terminal(name):
        calls.append(name)
        model(root, name, True)
        raise RuntimeError(('Terminal learning sufficiency requires review; thresholds not changed', name))

    monkeypatch.setattr(stages, '_train', saved_terminal)
    result = stages.execute_stage(root, 'stage2-experts')
    assert calls == [f'expert_{i}' for i in range(6)]
    assert all(r['action'] == 'reuse' and 'remains FAILED' in r['quality_note'] for r in result['models'])
    calls.clear()
    stages.execute_stage(root, 'stage2-experts')
    assert calls == []


def test_unrelated_training_error_is_not_swallowed(tmp_path, monkeypatch):
    root = setup(tmp_path / 'new')
    for name in ['classifier', 'plain', 'film']:
        model(root, name)
    model(root, 'condconv', True)
    approve(root)
    monkeypatch.setattr(stages, 'REPO_ROOT', root)

    def unrelated(name):
        model(root, name, True)
        raise RuntimeError('OOM fixture')

    monkeypatch.setattr(stages, '_train', unrelated)
    with pytest.raises(RuntimeError, match='OOM fixture'):
        stages.execute_stage(root, 'stage2-experts')
