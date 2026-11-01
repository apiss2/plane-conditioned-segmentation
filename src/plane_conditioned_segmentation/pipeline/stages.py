"""Stage plans, prerequisite checks, the execution lock and stage execution over frozen run artifacts."""

import contextlib
import json
import math
import os

from ..common import FULL_MODEL_COUNT, LOCK_FILE, NUM_TARGETS, SEED, utc_now
from ..common import read_json as read
from ..common import sha256_file as sha
from ..config import run_protocol
from ..paths import REPO_ROOT

STAGES = {
    'stage1-classifier': ('classifier',),
    'stage2-experts': tuple(f'expert_{i}' for i in range(NUM_TARGETS)),
    'stage3-baseline': ('plain',),
    'stage4-film': ('film',),
    'stage5-condconv': ('condconv',),
}


def _blocked(name, state, action):
    return dict(name=name, state=state, action=action)


def _classifier_score_ok(value, protocol):
    score = value.get('macro_accuracy')
    minimum = protocol.classifier_rule.min_validation_macro_accuracy
    return type(score) in (int, float) and math.isfinite(score) and minimum <= score <= 1


def _selected_quality_note(value, protocol):
    """Caveat when terminal-history sufficiency passed but the selected checkpoint misses the original gate."""
    if value.get('learning_sufficiency_basis') != 'terminal_training_history':
        return None
    train = value.get('train', {})
    if not train:
        return (
            'Terminal-state sufficiency does not establish selected-checkpoint quality; '
            'saved selected-training metrics are unavailable.'
        )
    gate = protocol.sufficiency_gate
    if (
        train.get('macro_dice', 0) < gate.train_dice
        or train.get('macro_global_dice', 0) < gate.global_dice
        or min((r.get('dice', 0) for r in train.get('per_target', {}).values()), default=0) < gate.minimum_target
    ):
        return (
            'Selected best remains below original selected-checkpoint train/global gate; terminal-state '
            'assessment is a semantic change, not a proven original-intent bug fix.'
        )
    return None


def existing_model(root, name):
    """State of one model folder and the action a stage run would take for it."""
    folder = root / 'full' / name
    result = folder / 'result.json'
    weight = folder / 'best.pt'
    if not folder.exists():
        return _blocked(name, 'not_started', 'train')
    if not result.exists() or not weight.exists():
        return _blocked(name, 'partial', 'blocked_inspect_partial')
    value = read(result)
    if value.get('seed') != SEED:
        raise ValueError(f'{name}: unexpected seed')
    lock = root / LOCK_FILE
    if lock.exists():
        expected = read(lock)['checkpoints'].get(str(weight.relative_to(root)))
        if not expected or sha(weight) != expected:
            raise ValueError(f'{name}: frozen checkpoint hash mismatch')
    protocol = run_protocol(root)
    if name == 'classifier':
        if not _classifier_score_ok(value, protocol):
            return _blocked(name, 'quality_review_required', 'blocked_quality_review')
        note = None
    else:
        if value.get('learning_sufficiency_pass') is not True:
            return _blocked(name, 'quality_review_required', 'blocked_quality_review')
        note = _selected_quality_note(value, protocol)
    return dict(
        name=name,
        state='completed',
        action='reuse',
        epochs=value.get('epochs'),
        best_epoch=value.get('best_epoch'),
        quality_note=note,
    )


def all_model_names():
    return [name for names in STAGES.values() for name in names]


def stage_plan(root, stage, target=None):
    if target is not None and target not in range(NUM_TARGETS):
        raise ValueError('Target must be 0 through 6')
    names = STAGES[stage]
    if stage == 'stage2-experts' and target is not None:
        names = (f'expert_{target}',)
    return dict(
        stage=stage,
        experiment=root.name,
        models=[existing_model(root, name) for name in names],
        default='read_only_plan',
        training_conditions='Recorded seed17/study_plan.json; no test-based tuning',
    )


def prerequisites(root, stage):
    prior = []
    for key, names in STAGES.items():
        if key == stage:
            break
        prior.extend(names)
    for name in prior:
        if existing_model(root, name)['action'] != 'reuse':
            raise ValueError(f'Prerequisite incomplete: {name}')


@contextlib.contextmanager
def execution_lock():
    """Exclusive repository-wide lock; a lock owned by another run is never removed."""
    folder = REPO_ROOT / '.local'
    folder.mkdir(exist_ok=True)
    lock = folder / 'execution.lock'
    try:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), utc=utc_now()), stream)
    except FileExistsError:
        raise ValueError('An execution lock exists; inspect the recorded process before retrying')
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _train(name):
    if name == 'classifier':
        from ..training.classifier import train_classifier

        train_classifier()
    else:
        from ..training.segmentation import train_seg

        target = int(name.split('_')[1]) if name.startswith('expert_') else None
        train_seg(name, 'expert' if target is not None else name, target)


def execute_stage(root, stage, target=None):
    plan = stage_plan(root, stage, target)
    prerequisites(root, stage)
    if any(model['action'].startswith('blocked') for model in plan['models']):
        raise ValueError('Existing partial/quality-failed run requires review; no overwrite or automatic restart')
    with execution_lock():
        for model in plan['models']:
            if model['action'] != 'reuse':
                _train(model['name'])
    return {
        **plan,
        'execution': 'completed_or_reused',
        'models': [existing_model(root, row['name']) for row in plan['models']],
    }


def _reuse_saved_evaluation(root, destination):
    if not (root / LOCK_FILE).is_file():
        raise ValueError('Saved evaluation has no fixed checkpoint lock')
    summary = read(destination)
    if summary.get('seed') != SEED or summary.get('full_training_models') != FULL_MODEL_COUNT:
        raise ValueError('Saved evaluation protocol mismatch')
    manifest = root / 'run_manifest.json'
    if manifest.exists():
        registered = read(manifest).get('artifacts_sha256', {}).get('final_evaluation/summary.json')
        if registered != sha(destination):
            raise ValueError('Registered saved evaluation hash mismatch')
    return dict(
        stage='stage6-evaluate',
        action='reuse_frozen_results',
        models=FULL_MODEL_COUNT,
        test_images=summary['test_segmentation_images'],
        summary=str(destination),
    )


def _write_model_lock(root, names):
    """Freeze checkpoint and protocol hashes before the test set is first touched."""
    weights = [root / 'full' / n / 'best.pt' for n in names]
    # Keys use the same relative-path spelling that existing_model looks up.
    checkpoints = {str(w.relative_to(root)): sha(w) for w in weights}
    record = dict(
        utc=utc_now(),
        full_training_models=FULL_MODEL_COUNT,
        seed=SEED,
        test_never_used_for_checkpoint_selection=True,
        checkpoints=checkpoints,
        study_plan_sha256=sha(root / 'study_plan.json'),
        cohort_sha256=sha(root / 'cohort.json'),
    )
    (root / LOCK_FILE).write_text(json.dumps(record, indent=2), encoding='utf-8')


def evaluate_stage(root, execute=False):
    names = all_model_names()
    if any(existing_model(root, name)['action'] != 'reuse' for name in names):
        raise ValueError('All 11 models must be complete before final test')
    destination = root / 'final_evaluation/summary.json'
    if destination.exists():
        return _reuse_saved_evaluation(root, destination)
    if not execute:
        return dict(
            stage='stage6-evaluate', action='lock_checkpoints_then_evaluate', models=FULL_MODEL_COUNT, read_only=True
        )
    with execution_lock():
        if (root / 'final_evaluation').exists():
            raise ValueError('Partial final evaluation exists; inspect before continuing')
        if not (root / LOCK_FILE).exists():
            _write_model_lock(root, names)
        from ..evaluation.final import main as evaluate

        evaluate()
    return dict(stage='stage6-evaluate', action='evaluated', summary=str(destination))


def report_stage(root, execute=False):
    path = root / 'final_evaluation/REPORT.md'
    if path.exists():
        return dict(stage='stage7-report', action='reuse', report=str(path))
    if not execute:
        return dict(stage='stage7-report', action='generate_from_frozen_results', read_only=True)
    if not (root / 'final_evaluation/summary.json').exists():
        raise ValueError('Final evaluation is required')
    with execution_lock():
        from .report import main as report

        report()
    return dict(stage='stage7-report', action='generated', report=str(path))
