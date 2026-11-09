"""Stage plans, prerequisite checks, the execution lock and stage execution over frozen run artifacts."""

import contextlib
import json
import math
import os

from ..common import LOCK_FILE, SEED, full_model_count, num_targets, utc_now
from ..common import read_json as read
from ..common import sha256_file as sha
from ..config import run_protocol
from ..paths import REPO_ROOT

STAGES = {
    'stage1-classifier': ('classifier',),
    'stage2-experts': tuple(f'expert_{i}' for i in range(num_targets())),
    'stage3-baseline': ('plain',),
    'stage4-film': ('film',),
    'stage5-condconv': ('condconv',),
}


def stage_models(root):
    models = {**STAGES, 'stage2-experts': tuple(f'expert_{i}' for i in range(num_targets(root)))}
    plan = root / 'study_plan.json' if root is not None else None
    order = read(plan).get('training_stage_order') if plan is not None and plan.is_file() else None
    if order is not None:
        if len(order) != len(models) or set(order) != set(models):
            raise ValueError('Training stage order must contain each model stage exactly once')
        return {stage: models[stage] for stage in order}
    return models


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


def _budget_cap_warning(root, name, value, protocol):
    """A recorded operational decision permits nonconvergence warnings, never a false gate pass."""
    decision = root / 'budget_cap_review.json'
    if not decision.is_file():
        return None
    review = read(decision)
    if review.get('allow_budget_cap_nonconvergence') is not True or name not in review.get('models', []):
        return None
    plan = read(root / 'study_plan.json')
    if review.get('run_protocol_sha256') != sha(root / 'study_plan.json'):
        raise ValueError('Budget-cap review belongs to a different run protocol')
    import dataclasses

    from ..training.control import load_policy

    policy = dataclasses.asdict(load_policy(root / 'study_plan.json', name))
    policy = json.loads(json.dumps(policy))
    budget_stop = value.get('stop_reason') == 'max_epochs' and value.get('epochs') == policy.get('max_epochs')
    continuation = root / 'continuation_preparation.json'
    patience_stop = (
        review.get('allow_patience_nonconvergence') is True
        and continuation.is_file()
        and read(continuation).get('new_protocol_sha256') == sha(root / 'study_plan.json')
        and value.get('stop_reason') == 'validation_patience'
        and read(continuation).get('source_terminal_epoch', 200) < value.get('epochs', 0) < policy['max_epochs']
        and value.get('early_stopping', {}).get('bad_checks', 0) >= policy['patience']
        and value.get('early_stopping', {}).get('last_epoch') == value.get('epochs')
    )
    if (
        value.get('optimizer_name') != 'RAdamScheduleFree'
        or not (budget_stop or patience_stop)
        or review.get('maximum_epochs') != policy.get('max_epochs')
        or value.get('policy') != policy
    ):
        return None
    if value.get('protocol_sha256') != review.get('training_protocol_sha256'):
        raise ValueError('Budget-cap model training protocol differs from reviewed protocol')
    fixed = review.get('fixed_existing_checkpoints', {}).get(name)
    if fixed is not None and sha(root / 'full' / name / 'best.pt') != fixed:
        raise ValueError('Reviewed fixed checkpoint changed')
    from ..training.sufficiency import assess

    history = read(root / 'full' / name / 'history.json')
    if len(history) != value['epochs'] or any(row['epoch'] != i + 1 for i, row in enumerate(history)):
        raise ValueError('Budget-cap history does not reach the reported terminal epoch')
    assessment = assess(history, protocol.sufficiency_gate)
    if assessment != value.get('terminal_learning_assessment'):
        raise ValueError('Saved sufficiency result differs from unchanged gate recomputation')
    gate = protocol.sufficiency_gate
    train = assessment['terminal_train']
    if (
        train['macro_dice'] < gate.train_dice
        or train['macro_global_dice'] < gate.global_dice
        or assessment['minimum_target_train_dice'] < gate.minimum_target
        or assessment['last20percent_selection_gain'] <= gate.maximum_gain
    ):
        return None
    return (
        ('Validation-patience nonconvergence warning: ' if patience_stop else 'Budget-capped nonconvergence warning: ')
        + 'the original sufficiency gate remains FAILED; '
        'train-quality thresholds passed. Fixed best checkpoint included without extra epochs or test-based tuning.'
    )


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
    reuse_audit = root / 'expert_reuse_audit.json'
    if name.startswith('expert_') and reuse_audit.exists():
        expected = next(
            (row['checkpoint_sha256'] for row in read(reuse_audit)['reused_experts'] if row['model'] == name), None
        )
        if expected is None or sha(weight) != expected:
            raise ValueError(f'{name}: reused expert checkpoint changed')
    protocol = run_protocol(root)
    if name == 'classifier':
        if not _classifier_score_ok(value, protocol):
            return _blocked(name, 'quality_review_required', 'blocked_quality_review')
        note = None
    else:
        if value.get('learning_sufficiency_pass') is not True:
            note = _budget_cap_warning(root, name, value, protocol)
            if note is None:
                return _blocked(name, 'quality_review_required', 'blocked_quality_review')
        else:
            note = _selected_quality_note(value, protocol)
    return dict(
        name=name,
        state='completed',
        action='reuse',
        epochs=value.get('epochs'),
        best_epoch=value.get('best_epoch'),
        quality_note=note,
    )


def all_model_names(root=None):
    return [name for names in stage_models(root).values() for name in names]


def stage_plan(root, stage, target=None):
    if target is not None and target not in range(num_targets(root)):
        raise ValueError('Target outside selected run definition')
    names = stage_models(root)[stage]
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
    for key, names in stage_models(root).items():
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
    if (
        stage == 'stage2-experts'
        and read(root / 'study_plan.json').get('reuse_experts_only')
        and any(row['action'] != 'reuse' for row in plan['models'])
    ):
        raise ValueError('This run permits verified expert reuse only; expert training is forbidden')
    if any(model['action'].startswith('blocked') for model in plan['models']):
        raise ValueError('Existing partial/quality-failed run requires review; no overwrite or automatic restart')
    with execution_lock():
        for model in plan['models']:
            if model['action'] != 'reuse':
                try:
                    _train(model['name'])
                except RuntimeError as error:
                    expected = (
                        ('Terminal learning sufficiency requires review; thresholds not changed', model['name']),
                    )
                    if error.args != expected:
                        raise
                    state = existing_model(root, model['name'])
                    if state['action'] != 'reuse' or not (state.get('quality_note') or '').startswith(
                        ('Budget-capped nonconvergence warning:', 'Validation-patience nonconvergence warning:')
                    ):
                        raise
    return {
        **plan,
        'execution': 'completed_or_reused',
        'models': [existing_model(root, row['name']) for row in plan['models']],
    }


def _reuse_saved_evaluation(root, destination):
    if not (root / LOCK_FILE).is_file():
        raise ValueError('Saved evaluation has no fixed checkpoint lock')
    summary = read(destination)
    if summary.get('seed') != SEED or summary.get('full_training_models') != full_model_count(root):
        raise ValueError('Saved evaluation protocol mismatch')
    manifest = root / 'run_manifest.json'
    if manifest.exists():
        registered = read(manifest).get('artifacts_sha256', {}).get('final_evaluation/summary.json')
        if registered != sha(destination):
            raise ValueError('Registered saved evaluation hash mismatch')
    return dict(
        stage='stage6-evaluate',
        action='reuse_frozen_results',
        models=full_model_count(root),
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
        full_training_models=full_model_count(root),
        seed=SEED,
        test_never_used_for_checkpoint_selection=True,
        checkpoints=checkpoints,
        study_plan_sha256=sha(root / 'study_plan.json'),
        cohort_sha256=sha(root / 'cohort.json'),
    )
    (root / LOCK_FILE).write_text(json.dumps(record, indent=2), encoding='utf-8')


def evaluate_stage(root, execute=False):
    names = all_model_names(root)
    if any(existing_model(root, name)['action'] != 'reuse' for name in names):
        raise ValueError('All run models must be complete before final test')
    destination = root / 'final_evaluation/summary.json'
    if destination.exists():
        return _reuse_saved_evaluation(root, destination)
    if not execute:
        return dict(
            stage='stage6-evaluate',
            action='lock_checkpoints_then_evaluate',
            models=full_model_count(root),
            read_only=True,
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
