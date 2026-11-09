"""Register immutable, already imported artifacts without training or evaluation."""

import datetime
import json

from ..common import LOCK_FILE, SEED, full_model_count
from ..common import read_json as read
from ..common import sha256_file as sha
from ..paths import data_root
from .stages import existing_model, stage_models

MODEL_ARTIFACTS = [
    ('checkpoint', 'best.pt'),
    ('result', 'result.json'),
    ('history', 'history.json'),
    ('configuration', 'configuration.json'),
]
FINAL_ARTIFACTS = [
    ('stage6-evaluate', 'final_evaluation/summary.json'),
    ('stage7-report', 'final_evaluation/REPORT.md'),
]
FROZEN_RUN_FILES = [
    LOCK_FILE,
    'study_plan.json',
    'confirmed_targets.json',
    'cohort.json',
    'data_references.json',
    'final_evaluation/summary.json',
    'final_evaluation/REPORT.md',
]


def describe(root):
    stages = []
    for stage, names in stage_models(root).items():
        models = []
        for name in names:
            row = existing_model(root, name)
            row['artifacts'] = {kind: f'full/{name}/{file}' for kind, file in MODEL_ARTIFACTS}
            models.append(row)
        stages.append(dict(stage=stage, completed=all(m['action'] == 'reuse' for m in models), models=models))
    for stage, file in FINAL_ARTIFACTS:
        done = (root / file).is_file()
        stages.append(dict(stage=stage, completed=done, artifact=file, action='reuse' if done else 'not_started'))
    return dict(
        experiment=root.name,
        run_directory=str(root),
        registered=(root / 'run_manifest.json').is_file(),
        model_count=sum(len(s.get('models', [])) for s in stages),
        complete=all(s['completed'] for s in stages),
        stages=stages,
    )


def _check_hashes(root, lock, refs):
    data = data_root(root, refs)
    for key, path in [
        ('manifest_sha256', data / 'manifest.json'),
        ('split_sha256', data / 'split.npy'),
        ('cohort_sha256', root / 'cohort.json'),
    ]:
        if sha(path) != refs[key]:
            raise ValueError('Data/cohort hash mismatch: ' + key)
    for key, path in [('study_plan_sha256', root / 'study_plan.json'), ('cohort_sha256', root / 'cohort.json')]:
        if key in lock and sha(path) != lock[key]:
            raise ValueError('Frozen protocol hash mismatch: ' + key)


def _record(root, status, plan, summary):
    model_stages = [s for s in status['stages'] if 'models' in s]
    final_stages = [s for s in status['stages'] if 'models' not in s]
    frozen = {}
    for stage in model_stages:
        for model in stage['models']:
            for path in model['artifacts'].values():
                frozen[path] = sha(root / path)
    for file in FROZEN_RUN_FILES:
        frozen[file] = sha(root / file)
    return dict(
        schema_version=1,
        experiment=root.name,
        imported_existing=True,
        training_or_test_rerun=False,
        storage_contract='Stage artifacts are run-relative; cache/archive overrides use PCS_*; '
        'source experiment is not an execution dependency.',
        stage_models={s['stage']: [m['name'] for m in s.get('models', [])] for s in status['stages']},
        stage_artifacts={s['stage']: s.get('artifact') for s in final_stages},
        artifacts_sha256=frozen,
        seed=plan['seed'],
        segmentation_counts=plan['segmentation_counts'],
        classification_counts=plan['classification_counts'],
        test_images=summary['test_segmentation_images'],
        condition_names=list(summary['scores']),
        quality_notes=[m['quality_note'] for s in model_stages for m in s['models'] if m.get('quality_note')],
        cache_format='Seven-target loader-compatible arrays with retained original indices, fixed cohort and '
        'geometry; not newly regenerated.',
        source_policy='Source folder retained; no move/delete performed by registration.',
    )


def registration(root, execute=False):
    status = describe(root)
    if not status['complete'] or status['model_count'] != full_model_count(root):
        raise ValueError('All run models, saved evaluation and report must exist before registration')
    lock = read(root / LOCK_FILE)
    plan = read(root / 'study_plan.json')
    refs = read(root / 'data_references.json')
    summary = read(root / 'final_evaluation/summary.json')
    if (
        len(lock['checkpoints']) != full_model_count(root)
        or summary['full_training_models'] != full_model_count(root)
        or summary['seed'] != SEED
    ):
        raise ValueError('Unexpected frozen run protocol')
    _check_hashes(root, lock, refs)
    record = _record(root, status, plan, summary)
    path = root / 'run_manifest.json'
    if path.exists():
        previous = read(path)
        previous.pop('registered_utc', None)
        if previous != record:
            raise ValueError('Registered artifact contract changed; review without overwriting registration')
        return dict(action='reuse_registration', manifest=str(path), complete=True, models=full_model_count(root))
    if not execute:
        return dict(action='register_existing_completed_run', manifest=str(path), read_only=True, record=record)
    record['registered_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with path.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    return dict(
        action='registered_existing_completed_run', manifest=str(path), complete=True, models=full_model_count(root)
    )
