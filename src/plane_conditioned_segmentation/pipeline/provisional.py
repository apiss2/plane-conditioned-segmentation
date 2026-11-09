"""A separate, explicitly mixed-optimizer provisional test after shared models, before fresh experts.

No test-based tuning. Copy the four newly selected models and six previously fixed experts; the
new all-Schedule-Free run is untouched and continues under its already frozen policy afterwards.
"""

import os
import shutil

from ..common import read_json, sha256_file, utc_now, write_json
from ..paths import archives_root, data_root, experiment_root, initialization_root
from .stages import existing_model

ARTIFACTS = [
    'best.pt',
    'result.json',
    'history.json',
    'configuration.json',
    'train_metrics.json',
    'validation.json',
    'terminal_learning_assessment.json',
]


def prepare(root):
    plan = read_json(root / 'study_plan.json')
    source = experiment_root(plan['provisional_reference_experts'])
    destination = experiment_root(root.name + '_provisional')
    refs = read_json(root / 'data_references.json')
    old_refs = read_json(source / 'data_references.json')
    if (refs['manifest_sha256'], refs['split_sha256'], refs['cohort_sha256']) != (
        old_refs['manifest_sha256'],
        old_refs['split_sha256'],
        old_refs['cohort_sha256'],
    ):
        raise ValueError('Provisional old experts do not have identical data/split/cohort')
    if read_json(root / 'confirmed_targets.json') != read_json(source / 'confirmed_targets.json'):
        raise ValueError('Provisional target definitions differ')
    inputs = {
        name: (source if name.startswith('expert_') else root)
        for name in ['classifier', 'plain', 'film', 'condconv', *[f'expert_{i}' for i in range(6)]]
    }
    hashes = {}
    for name, owner in inputs.items():
        if existing_model(owner, name)['action'] != 'reuse':
            raise ValueError('Provisional model is incomplete or failed quality review: ' + name)
        for filename in ARTIFACTS:
            path = owner / 'full' / name / filename
            if path.is_file():
                hashes[f'full/{name}/{filename}'] = sha256_file(path)
    if destination.exists():
        audit = destination / 'provisional_model_provenance.json'
        if not audit.is_file() or read_json(audit)['copied_artifacts_sha256'] != hashes:
            raise ValueError('Provisional run already exists with a different/partial artifact contract')
        review_hash = read_json(audit).get('budget_cap_review_sha256')
        if review_hash and sha256_file(destination / 'budget_cap_review.json') != review_hash:
            raise ValueError('Provisional budget-cap review changed')
        if any(sha256_file(destination / p) != h for p, h in hashes.items()):
            raise ValueError('Provisional copies changed; no overwrite')
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    for name, owner in inputs.items():
        folder = destination / 'full' / name
        folder.mkdir(parents=True)
        for filename in ARTIFACTS:
            path = owner / 'full' / name / filename
            if path.is_file():
                shutil.copy2(path, folder / filename)
    for filename in ['confirmed_targets.json', 'cohort.json']:
        shutil.copy2(root / filename, destination / filename)
    data, archives, initial = data_root(root, refs), archives_root(root, refs), initialization_root(root)
    write_json(
        destination / 'data_references.json',
        dict(
            immutable_data_source=os.path.relpath(data, destination),
            source_archives=os.path.relpath(archives, destination),
            initialization_source=os.path.relpath(initial, destination),
            manifest_sha256=refs['manifest_sha256'],
            split_sha256=refs['split_sha256'],
            cohort_sha256=refs['cohort_sha256'],
        ),
    )
    provisional_plan = {
        **plan,
        'comparison': 'PROVISIONAL mixed optimizer: RAdamScheduleFree classifier/shared3; old AdamW+cosine experts6. Not the final equal-policy comparison.',
        'expert_training': 'Six old fixed experts copied without training into this provisional comparison only',
        'reuse_experts_only': True,
        'provisional': True,
        'training_or_tuning_permitted': False,
    }
    write_json(destination / 'study_plan.json', provisional_plan)
    review_path = root / 'budget_cap_review.json'
    if review_path.exists():
        write_json(
            destination / 'budget_cap_review.json',
            {
                **read_json(review_path),
                'run_protocol_sha256': sha256_file(destination / 'study_plan.json'),
                'source_review_sha256': sha256_file(review_path),
                'copied_from': root.name,
            },
        )
    for p, h in hashes.items():
        if sha256_file(destination / p) != h:
            raise ValueError('Provisional copy hash mismatch')
    write_json(
        destination / 'provisional_model_provenance.json',
        dict(
            utc=utc_now(),
            shared_source=root.name,
            budget_cap_review_sha256=sha256_file(destination / 'budget_cap_review.json')
            if review_path.exists()
            else None,
            old_expert_source=source.name,
            copied_artifacts_sha256=hashes,
            model_optimizers={
                name: 'AdamW+cosine (prior training)' if name.startswith('expert_') else 'RAdamScheduleFree 1.4.1'
                for name in inputs
            },
            conditions_fixed_before_test=True,
            test_used_for_model_or_threshold_selection=False,
            caveat=provisional_plan['comparison'],
        ),
    )
    return destination
