"""Prepare a fresh six-target comparison; reuse data/initial weights, never trained model weights."""

import dataclasses
import os
import shutil
from pathlib import Path

from ..common import read_json, sha256_file, utc_now, write_json
from ..config import load_protocol
from ..paths import REPO_ROOT, archives_root, data_root, initialization_root
from ..training.control import load_policy


def prepare(root, source, execute=False):
    root, source = Path(root), Path(source)
    if root.exists():
        raise ValueError('Destination already exists; no overwrite')
    plan_path = REPO_ROOT / 'configs/sixplane_radam_protocol.json'
    new_protocol = load_protocol(plan_path)
    policy = load_policy(plan_path)
    old_protocol = load_protocol(source / 'study_plan.json')
    for name in ['seed', 'inputs', 'learning_rates', 'preprocessing', 'augmentation', 'evaluation']:
        if getattr(new_protocol, name) != getattr(old_protocol, name):
            raise ValueError('Unexpected change of ' + name)
    targets = read_json(source / 'confirmed_targets.json')
    expected = read_json(REPO_ROOT / 'configs/sixplane_data_definition.json')['targets']
    if targets['targets'] != expected:
        # Metadata may carry extra keys; identity/semantic mapping must still match exactly.
        keys = ['id', 'name', 'source_plane_id', 'source_plane', 'source_cached_class_id']
        if [[t[k] for k in keys] for t in targets['targets']] != [[t[k] for k in keys] for t in expected]:
            raise ValueError('Expected exactly six CSP-free targets')
    refs = read_json(source / 'data_references.json')
    data, archives, initial = data_root(source, refs), archives_root(source, refs), initialization_root(source)
    for p, k in [
        (data / 'manifest.json', 'manifest_sha256'),
        (data / 'split.npy', 'split_sha256'),
        (source / 'cohort.json', 'cohort_sha256'),
    ]:
        if sha256_file(p) != refs[k]:
            raise ValueError('Changed source data/cohort: ' + k)
    dest_initial = REPO_ROOT / 'data' / root.name / 'initialization'
    if dest_initial.exists():
        raise ValueError('Initialization destination already exists')
    report = dict(
        action='prepare_fresh_radam',
        experiment=root.name,
        source_cohort=source.name,
        new_training_models=['classifier', *[f'expert_{i}' for i in range(6)], 'plain', 'film', 'condconv'],
        reused_trained_models=0,
        policy=dataclasses.asdict(policy),
        test_selection_or_tuning=False,
    )
    if not execute:
        return report
    root.mkdir(parents=True, exist_ok=False)
    dest_initial.mkdir(parents=True, exist_ok=False)
    for filename in [
        'fresh_common_initialization_seed17.pt',
        'encoder_mobilenet_v2_imagenet_gray.pt',
        'encoder_weights_metadata.json',
    ]:
        shutil.copy2(initial / filename, dest_initial / filename)
        if sha256_file(initial / filename) != sha256_file(dest_initial / filename):
            raise ValueError('Initialization copy mismatch')
    for filename in ['confirmed_targets.json', 'cohort.json']:
        shutil.copy2(source / filename, root / filename)
    old_plan = read_json(source / 'study_plan.json')
    plan = read_json(plan_path)
    plan.update(
        segmentation_counts=old_plan['segmentation_counts'],
        classification_counts=old_plan['classification_counts'],
        reuse_experts_only=False,
        new_training_models=report['new_training_models'],
    )
    write_json(root / 'study_plan.json', plan)
    write_json(
        root / 'data_references.json',
        dict(
            immutable_data_source=os.path.relpath(data, root),
            source_archives=os.path.relpath(archives, root),
            initialization_source=os.path.relpath(dest_initial, root),
            manifest_sha256=refs['manifest_sha256'],
            split_sha256=refs['split_sha256'],
            cohort_sha256=refs['cohort_sha256'],
        ),
    )
    report.update(
        utc=utc_now(),
        initialization_sha256=sha256_file(dest_initial / 'fresh_common_initialization_seed17.pt'),
        protocol_sha256=sha256_file(root / 'study_plan.json'),
        cohort_sha256=refs['cohort_sha256'],
    )
    write_json(root / 'preparation_audit.json', report)
    return report
