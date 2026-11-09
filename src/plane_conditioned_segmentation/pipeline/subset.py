"""Derive the CSP-free cohort and reuse only verified frozen binary experts."""

import dataclasses
import hashlib
import os
import shutil
from pathlib import Path

from ..common import SEED, read_json, sha256_file, utc_now, write_json
from ..config import load_protocol, run_protocol
from ..paths import REPO_ROOT, archives_root, data_root, initialization_root


def prepare_sixplane(root, source, execute=False):
    """No overwrite; retain immutable cache indices/splits and copy selected expert weights unchanged."""
    import numpy as np

    root, source = Path(root), Path(source)
    if root.exists():
        raise ValueError('Destination run already exists; no overwrite or implicit reuse')
    spec = read_json(source / 'confirmed_targets.json')
    targets = spec['targets']
    if len(targets) != 7 or targets[6]['name'] != 'csp_transventricular':
        raise ValueError('Source must be the verified seven-target run with CSP last')
    expected = read_json(REPO_ROOT / 'configs/sixplane_data_definition.json')['targets']
    keys = ['id', 'name', 'source_plane_id', 'source_plane', 'source_cached_class_id']
    if [[t[k] for k in keys] for t in targets[:6]] != [[t[k] for k in keys] for t in expected]:
        raise ValueError('Source expert target mapping differs from six-target definition')
    template = REPO_ROOT / 'configs/sixplane_protocol.json'
    if dataclasses.asdict(run_protocol(source)) != dataclasses.asdict(load_protocol(template)):
        raise ValueError('Preprocessing/augmentation/optimizer/stopping conditions differ from source')
    refs = read_json(source / 'data_references.json')
    data, archives, initial = data_root(source, refs), archives_root(source, refs), initialization_root(source)
    for path, key in [
        (data / 'manifest.json', 'manifest_sha256'),
        (data / 'split.npy', 'split_sha256'),
        (source / 'cohort.json', 'cohort_sha256'),
    ]:
        if sha256_file(path) != refs[key]:
            raise ValueError('Source data/split/cohort signature mismatch')
    cohort = read_json(source / 'cohort.json')
    plane, split, areas = [np.load(data / n, mmap_mode='r') for n in ['plane.npy', 'split.npy', 'areas.npy']]
    keep_planes = {t['source_plane_id'] for t in targets[:6]}
    allowed = {
        key: [i for i in cohort['allowed_indices'][key] if int(plane[i]) in keep_planes]
        for key in ['train', 'val', 'test']
    }
    new_cohort = dict(
        scope='Six targets; CSP excluded',
        policy=cohort['policy'],
        allowed_indices=allowed,
        patient_independence_verified=False,
        source_cohort_sha256=refs['cohort_sha256'],
        quarantined=[r for r in cohort.get('quarantined', []) if r['plane'] in keep_planes],
        source_quarantine_preserved=True,
        no_split_reassignment=True,
    )
    new_cohort['quarantine_count'] = len(new_cohort['quarantined'])
    model_lock = read_json(source / 'FINAL_MODEL_LOCK.json')
    audits = []
    for t in targets[:6]:
        name = f'expert_{t["id"]}'
        result = read_json(source / f'full/{name}/result.json')
        hashes = {}
        for s, key in [(0, 'train'), (1, 'val')]:
            ids = np.array(
                [
                    i
                    for i in allowed[key]
                    if plane[i] == t['source_plane_id'] and areas[i, t['source_cached_class_id']] > 0
                ],
                dtype=np.int64,
            )
            assert (split[ids] == s).all()
            digest = hashlib.sha256(ids.tobytes()).hexdigest()
            field = 'train_ids_sha256' if s == 0 else 'valid_ids_sha256'
            if digest != result[field]:
                raise ValueError(f'{name}: unchanged expert split cannot be established')
            hashes[field] = digest
        path = source / f'full/{name}/best.pt'
        digest = sha256_file(path)
        if digest != model_lock['checkpoints'].get(str(path.relative_to(source))):
            raise ValueError(f'{name}: source checkpoint does not match fixed lock')
        if result.get('target') != t['id'] or result.get('seed') != SEED or result.get('output_channels') != 1:
            raise ValueError(f'{name}: incompatible semantic target/seed/output')
        audits.append(dict(model=name, target=t['id'], checkpoint_sha256=digest, **hashes))
    classification_counts = [len(allowed[k]) for k in ['train', 'val', 'test']]
    segmentation_counts = [
        sum(
            areas[i, t['source_cached_class_id']] > 0
            for i in allowed[k]
            for t in targets[:6]
            if plane[i] == t['source_plane_id']
        )
        for k in ['train', 'val', 'test']
    ]
    report = dict(
        action='prepare_sixplane',
        execute=execute,
        run=str(root),
        source_run=source.name,
        targets=6,
        total_models=10,
        reused_experts=audits,
        segmentation_counts=list(map(int, segmentation_counts)),
        classification_counts=classification_counts,
        test_selection_or_tuning=False,
    )
    if not execute:
        return report
    import torch

    init = REPO_ROOT / 'data' / root.name / 'initialization'
    if init.exists():
        raise ValueError('Destination initialization exists; no overwrite')
    state = torch.load(initial / f'fresh_common_initialization_seed{SEED}.pt', map_location='cpu', weights_only=True)
    for key in ['segmentation_head.0.weight', 'segmentation_head.0.bias']:
        if state[key].shape[0] != 7:
            raise ValueError('Source common initialization is not seven-channel')
        state[key] = state[key][:6].clone()
    mean_metadata = read_json(initial / 'encoder_weights_metadata.json')
    for audit in audits:
        ck = torch.load(source / f'full/{audit["model"]}/best.pt', map_location='cpu', weights_only=True)
        if ck['mode'] != 'expert' or ck['target'] != audit['target']:
            raise ValueError('Checkpoint architecture metadata differs from expert result')
        if ck['state']['network.segmentation_head.0.weight'].shape[0] != 1:
            raise ValueError('Expert checkpoint is not binary')
        if float(ck['state']['mean'].reshape(-1)[0]) != float(torch.tensor(mean_metadata['gray_mean']).float()):
            raise ValueError('Expert normalization mean differs')
        if float(ck['state']['std'].reshape(-1)[0]) != float(torch.tensor(mean_metadata['gray_std']).float()):
            raise ValueError('Expert normalization std differs')
    root.mkdir(parents=True, exist_ok=False)
    init.mkdir(parents=True, exist_ok=False)
    for name in ['encoder_weights_metadata.json', 'encoder_mobilenet_v2_imagenet_gray.pt']:
        shutil.copy2(initial / name, init / name)
    torch.save(state, init / f'fresh_common_initialization_seed{SEED}.pt')
    write_json(
        root / 'confirmed_targets.json',
        {**spec, 'targets': targets[:6], 'annotation_task': 'Six selected plane-target tasks; CSP excluded'},
    )
    write_json(root / 'cohort.json', new_cohort)
    plan = read_json(template)
    plan.update(
        segmentation_counts=report['segmentation_counts'],
        classification_counts=classification_counts,
        expert_training='Reused source experts without retraining; same individual train/validation IDs and preprocessing',
        new_training_models=['classifier', 'plain', 'film', 'condconv'],
        gate_basis='terminal_training_history; selected-best quality reported separately',
        reuse_experts_only=True,
    )
    write_json(root / 'study_plan.json', plan)
    relative = lambda p: os.path.relpath(p, root)
    write_json(
        root / 'data_references.json',
        dict(
            immutable_data_source=relative(data),
            source_archives=relative(archives),
            initialization_source=relative(init),
            manifest_sha256=refs['manifest_sha256'],
            split_sha256=refs['split_sha256'],
            cohort_sha256=sha256_file(root / 'cohort.json'),
        ),
    )
    for audit in audits:
        src = source / 'full' / audit['model']
        dst = root / 'full' / audit['model']
        dst.mkdir(parents=True)
        for name in [
            'best.pt',
            'result.json',
            'history.json',
            'configuration.json',
            'train_metrics.json',
            'validation.json',
        ]:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        assert sha256_file(dst / 'best.pt') == audit['checkpoint_sha256']
    report.update(
        utc=utc_now(),
        initialization=dict(
            source_sha256=sha256_file(initial / f'fresh_common_initialization_seed{SEED}.pt'),
            derived_sha256=sha256_file(init / f'fresh_common_initialization_seed{SEED}.pt'),
            derivation='Keep identical encoder/decoder, slice fresh seven-output head to targets 0..5',
        ),
        source_fixed_lock_sha256=sha256_file(source / 'FINAL_MODEL_LOCK.json'),
    )
    write_json(root / 'expert_reuse_audit.json', report)
    return report
