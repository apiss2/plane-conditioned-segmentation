"""Explicit approved terminal-state continuation; original runs remain read only."""

import copy
import dataclasses
import os
import shutil
from pathlib import Path

import torch

from ..common import LOCK_FILE, read_json, sha256_file, utc_now, write_json
from ..config import load_protocol
from ..paths import REPO_ROOT, archives_root, data_root, initialization_root, is_archived_run
from ..training.control import EarlyStopping, load_policy

SHARED = ('plain', 'film', 'condconv')
INHERITED = ('classifier', *(f'expert_{i}' for i in range(6)))


def _assert_equal(a, b):
    """Exact recursive comparison includes all optimizer, scaler, RNG and sampler progress."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError('Continuation changed a numerical state tensor')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError('Continuation changed state keys')
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('Continuation changed state sequence')
        for x, y in zip(a, b):
            _assert_equal(x, y)
    elif a != b:
        raise ValueError('Continuation changed state value')


def transition_checkpoint(ck, config, history, *, label, old_sha, new_sha, policy):
    """Only the new protocol identity and authorized stopping-policy metadata may change."""
    source_epoch = {300: 200, 500: 300}.get(policy.max_epochs)
    if source_epoch is None:
        raise ValueError('Only approved 200-to-300 or 300-to-500 continuation is supported')
    old_policy = {
        **dataclasses.asdict(policy),
        'max_epochs': source_epoch,
        'min_epochs': 60 if source_epoch == 200 else 1,
        'min_applied_updates': 1500 if source_epoch == 200 else 0,
    }
    old_policy = read_normalized(old_policy)
    if (
        label not in SHARED
        or not ck.get('resumable')
        or ck.get('optimizer_name') != 'RAdamScheduleFree'
        or ck.get('weight_sequence') != 'schedulefree_eval_x'
        or ck.get('protocol_sha256') != old_sha
        or ck.get('label') != label
        or ck.get('mode') != label
        or ck.get('target') is not None
        or ck.get('target_count') != 6
        or ck.get('epoch') != source_epoch
        or ck.get('next_epoch') != source_epoch
        or ck.get('completed_batches') != 0
        or len(history) != source_epoch
        or any(row['epoch'] != i + 1 for i, row in enumerate(history))
        or config.get('label') != label
        or config.get('protocol_sha256') != old_sha
        or config.get('policy') != old_policy
        or read_normalized(ck.get('early_stopping', {}).get('policy')) != old_policy
        or ck.get('early_stopping', {}).get('last_epoch') != source_epoch
        or ck.get('early_stopping', {}).get('last_updates') != ck.get('applied')
        or any(g.get('train_mode') is not False for g in ck.get('optimizer', {}).get('param_groups', []))
        or len(ck.get('optimizer', {}).get('param_groups', [])) != 2
        or not all(k in ck for k in ['scaler', 'torch_rng', 'cuda_rng', 'history', 'attempted', 'applied'])
    ):
        raise ValueError(f'Expected complete, compatible eval-x shared-model state at epoch{source_epoch}')
    if (policy.max_epochs, policy.min_epochs, policy.min_applied_updates, policy.patience, policy.min_delta) != (
        policy.max_epochs,
        1,
        0,
        30,
        0.0,
    ):
        raise ValueError('Only approved cap/patience30/no-extra-minima continuation is allowed')
    _assert_equal(ck['history'], history)
    groups = ck['optimizer']['param_groups']
    if [g.get('lr') for g in groups] != [config['encoder_lr'], config['decoder_lr']]:
        raise ValueError('Optimizer learning rates differ from source configuration')
    if any(g.get('k') != ck['applied'] for g in groups):
        raise ValueError('Optimizer step counts differ from saved progress')
    for state in ck['optimizer']['state'].values():
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite optimizer tensor')
    tracker = EarlyStopping(load_policy_dict(old_policy))
    tracker.load_state_dict(ck['early_stopping'])
    if tracker.best_epoch != history[-1]['epoch'] - tracker.bad_checks:
        raise ValueError('Best epoch and patience counter disagree')
    changed = copy.deepcopy(ck)
    changed['protocol_sha256'] = new_sha
    changed['early_stopping']['policy'] = dataclasses.asdict(policy)
    # Removing only these two metadata leaves must recover the original payload exactly.
    reverted = copy.deepcopy(changed)
    reverted['protocol_sha256'] = old_sha
    reverted['early_stopping']['policy'] = ck['early_stopping']['policy']
    _assert_equal(ck, reverted)
    return changed


def read_normalized(value):
    import json

    return json.loads(json.dumps(value))


def load_policy_dict(value):
    from ..config import _build
    from ..training.control import ScheduleFreePolicy

    return _build(ScheduleFreePolicy, value, 'schedulefree', strict=True)


def prepare(root, source, execute=False, *, max_epochs=300):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise ValueError('Continuation destination exists; no overwrite or duplicate fork')
    source_base = source.parent.parent if is_archived_run(source) else source.parent
    if root.parent != source_base or root == source or is_archived_run(root):
        raise ValueError('Continuation must be a new sibling run')
    old_plan = read_json(source / 'study_plan.json')
    old_sha = sha256_file(source / 'study_plan.json')
    old_policy = load_policy(source / 'study_plan.json')
    source_epoch = {300: 200, 500: 300}.get(max_epochs)
    if (
        old_policy is None
        or source_epoch is None
        or old_policy.max_epochs != source_epoch
        or len(read_json(source / 'confirmed_targets.json')['targets']) != 6
    ):
        raise ValueError('Expected completed six-target Schedule-Free source')
    lock = read_json(source / LOCK_FILE)
    if len(lock['checkpoints']) != 10 or not (source / 'COMPARISON_COMPLETE.json').is_file():
        raise ValueError('Source comparison is not complete and locked')
    for rel, digest in lock['checkpoints'].items():
        if sha256_file(source / rel) != digest:
            raise ValueError('Source fixed checkpoint changed')
    refs = read_json(source / 'data_references.json')
    data, archives, initial = data_root(source, refs), archives_root(source, refs), initialization_root(source)
    for key, path in [
        ('manifest_sha256', data / 'manifest.json'),
        ('split_sha256', data / 'split.npy'),
        ('cohort_sha256', source / 'cohort.json'),
    ]:
        if sha256_file(path) != refs[key]:
            raise ValueError('Source cohort/cache differs')
    policy = dataclasses.replace(old_policy, max_epochs=max_epochs, min_epochs=1, min_applied_updates=0)
    plan = copy.deepcopy(old_plan)
    plan['schedulefree'] = read_normalized(dataclasses.asdict(policy))
    plan['full_segmentation_rule'].update(min_epochs=1, min_attempted_updates=0)
    plan.update(
        new_training_models=list(SHARED),
        continued_models=list(SHARED),
        inherited_models=list(INHERITED),
        comparison=f'Three shared models continued from terminal{source_epoch} to cap{max_epochs}; classifier and six RAdam experts unchanged',
        early_stopping_definition=f'Carry existing best/reference/counter; strict validation improvement; patience30; total cap{max_epochs}; no extra minimum epochs/updates; test comparison exploratory only, never used for selection/tuning',
    )
    plan.pop('provisional_reference_experts', None)
    report = dict(
        action='prepare_separate_shared_continuation',
        source=source.name,
        destination=root.name,
        continued_models=list(SHARED),
        inherited_models=list(INHERITED),
        policy=plan['schedulefree'],
        source_protocol_sha256=old_sha,
        source_terminal_epoch=source_epoch,
        total_max_epochs=max_epochs,
        additional_epoch_budget=max_epochs - source_epoch,
        test_evaluation_scope='Exploratory reuse of previously observed fixed holdout; validation-only checkpoint selection',
        test_for_selection_or_tuning=False,
    )
    if not execute:
        return report
    # Preflight every model before creating any destination files.
    for name in SHARED:
        out = source / 'full' / name
        ck = torch.load(out / 'resume.pt', map_location='cpu', weights_only=True)
        transition_checkpoint(
            ck,
            read_json(out / 'configuration.json'),
            read_json(out / 'history.json'),
            label=name,
            old_sha=old_sha,
            new_sha='preflight',
            policy=policy,
        )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'study_plan.json', plan)
    new_sha = sha256_file(root / 'study_plan.json')
    for name in ['confirmed_targets.json', 'cohort.json']:
        shutil.copy2(source / name, root / name)
    refs.update(
        immutable_data_source=os.path.relpath(data, root),
        source_archives=os.path.relpath(archives, root),
        initialization_source=os.path.relpath(initial, root),
    )
    write_json(root / 'data_references.json', refs)
    copied = {}
    for name in INHERITED:
        src = source / 'full' / name
        dst = root / 'full' / name
        shutil.copytree(src, dst)
        for p in src.rglob('*'):
            if p.is_file():
                rel = p.relative_to(source)
                digest = sha256_file(p)
                if sha256_file(root / rel) != digest:
                    raise ValueError('Inherited model copy differs')
                copied[str(rel)] = digest
    transitions = {}
    for name in SHARED:
        src = source / 'full' / name
        dst = root / 'full' / name
        dst.mkdir(parents=True, exist_ok=False)
        for old, new in [
            ('resume.pt', 'resume_origin.pt'),
            ('configuration.json', 'configuration_origin.json'),
            ('result.json', 'result_origin.json'),
            ('history.json', 'history.json'),
            ('best.pt', 'best.pt'),
        ]:
            shutil.copy2(src / old, dst / new)
            if sha256_file(src / old) != sha256_file(dst / new):
                raise ValueError('Source copy differs')
        ck = torch.load(dst / 'resume_origin.pt', map_location='cpu', weights_only=True)
        config = read_json(dst / 'configuration_origin.json')
        changed = transition_checkpoint(
            ck, config, read_json(dst / 'history.json'), label=name, old_sha=old_sha, new_sha=new_sha, policy=policy
        )
        torch.save(changed, dst / 'resume.pt')
        _assert_equal(changed, torch.load(dst / 'resume.pt', map_location='cpu', weights_only=True))
        config.update(epochs=max_epochs, policy=plan['schedulefree'], protocol_sha256=new_sha)
        write_json(dst / 'configuration.json', config)
        transitions[name] = dict(
            source_resume_sha256=sha256_file(src / 'resume.pt'),
            copied_resume_origin_sha256=sha256_file(dst / 'resume_origin.pt'),
            prepared_resume_sha256=sha256_file(dst / 'resume.pt'),
            retained_best_sha256=sha256_file(dst / 'best.pt'),
            preserved_history_sha256=sha256_file(dst / 'history.json'),
            next_epoch=source_epoch + 1,
            source_best_epoch=ck['early_stopping']['best_epoch'],
            preserved_bad_checks=ck['early_stopping']['bad_checks'],
            preserved_applied=ck['applied'],
            numerical_state_exact=True,
            changed_metadata_only=['protocol_sha256', 'early_stopping.policy'],
        )
    report.update(
        utc=utc_now(),
        new_protocol_sha256=new_sha,
        transitions=transitions,
        inherited_artifacts_sha256=copied,
        stopping_condition_caveat='Original first200 history retained, including extra minima. Existing expert0 selected200 differs from strict-patience58; no historic58 weights exist.',
        initialization_unchanged=True,
        scheduler=None,
    )
    write_json(root / 'continuation_preparation.json', report)
    write_json(
        root / 'budget_cap_review.json',
        dict(
            utc=utc_now(),
            allow_budget_cap_nonconvergence=True,
            allow_patience_nonconvergence=True,
            models=list(SHARED),
            maximum_epochs=max_epochs,
            run_protocol_sha256=new_sha,
            training_protocol_sha256=new_sha,
            fixed_existing_checkpoints={},
            thresholds_and_failed_results_unchanged=True,
            test_for_selection_or_tuning=False,
            scope='Continue/evaluate authorized validation-patience or total-budget stop with unchanged nonconvergence warning; terminal train-quality failures remain blocked.',
        ),
    )
    archive = root / 'reproducibility/execution_source'
    archive.mkdir(parents=True)
    paths = [
        *sorted((REPO_ROOT / 'src').rglob('*.py')),
        *sorted((REPO_ROOT / 'configs').glob('*.json')),
        REPO_ROOT / 'pyproject.toml',
        REPO_ROOT / 'uv.lock',
        REPO_ROOT / 'tools/run_sixplane_comparison.py',
    ]
    hashes = {}
    for p in paths:
        rel = p.relative_to(REPO_ROOT)
        dst = archive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        hashes[str(rel)] = sha256_file(p)
    write_json(
        root / 'execution_source_manifest.json',
        dict(
            utc=utc_now(),
            sha256=hashes,
            run_sha256={'budget_cap_review.json': sha256_file(root / 'budget_cap_review.json')},
        ),
    )
    load_protocol(root / 'study_plan.json')
    return report
