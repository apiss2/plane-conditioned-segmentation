"""Read-only final results: ten selected models, without continuation execution."""

import re
from pathlib import PurePosixPath

from ..common import read_json, sha256_file

MANIFEST = 'FINAL_RESULTS.json'
MODEL_NAMES = ('classifier', *(f'expert_{i}' for i in range(6)), 'plain', 'film', 'condconv')
READ_COMMANDS = {'status', 'plan', 'verify', 'stage6-evaluate', 'stage7-report'}


def is_final_result(root):
    return (root / MANIFEST).is_file()


def _artifact_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts or ':' in relative or '\\' in relative:
        raise ValueError('Final result artifact must be a safe run-relative path')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Final result artifact escapes its run')
    return resolved


def validate(root):
    """Validate the frozen bundle without consulting archived training runs."""
    record = read_json(root / MANIFEST)
    if record.get('schema_version') != 1 or record.get('read_only') is not True:
        raise ValueError('Unsupported final result bundle')
    models = record.get('models', [])
    if len(models) != 10 or {row['name'] for row in models} != set(MODEL_NAMES):
        raise ValueError('Final result bundle requires one classifier and nine segmentation models')
    artifacts = record.get('artifacts_sha256', {})
    required = {
        'study_plan.json',
        'confirmed_targets.json',
        'cohort.json',
        'data_references.json',
        'FINAL_MODEL_LOCK.json',
        'provenance/original_model_lock.json',
        'provenance/original_study_plan.json',
        'final_evaluation/summary.json',
        'final_evaluation/REPORT.md',
    }
    for row in models:
        name = row['name']
        expected = f'full/{name}/best.pt'
        kind = 'classifier' if name == 'classifier' else 'segmentation'
        target = int(name.split('_')[1]) if name.startswith('expert_') else None
        if row.get('checkpoint') != expected or row.get('kind') != kind or row.get('target') != target:
            raise ValueError('Final result model mapping differs from the fixed ten-model contract')
        if row.get('checkpoint_sha256') != artifacts.get(expected):
            raise ValueError('Final result checkpoint identity differs from its artifact manifest')
        required.update(
            f'full/{name}/{filename}' for filename in ['best.pt', 'result.json', 'history.json', 'configuration.json']
        )
    if not required.issubset(artifacts):
        raise ValueError('Final result bundle is missing required artifact hashes')
    for relative, digest in artifacts.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid final result SHA256')
        if sha256_file(_artifact_path(root, relative)) != digest:
            raise ValueError('Final result artifact hash mismatch: ' + relative)
    plan = read_json(root / 'study_plan.json')
    if (
        plan.get('artifact_purpose') != 'read_only_final_results'
        or plan.get('training_or_tuning_permitted') is not False
        or plan.get('schedulefree')
        or plan.get('reuse_experts_only')
    ):
        raise ValueError('Final result plan must not enable a training workflow')
    original_lock = read_json(root / 'provenance/original_model_lock.json')
    original_checkpoints = {key.replace('\\', '/'): value for key, value in original_lock['checkpoints'].items()}
    if original_lock['study_plan_sha256'] != artifacts['provenance/original_study_plan.json']:
        raise ValueError('Original frozen training plan was changed')
    lock = read_json(root / 'FINAL_MODEL_LOCK.json')
    checkpoints = {key.replace('\\', '/'): value for key, value in lock['checkpoints'].items()}
    expected = {row['checkpoint']: row['checkpoint_sha256'] for row in models}
    if (
        checkpoints != expected
        or original_checkpoints != expected
        or lock['study_plan_sha256'] != artifacts['study_plan.json']
    ):
        raise ValueError('Final result bundle differs from original frozen model lock')
    return record


def _inference_checks(root, record):
    """Reload every classifier/expert/shared model on validation images only."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    from ..config import load_runtime
    from ..data.dataset import DATA, ROOT, Images, ids_for
    from ..models.classifier import load_classifier
    from ..models.segmentation import load

    if ROOT.resolve() != root.resolve():
        raise ValueError('Inference dataset belongs to a different experiment; use a fresh CLI process')
    torch.set_num_threads(load_runtime().torch_threads)
    widths = np.load(DATA / 'widths.npy')
    comparisons = []
    with torch.inference_mode():
        for row in record['models']:
            target = row['target']
            ids = ids_for(1, planes=[target] if target is not None else None)
            selected = [int(next(i for i in ids if widths[i] == width)) for width in sorted(set(widths[ids]))]
            if not selected:
                raise ValueError('No validation image for final model: ' + row['name'])
            loader = load_classifier if row['kind'] == 'classifier' else load
            model = loader(root / row['checkpoint'], 'cpu')
            reloaded = loader(root / row['checkpoint'], 'cpu')
            for idx in selected:
                image, _, _, plane, _ = Images([idx])[0]
                image = image[None].float() / 255
                condition = functional.one_hot(torch.tensor([plane]), 6).float()
                args = (image,) if row['kind'] == 'classifier' else (image, condition)
                logits, again = model(*args), reloaded(*args)
                expected = (
                    (1, 6) if row['kind'] == 'classifier' else (1, 1 if target is not None else 6, *image.shape[-2:])
                )
                delta = float((logits - again).abs().max())
                if tuple(logits.shape) != expected or not torch.isfinite(logits).all() or delta >= 1e-6:
                    raise ValueError('Final model reload inference failed: ' + row['name'])
                comparisons.append(
                    dict(
                        model=row['name'],
                        width=image.shape[-1],
                        shape=list(logits.shape),
                        max_reload_logit_difference=delta,
                    )
                )
            del model, reloaded
    return comparisons


def run_command(root, args):
    """Final results can be read and verified, never trained/resumed or overwritten."""
    if args.command not in READ_COMMANDS:
        raise ValueError('Final results are read-only. Choose a new --experiment for training or preparation.')
    if getattr(args, 'forbid_source', None):
        from .verification import _install_source_guard

        _install_source_guard(args.forbid_source, [])
    record = validate(root)
    result = dict(
        experiment=root.name,
        run_directory=str(root),
        read_only=True,
        complete=True,
        classifier_count=1,
        segmentation_count=9,
        model_count=10,
        models=record['models'],
        summary=str(root / 'final_evaluation/summary.json'),
        report=str(root / 'final_evaluation/REPORT.md'),
    )
    if args.command == 'plan':
        result['action'] = 'read_frozen_final_results_no_training'
    elif args.command == 'verify':
        result.update(
            all_artifact_hashes_match=True,
            new_training=False,
            test_used=False,
            inference=getattr(args, 'inference', False),
        )
        if result['inference']:
            result['model_reload_comparisons'] = _inference_checks(root, record)
    elif args.command in {'stage6-evaluate', 'stage7-report'}:
        result['action'] = 'reuse_frozen_results'
    return result
