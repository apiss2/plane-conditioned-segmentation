"""Command line: read-only plans by default; explicit execution reuses frozen completed models."""

import argparse
import json
import os
from pathlib import Path

from .paths import DEFAULT_EXPERIMENT, REPO_ROOT, experiment_root, prepared_data_root
from .pipeline.stages import (
    STAGES,
    evaluate_stage,
    execute_stage,
    execution_lock,
    report_stage,
    stage_models,
    stage_plan,
)


def build_parser():

    parser = argparse.ArgumentParser(
        prog='plane-seg',
        description='Read-only final results by default; choose a separate --experiment for new training.',
    )

    parser.add_argument('--experiment', default=os.environ.get('PCS_EXPERIMENT', DEFAULT_EXPERIMENT))

    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser(
        'prepare-radam', help='Prepare six-target fresh RAdamScheduleFree comparison; no trained weight reuse'
    )

    p.add_argument('--source-experiment', default='sixplane_20261104')

    p.add_argument('--execute', action='store_true')

    p = sub.add_parser('prepare-sixplane', help='Derive CSP-free cohort and verify/reuse frozen experts; no retraining')

    p.add_argument('--source-experiment', default='sevenplane_20261031')

    p.add_argument('--execute', action='store_true')

    p = sub.add_parser('prepare-continuation', help='Fork compatible terminal states into a separate approved continuation run')

    p.add_argument('--source-experiment', default='sixplane_radam_sf_20261104')

    p.add_argument('--max-epochs', type=int, choices=(300, 500), default=300)

    p.add_argument('--execute', action='store_true')

    p = sub.add_parser('resume-model', help='Explicit compatible Schedule-Free resume only; no completed overwrite')

    p.add_argument('--model', required=True)

    p.add_argument('--execute', action='store_true')

    p = sub.add_parser('prepare-data', help='Build private cache from explicit ZIP polygon manifest; never infer split')

    p.add_argument('--manifest', type=Path, required=True)

    p.add_argument('--definition', type=Path, default=REPO_ROOT / 'configs/sixplane_data_definition.json')

    p.add_argument('--protocol', type=Path, default=REPO_ROOT / 'configs/sixplane_radam_protocol.json')

    p.add_argument('--archives-dir', type=Path, default=os.environ.get('PCS_ARCHIVES_DIR'))

    p.add_argument('--data-dir', type=Path, default=os.environ.get('PCS_DATA_DIR'))

    p = sub.add_parser(
        'initialize-models', help='Create fresh common initialization from an explicitly supplied grayscale encoder'
    )

    p.add_argument('--encoder', type=Path, required=True)

    p.add_argument('--metadata', type=Path, required=True)

    sub.add_parser('check-data', help='Read cache hashes, split counts and one training sample')

    sub.add_parser('status', help='Show selected final models or historical training stages')

    p = sub.add_parser(
        'register-existing', help='Register already imported completed artifacts without training or evaluation'
    )

    p.add_argument('--execute', action='store_true', help='Write new run_manifest.json only; never overwrite')

    sub.add_parser('plan', help='Show final result inventory or a separate experiment plan')

    for name in STAGES:
        p = sub.add_parser(name, help='Plan or explicitly execute this stage, reusing existing frozen models')

        p.add_argument('--plan', action='store_true', help='Read-only plan (default)')

        p.add_argument('--execute', action='store_true', help='Execute missing jobs only; refuse partial/failed runs')

        if name == 'stage2-experts':
            p.add_argument('--target', type=int, help='One target in the selected run; default all targets')

    for name in ['stage6-evaluate', 'stage7-report']:
        p = sub.add_parser(name)

        p.add_argument('--plan', action='store_true')

        p.add_argument('--execute', action='store_true')

    p = sub.add_parser('verify', help='Read/hash checks; optional minimal CPU inference on validation only')

    p.add_argument('--inference', action='store_true')

    p.add_argument('--forbid-source', type=Path, help='Deny reads from an original source during verification')

    return parser


def _resume_model(root, args):

    from .common import read_json, sha256_file
    from .pipeline.stages import existing_model, prerequisites, stage_models

    stage = next((s for s, names in stage_models(root).items() if args.model in names), None)

    if stage is None or existing_model(root, args.model)['state'] != 'partial':
        raise ValueError('Only an explicit partial model can be resumed')

    prerequisites(root, stage)

    if not args.execute:
        return dict(action='resume_partial_only', model=args.model, read_only=True)

    manifest = root / 'execution_source_manifest.json'

    if manifest.exists() and any(sha256_file(REPO_ROOT / p) != h for p, h in read_json(manifest)['sha256'].items()):
        raise ValueError('Execution source changed; no partial resume')

    from .training.schedulefree_training import train

    target = int(args.model.split('_')[1]) if args.model.startswith('expert_') else None

    mode = 'expert' if target is not None else args.model

    with execution_lock():
        try:
            return train(args.model, mode, target, resume=True)
        except RuntimeError as error:
            expected = (('Terminal learning sufficiency requires review; thresholds not changed', args.model),)
            if error.args != expected:
                raise
            state = existing_model(root, args.model)
            if state['action'] != 'reuse' or not (state.get('quality_note') or '').startswith(
                ('Budget-capped nonconvergence warning:', 'Validation-patience nonconvergence warning:')
            ):
                raise
            return state


def _prepare_continuation(root, args):
    from .pipeline.continuation import prepare

    source = experiment_root(args.source_experiment)
    if not args.execute:
        return prepare(root, source, max_epochs=args.max_epochs)
    with execution_lock():
        return prepare(root, source, True, max_epochs=args.max_epochs)


def _prepare_radam(root, args):

    from .pipeline.radam import prepare

    source = experiment_root(args.source_experiment)

    if not args.execute:
        return prepare(root, source)

    with execution_lock():
        return prepare(root, source, True)


def _prepare_sixplane(root, args):

    from .pipeline.subset import prepare_sixplane

    source = experiment_root(args.source_experiment)

    if not args.execute:
        return prepare_sixplane(root, source)

    with execution_lock():
        return prepare_sixplane(root, source, True)


def _prepare_data(root, args):

    if not args.archives_dir:
        raise ValueError('Set PCS_ARCHIVES_DIR or --archives-dir')

    from .data.preparation import prepare

    return prepare(
        args.manifest,
        args.definition,
        root,
        args.data_dir or prepared_data_root(root),
        args.archives_dir,
        args.protocol,
    )


def _initialize_models(root, args):

    from .data.preparation import initialize
    from .paths import initialization_root

    return initialize(initialization_root(root), args.encoder, args.metadata)


def _check_data(root, args):

    from .data.dataset import DATA, SOURCE_ARCHIVES, Images, ids_for

    ids = ids_for(0)

    if not len(ids):
        raise ValueError('No segmentation training samples')

    a, y, v, p, _ = Images(ids[:1])[0]

    return dict(
        data=str(DATA),
        archives=str(SOURCE_ARCHIVES),
        segmentation_counts=[len(ids_for(i)) for i in range(3)],
        classification_counts=[len(ids_for(i, classification=True)) for i in range(3)],
        first_train_shape=list(a.shape),
        target=p,
        valid_pixels=int(v.sum()),
        target_pixels=int(y.sum()),
    )


def _status(root, args):

    from .pipeline.registry import describe

    return describe(root)


def _register_existing(root, args):

    from .pipeline.registry import registration

    if not args.execute:
        return registration(root)

    with execution_lock():
        return registration(root, True)


def _plan(root, args):

    return dict(
        stages=[stage_plan(root, s) for s in stage_models(root)],
        final_test='stage6-evaluate after all frozen models',
        report='stage7-report',
    )


def _verify(root, args):

    from .pipeline.verification import verify

    return verify(root, args.inference, args.forbid_source)


COMMANDS = {
    'prepare-sixplane': _prepare_sixplane,
    'prepare-radam': _prepare_radam,
    'prepare-continuation': _prepare_continuation,
    'resume-model': _resume_model,
    'prepare-data': _prepare_data,
    'initialize-models': _initialize_models,
    'check-data': _check_data,
    'status': _status,
    'register-existing': _register_existing,
    'plan': _plan,
    'stage6-evaluate': lambda root, args: evaluate_stage(root, args.execute),
    'stage7-report': lambda root, args: report_stage(root, args.execute),
    'verify': _verify,
}


def _run_stage(root, args):

    target = getattr(args, 'target', None)

    if args.execute:
        return execute_stage(root, args.command, target)

    return stage_plan(root, args.command, target)


def main(argv=None):

    parser = build_parser()

    args = parser.parse_args(argv)

    try:
        root = experiment_root(args.experiment)

        os.environ['PCS_EXPERIMENT'] = args.experiment

        if getattr(args, 'execute', False) and getattr(args, 'plan', False):
            parser.error('--plan and --execute are mutually exclusive')

        handler = _run_stage if args.command in STAGES else COMMANDS[args.command]

        from .paths import is_archived_run
        from .pipeline.results import is_final_result, run_command

        if is_archived_run(root) and (
            getattr(args, 'execute', False) or args.command in {'prepare-data', 'initialize-models'}
        ):
            raise ValueError('Archived runs are read-only; prepare a new experiment instead')
        result = run_command(root, args) if is_final_result(root) else handler(root, args)

        print(json.dumps(result, ensure_ascii=False, indent=2))

        return 0

    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
