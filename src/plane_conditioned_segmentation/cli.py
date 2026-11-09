"""Command line: read-only plans by default; explicit execution reuses frozen completed models."""

import argparse
import json
import os
from pathlib import Path

from .common import NUM_TARGETS
from .paths import DEFAULT_EXPERIMENT, REPO_ROOT, experiment_root, prepared_data_root
from .pipeline.stages import STAGES, evaluate_stage, execute_stage, execution_lock, report_stage, stage_plan


def build_parser():
    parser = argparse.ArgumentParser(
        prog='plane-seg',
        description='Read-only final results by default; choose a separate --experiment for new training.',
    )
    parser.add_argument('--experiment', default=os.environ.get('PCS_EXPERIMENT', DEFAULT_EXPERIMENT))
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-data', help='Build private cache from explicit ZIP polygon manifest; never infer split')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--definition', type=Path, default=REPO_ROOT / 'configs/data_definition.json')
    p.add_argument('--protocol', type=Path, default=REPO_ROOT / 'configs/sevenplane_protocol.json')
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
            p.add_argument('--target', type=int, choices=range(NUM_TARGETS), help='One target; default all seven')
    for name in ['stage6-evaluate', 'stage7-report']:
        p = sub.add_parser(name)
        p.add_argument('--plan', action='store_true')
        p.add_argument('--execute', action='store_true')
    p = sub.add_parser('verify', help='Read/hash checks; optional minimal CPU inference on validation only')
    p.add_argument('--inference', action='store_true')
    p.add_argument('--forbid-source', type=Path, help='Deny reads from an original source during verification')
    return parser


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
        stages=[stage_plan(root, s) for s in STAGES],
        final_test='stage6-evaluate after all frozen models',
        report='stage7-report',
    )


def _verify(root, args):
    from .pipeline.verification import verify

    return verify(root, args.inference, args.forbid_source)


COMMANDS = {
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
