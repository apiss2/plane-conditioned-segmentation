"""Prepare/run three fresh FiLM treatments sequentially; validation only, no automatic adoption."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from plane_conditioned_segmentation.common import read_json, sha256_file, utc_now, write_json
from plane_conditioned_segmentation.config import conditioning_from_plan, load_protocol
from plane_conditioned_segmentation.paths import REPO_ROOT, experiment_root

TREATMENTS = {
    'onehot': dict(mode='onehot', epsilon=0.0, clean_probability=1.0),
    'smooth': dict(mode='uniform', epsilon=0.1, clean_probability=0.0),
    'noise': dict(mode='random_pair', epsilon=0.4, clean_probability=0.5),
}


def training_plan(plan, condition):
    plan = {
        **plan,
        'condition_training': condition,
        'new_training_models': ['film'],
        'comparison': 'Fresh FiLM condition ablation; validation only; fixed inherited classifier for diagnosis',
        'test_evaluation_permitted': False,
    }
    plan['schedulefree'] = {**plan['schedulefree'], 'max_epochs': 500, 'min_epochs': 1, 'min_applied_updates': 0}
    plan['full_segmentation_rule'] = {**plan['full_segmentation_rule'], 'min_epochs': 1, 'min_attempted_updates': 0}
    plan.pop('provisional_reference_experts', None)
    plan['early_stopping_definition'] = (
        'Every epoch, strict oracle validation selection, patience30, cap500, no extra minima'
    )
    plan['comparison_schedule'] = 'FiLM only, then validation sensitivity; no test or final-evaluation stage'
    conditioning_from_plan(plan)
    return plan


def prepare(output, prefix):
    from plane_conditioned_segmentation.pipeline.radam import prepare as prepare_run
    from plane_conditioned_segmentation.pipeline.results import validate

    source = experiment_root('final')
    validate(source)
    if (output / 'ablation_plan.json').exists():
        raise ValueError('Ablation plan exists; no overwrite')
    roots = {key: experiment_root(f'{prefix}_{key}') for key in TREATMENTS}
    for root in roots.values():
        prepare_run(root, source)  # All destinations checked before any mutation.
    for key, root in roots.items():
        prepare_run(root, source, True)
        plan = training_plan(read_json(root / 'study_plan.json'), TREATMENTS[key])
        write_json(root / 'study_plan.json', plan)
        load_protocol(root / 'study_plan.json')
    files = [
        *REPO_ROOT.glob('src/**/*.py'),
        *REPO_ROOT.glob('configs/*.json'),
        REPO_ROOT / 'uv.lock',
        REPO_ROOT / 'pyproject.toml',
        Path(__file__).resolve(),
    ]
    hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in files}
    for p in files:
        dest = output / 'execution_source' / p.relative_to(REPO_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    record = dict(
        utc=utc_now(),
        source='final',
        treatments={
            k: dict(experiment=r.name, condition=TREATMENTS[k], protocol_sha256=sha256_file(r / 'study_plan.json'))
            for k, r in roots.items()
        },
        source_sha256=hashes,
        seed=17,
        max_epochs=500,
        patience=30,
        selection='unchanged oracle validation selection',
        test_used=False,
        retrain_classifier=False,
        retrain_experts=False,
        tuning_after_test=False,
        quality_policy='Retain original sufficiency failure as a warning for research comparison; never adopt automatically',
        screening_rule=dict(
            maximum_oracle_macro_dice_drop=0.01,
            require_fewer_estimated_soft_dice_below_0_2=True,
            require_nonworse_estimated_soft_macro_dice=True,
        ),
        pending=[
            'human correction-time evaluation',
            'new independent patient/site test',
            'multiple seeds for promising treatments',
            'OOF classifier-shaped conditions if needed',
        ],
    )
    write_json(output / 'ablation_plan.json', record)
    return record


def train_worker(name):
    os.environ['PCS_EXPERIMENT'] = name
    root = experiment_root(name)
    plan = read_json(root / 'study_plan.json')
    if (root / 'FINAL_RESULTS.json').exists() or plan.get('new_training_models') != ['film']:
        raise ValueError('Only a prepared isolated FiLM ablation can be trained')
    if conditioning_from_plan(plan) is None or plan.get('test_evaluation_permitted') is not False:
        raise ValueError('Explicit validation-only condition policy required')
    from plane_conditioned_segmentation.pipeline.stages import execution_lock
    from plane_conditioned_segmentation.training.schedulefree_training import train

    with execution_lock():
        try:
            train('film', 'film')
        except RuntimeError as error:
            if error.args != (('Terminal learning sufficiency requires review; thresholds not changed', 'film'),):
                raise
            result = read_json(root / 'full/film/result.json')
            if result.get('learning_sufficiency_pass') is not False:
                raise
            write_json(
                root / 'ABLATION_QUALITY_WARNING.json',
                dict(utc=utc_now(), error=str(error), adopted=False, original_gate_unchanged=True),
            )


def verify_inputs(record):
    for relative, digest in record['source_sha256'].items():
        if sha256_file(REPO_ROOT / relative) != digest:
            raise ValueError('Execution source changed: ' + relative)
    for treatment in record['treatments'].values():
        root = experiment_root(treatment['experiment'])
        if sha256_file(root / 'study_plan.json') != treatment['protocol_sha256']:
            raise ValueError('Frozen treatment changed')


def comparison(output, record):
    scores = {key: read_json(output / f'validation_{key}/summary.json') for key in TREATMENTS}
    reference = scores['onehot']['manifest']
    for score in scores.values():
        if (
            not score.get('complete')
            or any(
                score['manifest'][key] != reference[key]
                for key in ['split', 'validation_ids_sha256', 'cohort_sha256', 'classifier_sha256']
            )
            or score['manifest']['split'] != 'val'
        ):
            raise ValueError('Comparison requires identical validation cohort and fixed classifier')
    baseline = scores['onehot']['conditions']
    table = {}
    for key, score in scores.items():
        cases = score['conditions']
        oracle = cases['film_oracle']['scores']['macro_dice']
        soft = cases['film_estimated_soft']
        table[key] = dict(
            oracle_macro_dice=oracle,
            soft_macro_dice=soft['scores']['macro_dice'],
            soft_severe_failures=soft['failures']['dice_below_0_2'],
            passes_exploratory_screen=key != 'onehot'
            and oracle >= baseline['film_oracle']['scores']['macro_dice'] - 0.01
            and soft['scores']['macro_dice'] >= baseline['film_estimated_soft']['scores']['macro_dice']
            and soft['failures']['dice_below_0_2'] < baseline['film_estimated_soft']['failures']['dice_below_0_2'],
            quality=read_json(experiment_root(record['treatments'][key]['experiment']) / 'full/film/result.json')[
                'learning_sufficiency_pass'
            ],
        )
    write_json(
        output / 'comparison.json',
        dict(utc=utc_now(), treatments=table, test_used=False, automatic_adoption=False, pending=record['pending']),
    )


def execute(output):
    record = read_json(output / 'ablation_plan.json')
    baseline = read_json(output / 'baseline/summary.json')
    if baseline.get('complete') is not True or baseline['manifest']['split'] != 'val':
        raise ValueError('Complete baseline validation sensitivity required before training')
    lock = output / 'ablation_process.lock'
    with lock.open('x', encoding='utf-8') as stream:
        import json

        json.dump(dict(pid=os.getpid(), utc=utc_now()), stream)
    completed = []
    try:
        for key, treatment in record['treatments'].items():
            name = treatment['experiment']
            root = experiment_root(name)
            env = {**os.environ, 'PCS_EXPERIMENT': name, 'PYTHONUTF8': '1'}
            for stage, args in [
                ('train', [str(Path(__file__).resolve()), '--train-experiment', name]),
                (
                    'validation',
                    [
                        '-m',
                        'plane_conditioned_segmentation.evaluation.sensitivity',
                        '--models',
                        'film',
                        '--output',
                        str(output / f'validation_{key}'),
                    ],
                ),
            ]:
                verify_inputs(record)
                if (root / 'STOP_REQUESTED.json').exists() or (output / 'STOP_REQUESTED.json').exists():
                    raise RuntimeError('Stop requested; no next stage')
                write_json(
                    output / 'status.json',
                    dict(
                        utc=utc_now(),
                        pid=os.getpid(),
                        state='running',
                        treatment=key,
                        stage=stage,
                        experiment=name,
                        completed=completed,
                    ),
                )
                with (output / f'{key}_{stage}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(
                        [sys.executable, *args],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                verify_inputs(record)
                completed.append(f'{key}_{stage}')
        comparison(output, record)
        write_json(
            output / 'status.json',
            dict(utc=utc_now(), state='screening_complete', completed=completed, pending=record['pending']),
        )
    except BaseException as error:
        write_json(output / 'status.json', dict(utc=utc_now(), state='stopped', completed=completed, error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prefix', default='film_condition_20261109')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--train-experiment')
    args = parser.parse_args()
    if args.train_experiment:
        train_worker(args.train_experiment)
    elif args.output is None:
        parser.error('--output required')
    elif args.prepare:
        prepare(args.output.resolve(), args.prefix)
    else:
        execute(args.output.resolve())


if __name__ == '__main__':
    main()
