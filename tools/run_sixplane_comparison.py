"""Run the seven existing CLI stages once, sequentially, without retries or test-based tuning."""

import json
import os
import subprocess
import sys
from pathlib import Path

from plane_conditioned_segmentation.common import read_json, sha256_file, utc_now, write_json
from plane_conditioned_segmentation.paths import REPO_ROOT, experiment_root


def main():

    if Path(sys.prefix).resolve() != REPO_ROOT / '.venv':
        raise ValueError('Use the repository root .venv through uv run')

    root = experiment_root()

    plan = read_json(root / 'study_plan.json')

    if not plan.get('reuse_experts_only') and not plan.get('schedulefree'):
        raise ValueError('This driver requires an explicit prepared six-target protocol')

    lock = root / 'comparison_process.lock'

    with lock.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), utc=utc_now()), f)

    source_files = [
        *REPO_ROOT.glob('src/**/*.py'),
        *REPO_ROOT.glob('configs/*.json'),
        REPO_ROOT / 'pyproject.toml',
        REPO_ROOT / 'uv.lock',
        Path(__file__).resolve(),
    ]

    review = root / 'budget_cap_review.json'
    if review.exists() and not plan.get('continued_models'):
        source_files.append(review)

    stopping_override = root / 'stopping_policy_override.json'
    if stopping_override.exists():
        source_files.append(stopping_override)

    source_hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in source_files}

    source_record = root / 'execution_source_manifest.json'

    if source_record.exists():
        if read_json(source_record)['sha256'] != source_hashes:
            raise ValueError('Run code changed; review before continuing')

    else:
        write_json(
            source_record, dict(utc=utc_now(), sha256=source_hashes, python=sys.version, executable=sys.executable)
        )

    run_hashes = read_json(source_record).get('run_sha256', {})
    if any(sha256_file(root / p) != h for p, h in run_hashes.items()):
        raise ValueError('Run continuation review changed')

    from plane_conditioned_segmentation.cli import main as cli

    completed = []

    try:
        order = plan.get(
            'training_stage_order',
            ['stage1-classifier', 'stage2-experts', 'stage3-baseline', 'stage4-film', 'stage5-condconv'],
        )
        for stage in [*order, 'stage6-evaluate', 'stage7-report']:
            if (root / 'STOP_REQUESTED.json').exists():
                raise RuntimeError('Stop marker; no next stage or evaluation')

            if any(sha256_file(root / p) != h for p, h in run_hashes.items()):
                raise ValueError('Run continuation review changed; no next stage started')

            if any(sha256_file(REPO_ROOT / p) != h for p, h in source_hashes.items()):
                raise ValueError('Execution source changed; no next stage started')

            write_json(
                root / 'comparison_status.json',
                dict(pid=os.getpid(), utc=utc_now(), current_stage=stage, completed_stages=completed),
            )

            if stage == 'stage2-experts' and plan.get('provisional_reference_experts'):
                from plane_conditioned_segmentation.pipeline.provisional import prepare as provisional

                provisional_root = provisional(root)
                for action in ['stage6-evaluate', 'stage7-report']:
                    with (provisional_root / 'console.log').open('a', encoding='utf-8') as output:
                        subprocess.run(
                            [
                                sys.executable,
                                '-m',
                                'plane_conditioned_segmentation.cli',
                                '--experiment',
                                provisional_root.name,
                                action,
                                '--execute',
                            ],
                            check=True,
                            env={**os.environ, 'PCS_EXPERIMENT': provisional_root.name},
                            stdout=output,
                            stderr=subprocess.STDOUT,
                        )
                write_json(
                    root / 'provisional_evaluation.json',
                    dict(
                        utc=utc_now(),
                        run=provisional_root.name,
                        caveat='Old AdamW experts vs new Schedule-Free classifier/shared models; provisional mixed-optimizer comparison, not final',
                    ),
                )
            if stage == 'stage2-experts' and stopping_override.exists():
                from plane_conditioned_segmentation.pipeline.stages import existing_model

                for label in read_json(stopping_override).get('resume_models', []):
                    if existing_model(root, label)['state'] == 'partial':
                        code = cli(['--experiment', root.name, 'resume-model', '--model', label, '--execute'])
                        if code:
                            raise RuntimeError(f'Explicit policy-transition resume failed: {label}')
            if plan.get('continued_models'):
                from plane_conditioned_segmentation.pipeline.stages import existing_model, stage_models

                for label in stage_models(root).get(stage, ()):
                    if label in plan['continued_models'] and existing_model(root, label)['state'] == 'partial':
                        code = cli(['--experiment', root.name, 'resume-model', '--model', label, '--execute'])
                        if code:
                            raise RuntimeError(f'Continuation resume failed: {label}')
            code = cli(['--experiment', root.name, stage, '--execute'])

            if code:
                raise RuntimeError(f'{stage}: exit {code}')

            completed.append(stage)

        write_json(
            root / 'COMPARISON_COMPLETE.json',
            dict(
                utc=utc_now(),
                stages=completed,
                reused_experts=6 if plan.get('reuse_experts_only') else 0,
                newly_trained_models=0
                if plan.get('continued_models')
                else (4 if plan.get('reuse_experts_only') else 10),
                continued_models=plan.get('continued_models', []),
                inherited_models=plan.get('inherited_models', []),
                target_count=6,
                seed=17,
            ),
        )

    except BaseException as exc:
        write_json(root / 'COMPARISON_STOPPED.json', dict(utc=utc_now(), stages=completed, error=repr(exc)))

        raise

    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
