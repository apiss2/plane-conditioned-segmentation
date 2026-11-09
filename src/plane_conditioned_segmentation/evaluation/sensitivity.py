"""Validation-only condition sweeps; never select a checkpoint or evaluate test images."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ..common import sha256_file, utc_now, write_json


def condition_cases(labels, probabilities):
    labels = np.asarray(labels)
    q = np.asarray(probabilities, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] != len(labels) or q.shape[1] < 2:
        raise ValueError('Aligned labels and class probabilities required')
    if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError('Finite normalized class probabilities required')
    if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= q.shape[1])).any():
        raise ValueError('Integer target labels required')
    eye = np.eye(q.shape[1], dtype=np.float32)
    oracle = eye[labels]
    yield 'oracle', oracle
    yield 'estimated_hard', eye[q.argmax(1)]
    yield 'estimated_soft', q
    for epsilon in [0.05, 0.1, 0.2, 0.4, 1.0]:
        yield f'uniform_{epsilon}', (1 - epsilon) * oracle + epsilon / len(eye)
    for offset in range(1, len(eye)):
        other = eye[(labels + offset) % len(eye)]
        for epsilon in [0.1, 0.2, 0.4, 1.0]:
            yield f'pair_{offset}_{epsilon}', (1 - epsilon) * oracle + epsilon * other


def failure_summary(rows, reference):
    reference = {r['index']: r for r in reference}
    if len(reference) != len(rows) or {r['index'] for r in rows} != reference.keys():
        raise ValueError('One matched reference per image required')
    dice = np.array([r['dice'] for r in rows])
    drop = np.array([reference[r['index']]['dice'] - r['dice'] for r in rows])
    area_change = [
        abs(r['target_predicted_pixels'] - reference[r['index']]['target_predicted_pixels'])
        / max(reference[r['index']]['target_predicted_pixels'], 1)
        for r in rows
    ]
    return dict(
        images=len(rows),
        dice_p10=float(np.quantile(dice, 0.1)),
        dice_median=float(np.median(dice)),
        dice_below_0_2=int((dice < 0.2).sum()),
        empty_masks=sum(r['target_predicted_pixels'] == 0 for r in rows),
        drops_over_0_1=int((drop > 0.1).sum()),
        mean_dice_drop=float(drop.mean()),
        mean_relative_area_change=float(np.mean(area_change)),
    )


def run(output, models, classifier_path):
    import torch

    from ..data.dataset import DATA, ROOT, ids_for, plane_ids
    from ..models.classifier import load_classifier
    from ..models.segmentation import load
    from .classification import evaluate as classify
    from .metrics import evaluate

    ids = ids_for(1)
    if not len(ids) or not (np.load(DATA / 'split.npy')[ids] == 1).all():
        raise ValueError('Nonempty validation-only cohort required')
    output = Path(output).resolve()
    if output.is_relative_to(ROOT.resolve()) and (ROOT / 'FINAL_RESULTS.json').exists():
        raise ValueError('Final bundle is read-only; use a separate diagnostic directory')
    output.mkdir(parents=True, exist_ok=False)
    labels = plane_ids(ids)
    model_paths = {name: ROOT / f'full/{name}/best.pt' for name in models}
    manifest = dict(
        utc=utc_now(),
        experiment=ROOT.name,
        split='val',
        test_used=False,
        validation_ids=ids.tolist(),
        validation_ids_sha256=hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest(),
        protocol_sha256=sha256_file(ROOT / 'study_plan.json'),
        cohort_sha256=sha256_file(ROOT / 'cohort.json'),
        classifier_sha256=sha256_file(classifier_path),
        checkpoints={k: sha256_file(v) for k, v in model_paths.items()},
        note='Sensitivity diagnosis only. Fully wrong conditions are stress cases, not automatic rejection failures.',
    )
    write_json(output / 'manifest.json', manifest)
    classifier = load_classifier(classifier_path, 'cuda')
    classified = classify(classifier, ids)
    write_json(output / 'classifier_validation.json', classified)
    predictions = {r['index']: r for r in classified['rows']}
    probabilities = np.array([predictions[int(i)]['probabilities'] for i in ids])
    del classifier
    torch.cuda.empty_cache()
    summaries = {}
    for name, path in model_paths.items():
        model = load(path, 'cuda')
        reference = None
        for case, conditions in condition_cases(labels, probabilities):
            if name == 'plain' and case != 'oracle':
                continue
            result = evaluate(model, ids, dict(zip(map(int, ids), conditions)))
            if reference is None:
                reference = result['rows']
            result['failures'] = failure_summary(result['rows'], reference)
            result['failures_per_target'] = {
                str(p): failure_summary(
                    [r for r in result['rows'] if r['plane'] == p], [r for r in reference if r['plane'] == p]
                )
                for p in sorted(set(labels.tolist()))
            }
            key = f'{name}_{case}'
            write_json(output / f'{key}.json', result)
            summaries[key] = dict(
                scores=result['summary'], failures=result['failures'], failures_per_target=result['failures_per_target']
            )
            write_json(output / 'progress.json', dict(utc=utc_now(), completed=list(summaries)))
            print(key, result['summary']['macro_dice'], result['failures'], flush=True)
        del model
        torch.cuda.empty_cache()
    if any(sha256_file(path) != manifest['checkpoints'][name] for name, path in model_paths.items()):
        raise ValueError('Checkpoint changed during diagnosis')
    write_json(output / 'summary.json', dict(manifest=manifest, conditions=summaries, complete=True))
    return summaries


def main():
    import torch

    from ..config import load_runtime
    from ..paths import experiment_root
    from ..pipeline.stages import execution_lock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--models', nargs='+', choices=['plain', 'film', 'condconv'], default=['plain', 'film', 'condconv']
    )
    parser.add_argument('--classifier-experiment', default='final')
    args = parser.parse_args()
    torch.set_num_threads(load_runtime().torch_threads)
    with execution_lock():
        run(args.output, args.models, experiment_root(args.classifier_experiment) / 'full/classifier/best.pt')


if __name__ == '__main__':
    main()
