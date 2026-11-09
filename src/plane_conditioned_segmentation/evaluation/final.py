"""Frozen-checkpoint test evaluation: every model and condition, original-grid scores, timing and baseline."""

import gc
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ..common import LOCK_FILE, SEED, full_model_count, num_targets, read_json
from ..common import sha256_file as sha
from ..common import write_json as dump
from ..data.batch import to_unit
from ..data.dataset import (
    CLASSES,
    DATA,
    OLD_CLASSES,
    PROTOCOL,
    ROOT,
    SOURCE_ARCHIVES,
    SPEC,
    Images,
    ids_for,
    plane_ids,
    read_geometry,
)
from ..models.classifier import load_classifier
from ..models.segmentation import load
from ..training.runtime import RUNTIME, event, resources
from .classification import evaluate as classify
from .metrics import EPS, evaluate, metric_row, metrics, summarize

SHARED_MODES = ['plain', 'film', 'condconv']
BENCHMARK_WARMUP = 4
BENCHMARK_REPEATS = 15
BENCHMARK_LOADS = 6


def _original_protocol():
    e = PROTOCOL.evaluation
    return (
        f'Same original-polygon GT. Remove letterbox padding, {e.original_grid_interpolation} probabilities '
        f'align_corners=False, then threshold>{str(e.probability_threshold).removeprefix("0")}. '
        'No target polygons unioned with other anatomies.'
    )


def _release_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def _one_hot(p, device='cuda'):
    return F.one_hot(torch.tensor([p], device=device), num_targets()).float()


def _dice(tp, predicted, truth):
    return (2 * tp + EPS) / (predicted + truth + EPS)


def _macro(rows, planes):
    return float(np.mean([np.mean([r['dice'] for r in rows if r['plane'] == p]) for p in planes]))


def _label_members(archive):
    """YOLO label members by image stem, for manifest records that do not name their label."""
    return {
        Path(n).stem: n
        for n in archive.namelist()
        if '/labels/' in n and n.endswith('.txt') and not n.startswith('__MACOSX/')
    }


def _original_prediction(m, a, p, g):
    """7xHxW boolean prediction on the original image grid: crop letterbox, resize probabilities, threshold."""
    prob = m(to_unit(a[None]), _one_hot(p)).sigmoid()
    prob = prob[:, :, g['top'] : g['top'] + g['content_height'], g['left'] : g['left'] + g['content_width']]
    size = (g['original_height'], g['original_width'])
    e = PROTOCOL.evaluation
    prob = F.interpolate(prob, size=size, mode=e.original_grid_interpolation, align_corners=False)
    pred = prob[0].cpu().numpy() > e.probability_threshold
    if m.target is not None:
        full = np.zeros((num_targets(), *pred.shape[-2:]), bool)
        full[m.target] = pred[0]
        pred = full
    return pred


def _original_truth(text, local, g):
    """Rasterize only the polygons of source class ``local`` at original resolution."""
    width, height = g['original_width'], g['original_height']
    mask = Image.new('1', (width, height))
    for line in text.splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if int(values[0]) == local:
            xy = np.array(values[1:]).reshape(-1, 2)
            ImageDraw.Draw(mask).polygon([(float(x * (width - 1)), float(y * (height - 1))) for x, y in xy], fill=1)
    return np.asarray(mask)


@torch.inference_mode()
def common_original(m, ids):
    data = Images(ids)
    audit = read_json(DATA / 'audit.json')
    records = read_json(DATA / 'manifest.json')
    geometry = read_geometry(DATA)
    archives = {r['source_plane']: zipfile.ZipFile(SOURCE_ARCHIVES / r['source_plane']) for r in SPEC['targets']}
    try:
        labels = {name: _label_members(z) for name, z in archives.items()}
        rows = []
        for j in range(len(ids)):
            a, _y, _v, p, idx = data[j]
            g = geometry[idx]
            pred = _original_prediction(m, a, p, g)
            record = records[idx]
            archive = record['archive']
            local = audit['class_map'][archive].index(audit['classes'][OLD_CLASSES[p]])
            member = record.get('label') or labels[archive][Path(record['name']).stem]
            gt = _original_truth(archives[archive].read(member).decode(), local, g)
            assert gt.any()
            tp = int((pred[p] & gt).sum())
            tc = int(gt.sum())
            rows.append(
                dict(
                    index=int(idx),
                    plane=p,
                    dice=_dice(tp, int(pred[p].sum()), tc),
                    global_dice=_dice(tp, int(pred.sum()), tc),
                )
            )
    finally:
        for z in archives.values():
            z.close()
    return dict(
        rows=rows,
        dice=float(np.mean([r['dice'] for r in rows])),
        global_dice=float(np.mean([r['global_dice'] for r in rows])),
        macro_dice=_macro(rows, sorted(set(r['plane'] for r in rows))),
        protocol=_original_protocol(),
    )


def _timings(forward, device):
    for _ in range(BENCHMARK_WARMUP):
        forward()
    times = []
    for _ in range(BENCHMARK_REPEATS):
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        forward()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
    return times


@torch.inference_mode()
def benchmark(path, classifier=False):
    make = load_classifier if classifier else load
    m = make(path).eval()
    outputs = []
    torch.set_num_threads(1)
    for device in ['cpu', 'cuda']:
        m.to(device)
        for width, height in PROTOCOL.inputs:
            a = torch.zeros(1, 1, height, width, device=device)
            c = _one_hot(0, device)
            times = _timings((lambda: m(a)) if classifier else (lambda: m(a, c)), device)
            outputs.append(
                dict(
                    device=device,
                    width=width,
                    height=height,
                    median_ms=float(np.median(times)),
                    p95_ms=float(np.quantile(times, 0.95)),
                )
            )
    del m
    torch.cuda.empty_cache()
    torch.set_num_threads(RUNTIME.torch_threads)
    times = []
    for _ in range(BENCHMARK_LOADS):
        started = time.perf_counter()
        m = make(path)
        times.append((time.perf_counter() - started) * 1000)
        del m
    return dict(
        forward=outputs,
        warm_load_median_ms=float(np.median(times)),
        warm_load_p95_ms=float(np.quantile(times, 0.95)),
        checkpoint_bytes=path.stat().st_size,
        note='Current host only; CPU1thread, FP32; warm filesystem. Includes normalization, excludes image '
        'decode, Python/framework startup. Not measured on target deployment device.',
    )


def _expert_path(p):
    return ROOT / f'full/expert_{p}/best.pt'


@torch.inference_mode()
def routed_experts(ids, routing):
    """Send each image to the expert chosen by ``routing[index]`` and score the union."""
    parts = []
    for p in range(num_targets()):
        subset = np.array([i for i in ids if routing[int(i)] == p], np.int64)
        if not len(subset):
            continue
        m = load(_expert_path(p), 'cuda')
        parts.append(evaluate(m, subset))
        del m
        _release_gpu()
    rows = sum([r['rows'] for r in parts], [])
    rows.sort(key=lambda r: r['index'])
    return dict(rows=rows, summary=summarize(rows))


@torch.inference_mode()
def selected_output(m, ids, conditions, selected):
    """Keep only channel ``selected[index]`` of a shared model; all other channels are forced negative."""
    data = Images(ids)
    rows = []
    for j in range(len(ids)):
        a, y, v, p, idx = data[j]
        q = selected[int(idx)]
        c = torch.tensor(conditions[int(idx)], device='cuda').float()[None]
        z = m(to_unit(a[None]), c)
        kept = torch.full_like(z, -100)
        kept[:, q] = z[:, q]
        r = metrics(
            kept,
            y[None].cuda(),
            v[None].cuda(),
            torch.tensor([p], device='cuda'),
            threshold=PROTOCOL.evaluation.probability_threshold,
        )
        rows.append(metric_row(r, 0, idx, p))
    return dict(rows=rows, summary=summarize(rows))


def _verify_lock():
    lock = read_json(ROOT / LOCK_FILE)
    for filename, key in [('study_plan.json', 'study_plan_sha256'), ('cohort.json', 'cohort_sha256')]:
        if sha(ROOT / filename) != lock.get(key):
            raise ValueError('Frozen evaluation input changed: ' + filename)
    assert lock['full_training_models'] == full_model_count()
    assert all(sha(ROOT / path) == digest for path, digest in lock['checkpoints'].items())


def _classify_test(out, ids):
    """Classifier test scores plus hard/soft condition vectors and routes for the segmentation test images."""
    eye = np.eye(num_targets(), dtype=np.float32)
    classifier = load_classifier(ROOT / 'full/classifier/best.pt', 'cuda')
    classification_ids = ids_for(2, classification=True)
    cls_all = classify(classifier, classification_ids)
    cls_seg = cls_all if np.array_equal(classification_ids, ids) else classify(classifier, ids)
    dump(out / 'classifier_test.json', cls_all)
    hard = {r['index']: eye[r['prediction']] for r in cls_seg['rows']}
    soft = {r['index']: r['probabilities'] for r in cls_seg['rows']}
    route = {r['index']: r['prediction'] for r in cls_seg['rows']}
    del classifier
    _release_gpu()
    return cls_all, hard, soft, route


def _all_foreground_baseline(ids):
    """Analytic all-foreground baseline on identical target GT; no extra trained model."""
    ds = Images(ids)
    fill = []
    for j in range(len(ds)):
        _a, y, v, p, idx = ds[j]
        gt = int(y[p].sum())
        pixels = int(v.sum())
        fill.append(dict(index=int(idx), plane=p, dice=2 * gt / (pixels + gt)))
    return dict(
        rows=fill,
        macro_dice=_macro(fill, range(num_targets())),
        note='Correct target channel paints entirevalidimage; no trainedmodel. Shows effect of large skullforeground.',
    )


def main():
    _verify_lock()
    resources()
    out = ROOT / 'final_evaluation'
    out.mkdir(exist_ok=False)
    ids = ids_for(2)
    labels = plane_ids(ids)
    eye = np.eye(num_targets(), dtype=np.float32)
    oracle = {int(i): eye[p] for i, p in zip(ids, labels)}
    true_route = {int(i): int(p) for i, p in zip(ids, labels)}
    cls_all, hard, soft, route = _classify_test(out, ids)
    scores = {}
    originals = {}
    timing = {}

    def record(name, result):
        dump(out / f'{name}.json', result)
        scores[name] = result['summary']

    for mode in SHARED_MODES:
        path = ROOT / f'full/{mode}/best.pt'
        m = load(path, 'cuda')
        conditions = (
            dict(oracle=oracle) if mode == 'plain' else dict(oracle=oracle, estimated_hard=hard, estimated_soft=soft)
        )
        for name, condition in conditions.items():
            record(f'{mode}_{name}', evaluate(m, ids, condition))
        record(f'{mode}_oracle_selected', selected_output(m, ids, oracle, true_route))
        record(f'{mode}_classifier_selected', selected_output(m, ids, hard, route))
        originals[mode] = common_original(m, ids)
        dump(out / f'{mode}_original.json', originals[mode])
        del m
        _release_gpu()
        timing[mode] = benchmark(path)
    for name, routing in [('oracle', true_route), ('estimated', route)]:
        record(f'experts_{name}', routed_experts(ids, routing))
    expert_rows = []
    for p in range(num_targets()):
        m = load(_expert_path(p), 'cuda')
        expert_rows.extend(common_original(m, ids[labels == p])['rows'])
        del m
        _release_gpu()
        timing[f'expert_{p}'] = benchmark(_expert_path(p))
    originals['experts'] = dict(
        rows=expert_rows,
        dice=float(np.mean([r['dice'] for r in expert_rows])),
        global_dice=float(np.mean([r['global_dice'] for r in expert_rows])),
        macro_dice=_macro(expert_rows, range(num_targets())),
    )
    timing['classifier'] = benchmark(ROOT / 'full/classifier/best.pt', classifier=True)
    timing['expert_bank_checkpoint_bytes'] = sum(_expert_path(p).stat().st_size for p in range(num_targets()))
    baseline = _all_foreground_baseline(ids)
    dump(out / 'all_foreground_baseline.json', baseline)
    dump(out / 'timings.json', timing)
    dump(out / 'original_common_gt.json', originals)
    dump(
        out / 'summary.json',
        dict(
            scores=scores,
            classifier_accuracy=cls_all['accuracy'],
            classifier_macro_accuracy=cls_all['macro_accuracy'],
            class_names=CLASSES,
            test_segmentation_images=len(ids),
            test_classification_images=len(ids_for(2, classification=True)),
            full_training_models=full_model_count(),
            seed=SEED,
            old_results_used=False,
            baseline_macro_dice=baseline['macro_dice'],
        ),
    )
    event('FINAL_EVALUATION_DONE')


if __name__ == '__main__':
    main()
