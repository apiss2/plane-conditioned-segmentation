"""Segmentation metrics on valid pixels and their per-image / per-target aggregation."""

import numpy as np
import torch
import torch.nn.functional as F

from ..common import NUM_TARGETS
from ..data.batch import to_unit, validate_batch
from ..data.dataset import PROTOCOL, loader

# Smoothing term of every Dice in this module; part of the metric definition, not a tunable.
EPS = 1e-6
RATE_KEYS = ['dice', 'global_dice', 'offtarget_fp_rate']
COUNT_KEYS = ['target_fp', 'target_fn', 'target_predicted_pixels', 'target_gt_pixels']
NONNEGATIVE_KEYS = ['offtarget_fp_pixels', *COUNT_KEYS]
METRIC_KEYS = ['dice', 'global_dice', 'offtarget_fp_pixels', 'offtarget_fp_rate', *COUNT_KEYS]


@torch.no_grad()
def metrics(z, y, v, p, target=None, *, threshold):
    """Per-image target Dice, all-channel global Dice and off-target false positives at probability >threshold."""
    validate_batch(z, y, v, p)
    p = p.long()
    if z.shape[1] == 1 and (type(target) is not int or not 0 <= target < NUM_TARGETS):
        raise ValueError('Binary expert requires target ID 0 through 6')
    pred = (z.float().sigmoid() > threshold) & v.bool()
    gt = y.bool() & v.bool()
    batch = torch.arange(len(z), device=z.device)
    if z.shape[1] == 1:
        # A binary expert predicts only its own target channel; all others stay empty.
        full = torch.zeros_like(gt)
        full[:, target : target + 1] = pred
        pred = full
    tp = (pred & gt).sum((2, 3))
    pc = pred.sum((2, 3))
    tc = gt.sum((2, 3))
    channel_dice = (2 * tp + EPS) / (pc + tc + EPS)
    global_dice = (2 * tp.sum(1) + EPS) / (pc.sum(1) + tc.sum(1) + EPS)
    offtarget = torch.ones_like(pc, dtype=torch.bool)
    offtarget[batch, p] = False
    offtarget_fp = (pc * offtarget).sum(1)
    offtarget_pixels = v.sum((1, 2, 3)) * (NUM_TARGETS - 1)
    target_tp, target_pc, target_tc = tp[batch, p], pc[batch, p], tc[batch, p]

    def numpy(t):
        return t.cpu().numpy()

    return dict(
        dice=numpy(channel_dice[batch, p]),
        global_dice=numpy(global_dice),
        offtarget_fp_pixels=numpy(offtarget_fp),
        offtarget_fp_rate=numpy(offtarget_fp / offtarget_pixels.clamp_min(1)),
        target_fp=numpy(target_pc - target_tp),
        target_fn=numpy(target_tc - target_tp),
        target_predicted_pixels=numpy(target_pc),
        target_gt_pixels=numpy(target_tc),
        tp=numpy(tp.sum(0)),
        pc=numpy(pc.sum(0)),
        tc=numpy(tc.sum(0)),
    )


def metric_row(r, j, index, plane):
    """Row ``j`` of a batched ``metrics`` result as a JSON-ready per-image record."""
    return dict(
        index=int(index),
        plane=int(plane),
        dice=float(r['dice'][j]),
        global_dice=float(r['global_dice'][j]),
        offtarget_fp_pixels=int(r['offtarget_fp_pixels'][j]),
        offtarget_fp_rate=float(r['offtarget_fp_rate'][j]),
        **{key: int(r[key][j]) for key in COUNT_KEYS},
    )


@torch.inference_mode()
def evaluate(m, ids, conditions=None):
    """Score ``ids`` with oracle one-hot conditions, or with ``conditions[index]`` vectors when given."""
    m.eval()
    dl, _ = loader(ids)
    threshold = PROTOCOL.evaluation.probability_threshold
    rows = []
    tp, pc, tc = np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)
    for a, y, v, p, idx in dl:
        if conditions is None:
            c = F.one_hot(p.cuda(), NUM_TARGETS).float()
        else:
            c = torch.tensor(np.stack([conditions[int(i)] for i in idx]), device='cuda').float()
        z = m(to_unit(a), c)
        r = metrics(z, y.cuda(), v.cuda(), p.cuda(), m.target, threshold=threshold)
        rows.extend(metric_row(r, j, i, p[j]) for j, i in enumerate(idx))
        tp += r['tp']
        pc += r['pc']
        tc += r['tc']
    rows.sort(key=lambda r: r['index'])
    return dict(
        rows=rows,
        summary=summarize(rows),
        tp=tp.tolist(),
        pc=pc.tolist(),
        tc=tc.tolist(),
        class_micro_dice=((2 * tp + EPS) / (pc + tc + EPS)).tolist(),
    )


def _validate_rows(rows):
    if not rows or any(not np.isfinite(row[k]) for row in rows for k in METRIC_KEYS):
        raise ValueError('Nonempty finite metric rows required')
    if any(not 0 <= row[k] <= 1 for row in rows for k in RATE_KEYS) or any(
        row[k] < 0 for row in rows for k in NONNEGATIVE_KEYS
    ):
        raise ValueError('Impossible metric range')


def summarize(rows):
    """Image means plus target-macro Dice; ``selection`` is the validation checkpoint criterion."""
    _validate_rows(rows)

    def mean(subset):
        return {key: float(np.mean([r[key] for r in subset])) for key in METRIC_KEYS}

    result = mean(rows)
    planes = sorted(set(row['plane'] for row in rows))
    result['per_target'] = {str(i): mean([row for row in rows if row['plane'] == i]) for i in planes}
    result['macro_dice'] = float(np.mean([v['dice'] for v in result['per_target'].values()]))
    result['macro_global_dice'] = float(np.mean([v['global_dice'] for v in result['per_target'].values()]))
    result['selection'] = (result['macro_dice'] + result['macro_global_dice']) / 2
    return result
