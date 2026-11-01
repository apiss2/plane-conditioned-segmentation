"""Plane-classifier accuracy, per-class-recall macro accuracy and confusion."""

import numpy as np
import torch

from ..common import NUM_TARGETS
from ..data.batch import to_unit
from ..data.dataset import loader


@torch.inference_mode()
def evaluate(m, ids):
    """Per-image probabilities, accuracy, macro accuracy and confusion (target x prediction)."""
    m.eval()
    dl, _ = loader(ids)
    rows = []
    confusion = np.zeros((NUM_TARGETS, NUM_TARGETS), np.int64)
    for a, _y, _v, p, idx in dl:
        q = m(to_unit(a)).softmax(1).cpu().numpy()
        for j, i in enumerate(idx):
            gt = int(p[j])
            pred = int(q[j].argmax())
            confusion[gt, pred] += 1
            rows.append(dict(index=int(i), target=gt, prediction=pred, probabilities=q[j].tolist()))
    rows.sort(key=lambda r: r['index'])
    accuracy = float(sum(r['prediction'] == r['target'] for r in rows) / len(rows))
    macro = float(np.mean(np.diag(confusion) / confusion.sum(1).clip(1)))
    return dict(rows=rows, accuracy=accuracy, macro_accuracy=macro, confusion=confusion.tolist())
