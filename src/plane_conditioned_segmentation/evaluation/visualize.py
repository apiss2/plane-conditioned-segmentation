"""Validation figures: target overlay, TP/FP/FN map and the strongest off-target channel."""

import numpy as np
import torch
import torch.nn.functional as F

from ..common import NUM_TARGETS
from ..data.batch import to_unit
from ..data.dataset import CLASSES, PROTOCOL, Images
from .metrics import EPS


def _overlay(ax, mask, cmap, alpha):
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=cmap, vmin=0, vmax=1, alpha=alpha)


@torch.inference_mode()
def visualize(m, ids, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ds = Images(ids)
    threshold = PROTOCOL.evaluation.probability_threshold
    fig, axes = plt.subplots(len(ids), 4, figsize=(16, 3 * len(ids)), squeeze=False)
    for j in range(len(ids)):
        a, y, v, p, idx = ds[j]
        c = F.one_hot(torch.tensor([p], device='cuda'), NUM_TARGETS).float()
        z = m(to_unit(a[None]), c)[0].sigmoid().cpu().numpy()
        pred = (z > threshold) & v.numpy().astype(bool)
        gt = y[p].numpy().astype(bool)
        target = pred[0] if m.target is not None else pred[p]
        dice = (2 * (target & gt).sum() + EPS) / (target.sum() + gt.sum() + EPS)
        row = axes[j]
        for ax in row:
            ax.imshow(a[0], cmap='gray')
            ax.axis('off')
        _overlay(row[0], gt, 'spring', 0.5)
        row[0].set_title(f'ID{idx} {CLASSES[p]} GT')
        _overlay(row[1], target, 'spring', 0.5)
        row[1].set_title(f'Prediction Dice{dice:.3f}')
        rgb = np.repeat(a.numpy()[0, :, :, None], 3, 2).astype(float) / 255
        for where, color in [(target & gt, [0, 1, 0]), (target & ~gt, [1, 0, 0]), (~target & gt, [0, 0, 1])]:
            rgb[where] = rgb[where] * 0.4 + np.array(color) * 0.6
        row[2].imshow(rgb)
        row[2].set_title('greenTP redFP blueFN')
        if m.target is None:
            # Show the non-target channel with the most predicted pixels.
            others = [ch for ch in range(NUM_TARGETS) if ch != p]
            worst = max(others, key=lambda ch: int(pred[ch].sum()))
            mask = pred[worst]
            _overlay(row[3], mask, 'autumn', 0.6)
            row[3].set_title(f'Offtask {CLASSES[worst]}: {mask.sum()}px')
        else:
            row[3].set_title('Binary1output; no hidden other channel')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
