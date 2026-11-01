"""Seven-target segmentation loss: target BCE + Dice on valid pixels, plus non-target BCE for shared models."""

import torch
import torch.nn.functional as F

from ..common import NUM_TARGETS
from ..data.batch import validate_batch

# Recorded in every checkpoint and training configuration; bump ``version`` when the loss changes.
LOSS = dict(
    version='seven_target_balanced_v1',
    target='Per-image target-channel valid-pixel BCE + target Dice',
    other='Shared models additionally mean BCE across6nontarget channels, weight1; expert has no nontarget outputs',
    padding='ignored',
    missing='Source missing masks excluded before loader',
)


def loss_parts(z, y, v, p):
    """Return (target BCE+Dice, mean non-target BCE) summed over the batch, both restricted to valid pixels."""
    validate_batch(z, y, v, p)
    z, y, v, p = z.float(), y.float(), v.float(), p.long()
    batch = torch.arange(len(z), device=z.device)
    target = z[:, 0:1] if z.shape[1] == 1 else z[batch, p][:, None]
    gt = y[batch, p][:, None]
    pixels = v.sum((1, 2, 3)).clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(target, gt, reduction='none') * v).sum((1, 2, 3)) / pixels
    prob = target.sigmoid() * v
    gt = gt * v
    dice = 1 - (2 * (prob * gt).sum((1, 2, 3)) + 1) / ((prob + gt).sum((1, 2, 3)) + 1)
    extra = torch.zeros_like(bce)
    if z.shape[1] > 1:
        negative = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z), reduction='none') * v
        negative[batch, p] = 0
        extra = negative.sum((1, 2, 3)) / (pixels * (NUM_TARGETS - 1))
    return (bce + dice).sum(), extra.sum()


def loss(z, y, v, p, total):
    target, other = loss_parts(z, y, v, p)
    return (target + other) / total
