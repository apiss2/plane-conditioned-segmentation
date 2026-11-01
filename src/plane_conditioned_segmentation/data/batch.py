"""Contract shared by losses and metrics for one batch of segmentation tensors."""

import torch

from ..common import NUM_TARGETS

INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def validate_batch(z, y, v, p):
    """Segmentation tensors are finite, binary-supported and have explicit seven-view labels."""
    if not all(isinstance(t, torch.Tensor) for t in (z, y, v, p)):
        raise ValueError('Tensor inputs required')
    if z.ndim != 4 or z.shape[1] not in (1, NUM_TARGETS) or z.numel() == 0:
        raise ValueError('Nonempty Bx(1|7)xHxW logits required')
    batch = z.shape[0]
    spatial = z.shape[2:]
    if y.shape != (batch, NUM_TARGETS, *spatial) or v.shape != (batch, 1, *spatial) or p.shape != (batch,):
        raise ValueError('Mask, valid-support or plane shape mismatch')
    if p.dtype not in INTEGER_DTYPES or not ((p >= 0) & (p < NUM_TARGETS)).all():
        raise ValueError('Integer plane IDs 0 through 6 required')
    if not all(torch.isfinite(t).all() for t in (z, y, v)):
        raise ValueError('Finite logits, masks and support required')
    if not all(((t == 0) | (t == 1)).all() for t in (y, v)):
        raise ValueError('Binary ground truth and valid support required')
    if not (v.flatten(1).sum(1) > 0).all():
        raise ValueError('Every image must have positive valid support')


def to_unit(a):
    """uint8 image batch on the GPU as float in [0, 1]."""
    return a.cuda().float() / 255
