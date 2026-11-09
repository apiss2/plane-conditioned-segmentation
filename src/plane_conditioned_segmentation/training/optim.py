"""AdamW with a slow encoder, and AMP updates accumulated over microbatches."""

import torch
import torch.nn.functional as F

from ..common import num_targets
from ..data.batch import to_unit
from .losses import loss


def optimizer(m, *, encoder_lr, decoder_lr, weight_decay):
    encoder = list(m.network.encoder.parameters())
    rest = [p for name, p in m.named_parameters() if not name.startswith('network.encoder.')]
    return torch.optim.AdamW(
        [dict(params=encoder, lr=encoder_lr), dict(params=rest, lr=decoder_lr)], weight_decay=weight_decay
    )


def protocol_optimizer(m, protocol):
    return optimizer(
        m,
        encoder_lr=protocol.learning_rates.encoder,
        decoder_lr=protocol.learning_rates.decoder,
        weight_decay=protocol.optimization.weight_decay,
    )


def amp_step(m, opt, scaler, grad_clip_norm):
    """Clip, step and update the scaler; returns (grad norm, whether the step was applied)."""
    scaler.unscale_(opt)
    grad = torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip_norm)
    old = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    # GradScaler lowers its scale exactly when it skipped a non-finite step.
    return grad, scaler.get_scale() >= old


def update(m, opt, scaler, a, y, v, p, *, microbatch, grad_clip_norm):
    """One effective-batch segmentation step accumulated over microbatches."""
    opt.zero_grad(set_to_none=True)
    total = 0
    for j in range(0, len(a), microbatch):
        part = slice(j, j + microbatch)
        pp = p[part].cuda()
        c = F.one_hot(pp, num_targets()).float()
        with torch.autocast('cuda', dtype=torch.float16):
            z = m(to_unit(a[part]), c)
        ell = loss(z, y[part].cuda(), v[part].cuda(), pp, len(a))
        assert torch.isfinite(ell)
        scaler.scale(ell).backward()
        total += float(ell.detach())
    grad, applied = amp_step(m, opt, scaler, grad_clip_norm)
    return dict(loss=total, applied=applied, grad_norm=float(grad) if torch.isfinite(grad) else None)
