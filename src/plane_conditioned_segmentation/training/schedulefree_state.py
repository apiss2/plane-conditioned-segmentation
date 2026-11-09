"""RAdamScheduleFree construction, mode transitions and consistent eval-weight checkpoint/resume."""

import importlib.metadata
from pathlib import Path

import torch
from schedulefree import RAdamScheduleFree


def make_optimizer(groups, protocol, policy):
    if importlib.metadata.version('schedulefree') != policy.package_version:
        raise ValueError('Installed schedulefree differs from frozen policy')
    return RAdamScheduleFree(
        groups,
        betas=policy.betas,
        eps=policy.eps,
        weight_decay=protocol.optimization.weight_decay,
        r=policy.r,
        weight_lr_power=policy.weight_lr_power,
        foreach=policy.foreach,
        silent_sgd_phase=policy.silent_sgd_phase,
    )


def set_mode(model, optimizer, training):
    # Encoder BN stays frozen by Model/Classifier.train(); decoder uses GroupNorm. No BN calibration.
    optimizer.train() if training else optimizer.eval()
    model.train(training)


def save_state(path, model, optimizer, **metadata):
    """Both inference-only best and resumable states store x, never the training y buffer."""
    set_mode(model, optimizer, False)
    payload = dict(state=model.state_dict(), weight_sequence='schedulefree_eval_x', **metadata)
    if metadata.get('resumable'):
        payload.update(
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_state(path, model, optimizer, *, protocol_sha256, label):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if (
        not ck.get('resumable')
        or ck.get('weight_sequence') != 'schedulefree_eval_x'
        or ck.get('optimizer_name') != 'RAdamScheduleFree'
        or ck.get('protocol_sha256') != protocol_sha256
        or ck.get('label') != label
        or any(g['train_mode'] for g in ck['optimizer']['param_groups'])
    ):
        raise ValueError('Checkpoint is not a compatible eval-weight Schedule-Free resume state')
    model.load_state_dict(ck['state'])
    optimizer.load_state_dict(ck['optimizer'])
    # PyTorch moves floating optimizer buffers to the device of each model parameter.
    torch.set_rng_state(ck['torch_rng'])
    if ck.get('cuda_rng'):
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    model.eval()
    return ck


def learning_rates(optimizer):
    return [
        dict(
            lr=float(g['lr']),
            rectified_lr=float(g.get('scheduled_lr', 0)),
            optimizer_steps=int(g.get('k', 0)),
            train_mode=bool(g['train_mode']),
        )
        for g in optimizer.param_groups
    ]
