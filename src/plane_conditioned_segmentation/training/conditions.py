"""Condition-only augmentation, deterministic by image and epoch; GT and global RNG stay unchanged."""

import numpy as np
import torch.nn.functional as F

from ..common import SEED


def training_conditions(labels, indices, epoch, policy, targets):
    if labels.ndim != 1 or indices.shape != labels.shape or epoch < 0 or targets < 2:
        raise ValueError('Aligned labels/indices and nonnegative epoch required')
    condition = F.one_hot(labels, targets).float()
    if policy.mode == 'onehot':
        return condition
    if policy.mode == 'uniform':
        return condition * (1 - policy.epsilon) + policy.epsilon / targets
    if policy.mode != 'random_pair':
        raise ValueError('Unknown condition augmentation')
    for j, (label, index) in enumerate(zip(labels.tolist(), indices.tolist())):
        rng = np.random.default_rng((SEED + index * 1009 + epoch * 1000003) % 2**32)
        if rng.random() < policy.clean_probability:
            continue
        epsilon = float(rng.uniform(0, policy.epsilon))
        other = (label + int(rng.integers(1, targets))) % targets
        condition[j, label] = 1 - epsilon
        condition[j, other] = epsilon
    return condition
