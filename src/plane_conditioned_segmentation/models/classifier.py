"""Seven-way plane classifier sharing the segmentation encoder."""

import torch
from torch import nn

from ..common import num_targets
from .segmentation import Model, freeze_batchnorm


class Classifier(nn.Module):
    """Plane classifier on the shared segmentation encoder with global average pooling."""

    def __init__(self, initialize=True):
        super().__init__()
        base = Model('plain', initialize=initialize)
        self.encoder = base.network.encoder
        self.register_buffer('mean', base.mean)
        self.register_buffer('std', base.std)
        self.head = nn.Linear(self.encoder.out_channels[-1], num_targets())

    def train(self, mode=True):
        super().train(mode)
        freeze_batchnorm(self.encoder)
        return self

    def forward(self, x):
        return self.head(self.encoder((x - self.mean) / self.std)[-1].mean((2, 3)))


def load_classifier(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = Classifier(initialize=False)
    m.load_state_dict(ck['state'])
    return m.to(device).eval()
