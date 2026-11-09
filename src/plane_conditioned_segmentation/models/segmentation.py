"""MobileNetV2 U-Net with optional plane conditioning (FiLM or external-ID kernel mixture) and binary experts.

The architecture constants here define the checkpoint format: saved weights record only mode and target.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from ..common import SEED, num_targets, read_json
from ..paths import experiment_root, initialization_root

ROOT = experiment_root()
INITIALIZATION = initialization_root(ROOT)
DECODER = (64, 48, 32, 24, 16)
MODES = ('plain', 'film', 'condconv', 'expert')
ENCODER_FILE = 'encoder_mobilenet_v2_imagenet_gray.pt'
METADATA_FILE = 'encoder_weights_metadata.json'
INITIALIZATION_FILE = f'fresh_common_initialization_seed{SEED}.pt'


def norm(module):
    """Replace every BatchNorm2d below ``module`` with a GroupNorm carrying its affine parameters."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(8, child.num_features)
            while child.num_features % groups:
                groups -= 1
            gn = nn.GroupNorm(groups, child.num_features, eps=child.eps)
            gn.weight.data.copy_(child.weight)
            gn.bias.data.copy_(child.bias)
            setattr(module, name, gn)
        else:
            norm(child)


def build_unet(classes):
    """Grayscale MobileNetV2 U-Net with a GroupNorm decoder."""
    network = smp.Unet(
        encoder_name='mobilenet_v2',
        encoder_depth=5,
        encoder_weights=None,
        decoder_channels=DECODER,
        decoder_use_norm='batchnorm',
        decoder_interpolation='nearest',
        in_channels=1,
        classes=classes,
        activation=None,
    )
    norm(network.decoder)
    return network


def freeze_batchnorm(module):
    """Keep pretrained encoder BatchNorm statistics fixed while the rest trains."""
    for layer in module.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


class Condition(nn.Module):
    def __init__(self, ch, mode):
        super().__init__()
        self.mode = mode
        if mode == 'film':
            self.layer = nn.Linear(num_targets(), ch * 2, bias=False)
            self.layer.weight.data.zero_()

    def forward(self, x, c):
        if self.mode == 'film':
            gamma, beta = self.layer(c).chunk(2, 1)
            return (x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]).relu()
        return x


class ExternalIDKernelMixture(nn.Module):
    """DynamicConv-family softmax kernel bank; router uses external view, not image."""

    def __init__(self, conv, experts=4):
        super().__init__()
        assert conv.groups == 1 and conv.bias is None
        self.kernels = nn.Parameter(conv.weight.detach()[None].repeat(experts, 1, 1, 1, 1))
        self.router = nn.Linear(num_targets(), experts, bias=False)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.experts = experts

    def weights(self, c):
        alpha = self.router(c).softmax(1)
        # Algebraically sum(alpha_k*W_k). Differences give exact common initial W.
        return self.kernels[0][None] + torch.einsum('bk,kocuv->bocuv', alpha[:, 1:], self.kernels[1:] - self.kernels[0])

    def forward(self, x, c):
        b, channels, h, w = x.shape
        kernel = self.weights(c)
        outputs = kernel.shape[1]
        # Grouped convolution applies each sample's own kernel in one call.
        z = F.conv2d(
            x.reshape(1, b * channels, h, w),
            kernel.reshape(b * outputs, channels, *kernel.shape[-2:]),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b,
        )
        return z.reshape(b, outputs, *z.shape[-2:])


class Model(nn.Module):
    def __init__(self, mode='plain', target=None, initialize=True):
        super().__init__()
        assert mode in MODES
        assert (mode == 'expert') == (target is not None)
        self.mode = mode
        self.target = target
        self.outputs = 1 if target is not None else num_targets()
        self.network = build_unet(self.outputs)
        if initialize:
            self._load_initialization(mode, target)
        if mode == 'condconv':
            conv2 = self.network.decoder.blocks[0].conv2
            conv2[0] = ExternalIDKernelMixture(conv2[0])
        self.condition = nn.ModuleList([Condition(ch, 'film' if mode == 'film' else 'plain') for ch in DECODER])
        metadata = read_json(INITIALIZATION / METADATA_FILE)
        self.register_buffer('mean', torch.tensor(metadata['gray_mean']).reshape(1, 1, 1, 1))
        self.register_buffer('std', torch.tensor(metadata['gray_std']).reshape(1, 1, 1, 1))
        self.train()

    def _load_initialization(self, mode, target):
        path = INITIALIZATION / INITIALIZATION_FILE
        if path.exists():
            state = torch.load(path, map_location='cpu', weights_only=True)
            if target is not None:
                # Binary experts keep only their target's head channel.
                for key in ['segmentation_head.0.weight', 'segmentation_head.0.bias']:
                    state[key] = state[key][target : target + 1]
            self.network.load_state_dict(state)
        else:
            assert mode == 'plain' and target is None, 'Create common fresh initialization first'
            encoder = torch.load(INITIALIZATION / ENCODER_FILE, map_location='cpu', weights_only=True)
            self.network.encoder.load_state_dict(encoder)

    def train(self, mode=True):
        super().train(mode)
        freeze_batchnorm(self.network.encoder)
        return self

    def forward(self, a, c):
        assert a.shape[1] == 1 and a.shape[-1] % 32 == 0 and a.shape[-2] % 32 == 0
        features = self.network.encoder((a - self.mean) / self.std)
        sizes = [z.shape[-2:] for z in features][::-1]
        features = features[1:][::-1]
        x = self.network.decoder.center(features[0])
        for i, block in enumerate(self.network.decoder.blocks):
            skip = features[i + 1] if i + 1 < len(features) else None
            if self.mode == 'condconv' and i == 0:
                x = self._conditioned_block(block, x, sizes[i + 1], skip, c)
            else:
                x = block(x, *sizes[i + 1], skip_connection=skip)
            x = self.condition[i](x, c)
        return self.network.segmentation_head(x)

    @staticmethod
    def _conditioned_block(block, x, size, skip, c):
        """smp DecoderBlock.forward with the external-ID kernel mixture as conv2's convolution."""
        x = F.interpolate(x, size=size, mode=block.interpolation_mode)
        if skip is not None:
            x = block.attention1(torch.cat([x, skip], 1))
        x = block.conv1(x)
        x = block.conv2[0](x, c)
        for layer in list(block.conv2.children())[1:]:
            x = layer(x)
        return block.attention2(x)


def load(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('target_count', num_targets()) != num_targets():
        raise ValueError('Checkpoint target count differs from active run')
    m = Model(ck['mode'], ck['target'], initialize=False)
    m.load_state_dict(ck['state'])
    return m.to(device).eval()
