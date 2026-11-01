"""Typed access to the research protocol and the machine runtime settings.

The protocol template ``configs/sevenplane_protocol.json`` is copied into each run as ``study_plan.json`` by
``prepare-data``; every result-affecting value is then read from that frozen per-run copy, never from the
template. Runtime settings in ``configs/runtime.json`` describe the host and do not change results.
Stdlib only: the dataset module reads this before numpy/torch load their native runtimes.
"""

import dataclasses
import typing
from pathlib import Path

from .common import SEED, read_json
from .paths import REPO_ROOT

PROTOCOL_TEMPLATE = REPO_ROOT / 'configs/sevenplane_protocol.json'
RUNTIME_CONFIG = REPO_ROOT / 'configs/runtime.json'
INTERPOLATIONS = ('nearest', 'bilinear')
# Sections added to the protocol after the first run was prepared. Plans that predate them
# (sevenplane_20261031) were trained and evaluated with exactly these values; newer plans state them.
LEGACY_V1_SECTIONS = {
    'preprocessing': dict(image_resample='bilinear'),
    'augmentation': dict(
        horizontal_flip_p=0.5,
        affine_scale=[0.85, 1.15],
        affine_translate=[-0.06, 0.06],
        affine_rotate_degrees=[-12, 12],
        affine_p=0.85,
        contrast_limit=0.25,
        contrast_p=0.7,
        gamma_limit=[75, 130],
        gamma_p=0.35,
        noise_multiplier=[0.85, 1.15],
        noise_p=0.3,
        blur_sigma=[0.1, 0.7],
        blur_p=0.15,
    ),
    'optimization': dict(weight_decay=1e-4, grad_clip_norm=5.0, cosine_eta_min=1e-6, validation_every=2),
    'classifier_rule': dict(effective_batch=16, microbatch=2, min_validation_macro_accuracy=0.75),
    'stall_guard': dict(start_epoch=16, window=6, max_dice=0.35, min_gain=0.03),
    'sufficiency_gate': dict(
        train_dice=0.70, global_dice=0.60, minimum_target=0.5, maximum_gain=0.02, late_fraction=0.7
    ),
    'evaluation': dict(probability_threshold=0.5, original_grid_interpolation='bilinear'),
}


@dataclasses.dataclass(frozen=True)
class Preprocessing:
    image_resample: str


@dataclasses.dataclass(frozen=True)
class Augmentation:
    horizontal_flip_p: float
    affine_scale: tuple[float, float]
    affine_translate: tuple[float, float]
    affine_rotate_degrees: tuple[float, float]
    affine_p: float
    contrast_limit: float
    contrast_p: float
    gamma_limit: tuple[float, float]
    gamma_p: float
    noise_multiplier: tuple[float, float]
    noise_p: float
    blur_sigma: tuple[float, float]
    blur_p: float


@dataclasses.dataclass(frozen=True)
class LearningRates:
    encoder: float
    decoder: float


@dataclasses.dataclass(frozen=True)
class SegmentationRule:
    min_epochs: int
    min_attempted_updates: int
    effective_batch: int
    microbatch: int


@dataclasses.dataclass(frozen=True)
class ClassifierRule:
    effective_batch: int
    microbatch: int
    min_validation_macro_accuracy: float


@dataclasses.dataclass(frozen=True)
class Optimization:
    weight_decay: float
    grad_clip_norm: float
    cosine_eta_min: float
    validation_every: int


@dataclasses.dataclass(frozen=True)
class StallGuard:
    """Abort when a target's train Dice stays below ``max_dice`` without gaining ``min_gain`` over ``window`` epochs."""

    start_epoch: int
    window: int
    max_dice: float
    min_gain: float


@dataclasses.dataclass(frozen=True)
class SufficiencyGate:
    """Terminal train-quality gate; validation gain is measured over epochs after ``late_fraction`` of training."""

    train_dice: float
    global_dice: float
    minimum_target: float
    maximum_gain: float
    late_fraction: float

    def thresholds(self):
        return dict(
            train_dice=self.train_dice,
            global_dice=self.global_dice,
            minimum_target=self.minimum_target,
            maximum_gain=self.maximum_gain,
        )


@dataclasses.dataclass(frozen=True)
class Evaluation:
    probability_threshold: float
    original_grid_interpolation: str


@dataclasses.dataclass(frozen=True)
class Protocol:
    seed: int
    inputs: tuple[tuple[int, int], ...]
    learning_rates: LearningRates
    full_segmentation_rule: SegmentationRule
    classifier_epochs: int
    classifier_rule: ClassifierRule
    preprocessing: Preprocessing
    augmentation: Augmentation
    optimization: Optimization
    stall_guard: StallGuard
    sufficiency_gate: SufficiencyGate
    evaluation: Evaluation

    @property
    def input_height(self):
        return self.inputs[0][1]

    @property
    def input_widths(self):
        return tuple(width for width, _ in self.inputs)


@dataclasses.dataclass(frozen=True)
class Runtime:
    torch_threads: int
    native_threads: int
    train_loader_workers: int
    eval_batch_size: int
    min_free_gpu_mib: int
    pipeline_process_markers: tuple[str, ...]


def _convert(kind, value, where):
    if dataclasses.is_dataclass(kind):
        return _build(kind, value, where, strict=True)
    origin = typing.get_origin(kind)
    if origin is tuple:
        args = typing.get_args(kind)
        if not isinstance(value, list):
            raise ValueError(f'{where}: list required')
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item, where) for item in value)
        if len(value) != len(args):
            raise ValueError(f'{where}: {len(args)} values required')
        return tuple(_convert(arg, item, where) for arg, item in zip(args, value))
    if kind is float and type(value) in (int, float):
        return float(value)
    if type(value) is not kind:
        raise ValueError(f'{where}: {kind.__name__} required')
    return value


def _build(cls, data, where, strict):
    """Construct ``cls`` from JSON; nested sections must name exactly their fields."""
    if not isinstance(data, dict):
        raise ValueError(f'{where}: object required')
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    missing = [n for n in names if n not in data]
    unknown = [k for k in data if k not in names]
    if missing or (strict and unknown):
        raise ValueError(f'{where}: missing {missing}, unknown {unknown}')
    return cls(**{n: _convert(hints[n], data[n], f'{where}.{n}') for n in names})


def protocol_from_dict(plan, where='protocol'):
    """Validated protocol; descriptive top-level keys are ignored, legacy plans get ``LEGACY_V1_SECTIONS``."""
    merged = {**LEGACY_V1_SECTIONS, **plan}
    protocol = _build(Protocol, merged, where, strict=False)
    if protocol.seed != SEED:
        raise ValueError(f'{where}: this pipeline is fixed to seed {SEED}')
    heights = {height for _, height in protocol.inputs}
    if not protocol.inputs or len(heights) != 1:
        raise ValueError(f'{where}.inputs: one shared input height required')
    if any(size % 32 for pair in protocol.inputs for size in pair):
        raise ValueError(f'{where}.inputs: sizes must be multiples of the 32-pixel encoder stride')
    for name in (protocol.preprocessing.image_resample, protocol.evaluation.original_grid_interpolation):
        if name not in INTERPOLATIONS:
            raise ValueError(f'{where}: interpolation must be one of {INTERPOLATIONS}')
    rule = protocol.full_segmentation_rule
    if (
        rule.effective_batch % rule.microbatch
        or protocol.classifier_rule.effective_batch % protocol.classifier_rule.microbatch
    ):
        raise ValueError(f'{where}: effective batch must be a multiple of the microbatch')
    return protocol


def load_protocol(path):
    """Validated protocol from a JSON file (a run's ``study_plan.json`` or the template)."""
    path = Path(path)
    return protocol_from_dict(read_json(path), path.name)


def run_protocol(root):
    """Frozen protocol of a run; before ``prepare-data`` has written one, the template a new run would receive."""
    plan = Path(root) / 'study_plan.json'
    return load_protocol(plan if plan.exists() else PROTOCOL_TEMPLATE)


def load_runtime(path=RUNTIME_CONFIG):
    return _build(Runtime, read_json(path), Path(path).name, strict=True)
