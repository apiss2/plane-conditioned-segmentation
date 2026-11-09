"""Explicit Schedule-Free policy and validation-only early stopping. No test data access."""

import dataclasses
import math

from ..common import read_json, sha256_file
from ..config import _build


@dataclasses.dataclass(frozen=True)
class ScheduleFreePolicy:
    optimizer: str
    package_version: str
    betas: tuple[float, float]
    eps: float
    r: float
    weight_lr_power: float
    foreach: bool
    silent_sgd_phase: bool
    max_epochs: int
    min_epochs: int
    min_applied_updates: int
    patience: int
    min_delta: float

    def __post_init__(self):
        if self.optimizer != 'RAdamScheduleFree' or self.package_version != '1.4.1':
            raise ValueError('Only the audited schedulefree 1.4.1 RAdam API is supported')
        if not (0 < self.betas[0] < 1 and 0 < self.betas[1] < 1):
            raise ValueError('Finite betas in (0,1) required')
        if not all(math.isfinite(v) for v in [*self.betas, self.eps, self.r, self.weight_lr_power, self.min_delta]):
            raise ValueError('Finite policy values required')
        if self.eps <= 0 or self.r < 0 or self.weight_lr_power < 0 or self.min_delta < 0:
            raise ValueError('Invalid optimizer/early stopping parameters')
        if not (0 < self.min_epochs <= self.max_epochs and self.patience > 0 and self.min_applied_updates >= 0):
            raise ValueError('Invalid epoch/update/patience limits')


def stopping_override(path, label):
    override_path = path.parent / 'stopping_policy_override.json'
    if label is None or not override_path.exists():
        return None
    value = read_json(override_path)
    if label not in value.get('models', []):
        return None
    if value.get('base_protocol_sha256') != sha256_file(path) or value.get('changes') != {
        'min_epochs': 1,
        'min_applied_updates': 0,
    }:
        raise ValueError('Only explicit, protocol-bound removal of minimum stopping constraints is allowed')
    return value


def load_policy(path, label=None):
    value = read_json(path).get('schedulefree')
    policy = None if value is None else _build(ScheduleFreePolicy, value, 'schedulefree', strict=True)
    override = stopping_override(path, label)
    return dataclasses.replace(policy, **override['changes']) if override is not None else policy


class EarlyStopping:
    """Strict improvement > reference + min_delta; ties count; NaN aborts without advancing state.

    Patience counts validation checks, one per epoch for this protocol. The counter starts after the
    first finite observation and is not reset at minimum-budget eligibility. A hard epoch cap always wins.
    Selection of the exact best checkpoint is independent of the significant-improvement reference.
    """

    def __init__(self, policy):
        self.policy = policy
        self.reference = None
        self.bad_checks = 0
        self.last_epoch = 0
        self.last_updates = 0
        self.best = None
        self.best_epoch = 0

    def observe(self, score, epoch, applied):
        if (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError('Finite validation score in [0,1] required; no state advanced')
        if type(epoch) is not int or not self.last_epoch < epoch <= self.policy.max_epochs:
            raise ValueError('Epoch must increase within the fixed cap')
        if type(applied) is not int or applied < self.last_updates:
            raise ValueError('Applied updates must be nondecreasing integers')
        selected = self.best is None or score > self.best
        if selected:
            self.best, self.best_epoch = score, epoch
        if self.reference is None or score > self.reference + self.policy.min_delta:
            self.reference, self.bad_checks = score, 0
        else:
            self.bad_checks += 1
        self.last_epoch, self.last_updates = epoch, applied
        eligible = epoch >= self.policy.min_epochs and applied >= self.policy.min_applied_updates
        reason = self.stop_reason()
        return dict(
            selected=selected, stop=reason is not None, reason=reason, eligible=eligible, bad_checks=self.bad_checks
        )

    def stop_reason(self):
        if self.last_epoch >= self.policy.max_epochs:
            return 'max_epochs'
        if (
            self.last_epoch >= self.policy.min_epochs
            and self.last_updates >= self.policy.min_applied_updates
            and self.bad_checks >= self.policy.patience
        ):
            return 'validation_patience'
        return None

    def state_dict(self):
        return dict(
            policy=dataclasses.asdict(self.policy),
            reference=self.reference,
            bad_checks=self.bad_checks,
            last_epoch=self.last_epoch,
            last_updates=self.last_updates,
            best=self.best,
            best_epoch=self.best_epoch,
        )

    def load_state_dict(self, state, *, allow_minimum_removal=False):
        expected = dataclasses.asdict(self.policy)
        adjusted = {**state['policy'], 'min_epochs': 1, 'min_applied_updates': 0}
        if state['policy'] != expected and not (allow_minimum_removal and adjusted == expected):
            raise ValueError('Resume early-stopping policy mismatch')
        for key in ['reference', 'bad_checks', 'last_epoch', 'last_updates', 'best', 'best_epoch']:
            setattr(self, key, state[key])
