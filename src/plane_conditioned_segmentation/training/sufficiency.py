"""Training sufficiency uses terminal training, independently of validation checkpoint selection."""

import math
import numbers

from ..config import LEGACY_V1_SECTIONS, SufficiencyGate


def _validate(history):
    """Return the checked rows (those with train and validation metrics) of a well-formed history."""
    if not history or any(type(r.get('epoch')) is not int or r['epoch'] <= 0 for r in history):
        raise ValueError('Positive integer epochs and nonempty history required')
    if any(b['epoch'] <= a['epoch'] for a, b in zip(history, history[1:])):
        raise ValueError('Epochs must be strictly increasing')
    checked = [row for row in history if 'validation' in row and 'train' in row]
    if not checked or checked[-1] is not history[-1]:
        raise ValueError('Terminal train and validation metrics required')
    target_keys = set(checked[0]['train'].get('per_target', {}))
    if not target_keys or any(set(r['train'].get('per_target', {})) != target_keys for r in checked):
        raise ValueError('Consistent nonempty target metrics required')
    for row in checked:
        values = [
            row['train']['macro_dice'],
            row['train']['macro_global_dice'],
            row['validation']['selection'],
            *[r['dice'] for r in row['train']['per_target'].values()],
        ]
        if any(
            isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        ):
            raise ValueError('Finite metric values in [0,1] required')
    return checked


def assess(history, gate):
    checked = _validate(history)
    terminal = checked[-1]
    late_start = int(history[-1]['epoch'] * gate.late_fraction)
    past = [row['validation']['selection'] for row in checked if row['epoch'] <= late_start]
    gain = terminal['validation']['selection'] - max(past, default=0)
    train = terminal['train']
    minimum = min(row['dice'] for row in train['per_target'].values())
    passed = (
        train['macro_dice'] >= gate.train_dice
        and train['macro_global_dice'] >= gate.global_dice
        and minimum >= gate.minimum_target
        and gain <= gate.maximum_gain
    )
    return dict(
        pass_gate=passed,
        terminal_epoch=terminal['epoch'],
        terminal_train=train,
        terminal_validation=terminal['validation'],
        last20percent_selection_gain=gain,
        minimum_target_train_dice=minimum,
        thresholds=gate.thresholds(),
    )


def self_check():
    """Synthetic threshold cases against the recorded v1 gate only; no study observations are embedded."""
    gate = SufficiencyGate(**LEGACY_V1_SECTIONS['sufficiency_gate'])

    def row(epoch, train, val):
        return dict(
            epoch=epoch,
            train=dict(macro_dice=train, macro_global_dice=train, per_target={'6': dict(dice=train)}),
            validation=dict(selection=val),
        )

    assert assess([row(10, 0.8, 0.4), row(20, 0.9, 0.4)], gate)['pass_gate']
    assert not assess([row(10, 0.8, 0.4), row(20, 0.69, 0.4)], gate)['pass_gate']
    assert not assess([row(10, 0.80, 0.40), row(20, 0.80, 0.45)], gate)['pass_gate']
    assert assess([row(10, 0.80, 0.45), row(20, 0.80, 0.44)], gate)['pass_gate']
    return True
