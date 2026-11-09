import numpy as np
import pytest
import torch
import torch.nn.functional as F

from plane_conditioned_segmentation.config import ConditionTraining, conditioning_from_plan
from plane_conditioned_segmentation.training.conditions import training_conditions
from plane_conditioned_segmentation.training.optim import update


@pytest.mark.parametrize('mode,epsilon,clean', [('onehot', 0.0, 1.0), ('uniform', 0.1, 0.0), ('random_pair', 0.4, 0.5)])
def test_conditions_replay_by_image_epoch_and_leave_targets_and_rng_untouched(mode, epsilon, clean):
    policy = ConditionTraining(mode, epsilon, clean)
    p = torch.arange(60) % 6
    indices = torch.arange(100, 160)
    before = p.clone()
    torch_rng = torch.get_rng_state()
    numpy_rng = np.random.get_state()
    c = training_conditions(p, indices, 5, policy, 6)
    torch.testing.assert_close(c.sum(1), torch.ones(60))
    assert (c >= 0).all() and torch.equal(p, before)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    # Rebatching/reordering and a partial-epoch resume preserve each sample's condition.
    order = torch.arange(59, -1, -1)
    torch.testing.assert_close(training_conditions(p[order], indices[order], 5, policy, 6), c[order], rtol=0, atol=0)
    if mode == 'onehot':
        assert torch.equal(c, F.one_hot(p, 6).float())
    elif mode == 'uniform':
        torch.testing.assert_close(c, F.one_hot(p, 6).float() * 0.9 + 0.1 / 6)
    else:
        nonzero = (c > 0).sum(1)
        assert (nonzero == 1).any() and (nonzero == 2).any()
        assert not torch.equal(c, training_conditions(p, indices, 6, policy, 6))
        assert (c[torch.arange(60), p] >= 0.6).all()


@pytest.mark.parametrize(
    'change', [dict(mode='typo'), dict(epsilon=float('nan')), dict(epsilon=-0.1), dict(clean_probability=2)]
)
def test_bad_condition_policy_fails_before_training(change):
    value = dict(mode='random_pair', epsilon=0.4, clean_probability=0.5)
    with pytest.raises(ValueError):
        conditioning_from_plan(dict(schedulefree={}, condition_training={**value, **change}))


def test_invalid_condition_refused_before_any_optimizer_mutation():
    with pytest.raises(ValueError, match='condition vector'):
        update(
            None,
            None,
            None,
            torch.zeros(2, 1, 4, 4),
            None,
            None,
            None,
            microbatch=2,
            grad_clip_norm=5,
            conditions=torch.zeros(2, 7),
        )
