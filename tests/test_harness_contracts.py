import random
import sys

import numpy as np
import pytest
import torch

from plane_conditioned_segmentation.paths import REPO_ROOT


def test_network_guard_without_making_a_network_call():
    with pytest.raises(PermissionError):
        sys.audit('socket.connect', None, ('example.invalid', 443))


def test_research_write_guard_without_opening_or_creating_a_file():
    with pytest.raises(PermissionError):
        sys.audit('open', str(REPO_ROOT / 'runs/must_not_be_created'), 'w', 0)


def test_each_test_has_fixed_independent_random_streams():
    assert random.random() == random.Random(19).random()
    np.testing.assert_array_equal(np.random.random(3), np.random.RandomState(19).random_sample(3))
    generator = torch.Generator().manual_seed(19)
    torch.testing.assert_close(torch.rand(3), torch.rand(3, generator=generator), rtol=0, atol=0)
