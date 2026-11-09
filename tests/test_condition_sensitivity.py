import numpy as np
import pytest

from plane_conditioned_segmentation.evaluation.sensitivity import condition_cases, failure_summary


def test_sweep_covers_every_wrong_class_and_preserves_simplex():
    labels = np.arange(6)
    q = np.eye(6)[labels] * 0.8 + 0.2 / 6
    cases = dict(condition_cases(labels, q))
    assert len(cases) == 28
    np.testing.assert_array_equal(cases['oracle'], np.eye(6))
    np.testing.assert_allclose(cases['estimated_soft'], q)
    for p in range(6):
        assert {cases[f'pair_{o}_1.0'][p].argmax() for o in range(1, 6)} == set(range(6)) - {p}
    for c in cases.values():
        assert (c >= 0).all()
        np.testing.assert_allclose(c.sum(1), 1, atol=1e-6)
    with pytest.raises(ValueError):
        list(condition_cases(labels, q * 0.8))


def test_failure_summary_matches_ids_not_input_order():
    reference = [
        dict(index=1, dice=0.9, target_predicted_pixels=10),
        dict(index=2, dice=0.8, target_predicted_pixels=20),
    ]
    rows = [dict(index=2, dice=0.0, target_predicted_pixels=0), dict(index=1, dice=0.85, target_predicted_pixels=8)]
    r = failure_summary(rows, reference)
    assert r['empty_masks'] == r['drops_over_0_1'] == r['dice_below_0_2'] == 1
    assert r['mean_relative_area_change'] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        failure_summary(rows * 2, reference)
