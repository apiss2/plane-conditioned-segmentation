"""Protocol/runtime configuration: explicit template, frozen legacy plans and refused invalid settings."""

import copy
import json

import pytest

from plane_conditioned_segmentation import config
from plane_conditioned_segmentation.paths import REPO_ROOT


def template():
    return json.loads(config.PROTOCOL_TEMPLATE.read_text(encoding='utf-8'))


def test_template_states_every_section_explicitly():
    # Legacy fallbacks must only ever apply to plans written before the sections existed.
    assert set(config.LEGACY_V1_SECTIONS) <= set(template())
    protocol = config.load_protocol(config.PROTOCOL_TEMPLATE)
    assert protocol.input_height == 384 and protocol.input_widths == (544, 736)


def test_legacy_plan_without_new_sections_uses_recorded_v1_values():
    plan = {k: v for k, v in template().items() if k not in config.LEGACY_V1_SECTIONS}
    legacy = config.protocol_from_dict(plan)
    assert legacy.sufficiency_gate.train_dice == 0.70
    assert legacy.augmentation.affine_rotate_degrees == (-12.0, 12.0)
    assert legacy.evaluation.probability_threshold == 0.5
    assert legacy.classifier_rule.min_validation_macro_accuracy == 0.75


def test_run_without_plan_falls_back_to_template(tmp_path):
    assert config.run_protocol(tmp_path) == config.load_protocol(config.PROTOCOL_TEMPLATE)


@pytest.mark.parametrize(
    'fault',
    [
        'seed',
        'mixed_heights',
        'stride',
        'interpolation',
        'microbatch',
        'unknown_key',
        'missing_key',
        'string_number',
        'tuple_length',
    ],
)
def test_invalid_protocol_is_refused(fault):
    plan = copy.deepcopy(template())
    if fault == 'seed':
        plan['seed'] = 18
    elif fault == 'mixed_heights':
        plan['inputs'] = [[544, 384], [736, 416]]
    elif fault == 'stride':
        plan['inputs'] = [[540, 384]]
    elif fault == 'interpolation':
        plan['evaluation']['original_grid_interpolation'] = 'bicubic'
    elif fault == 'microbatch':
        plan['full_segmentation_rule']['microbatch'] = 3
    elif fault == 'unknown_key':
        plan['augmentation']['vertical_flip_p'] = 0.5
    elif fault == 'missing_key':
        del plan['stall_guard']['window']
    elif fault == 'string_number':
        plan['optimization']['weight_decay'] = '1e-4'
    else:
        plan['augmentation']['affine_scale'] = [0.9]
    with pytest.raises(ValueError):
        config.protocol_from_dict(plan)


def test_runtime_config_is_complete_and_strict(tmp_path):
    runtime = config.load_runtime()
    assert runtime.eval_batch_size > 0 and runtime.min_free_gpu_mib > 0
    value = json.loads((REPO_ROOT / 'configs/runtime.json').read_text(encoding='utf-8'))
    value['unexpected'] = 1
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        config.load_runtime(path)
