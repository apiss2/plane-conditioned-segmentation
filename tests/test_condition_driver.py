import copy
import importlib.util
import subprocess

import pytest

from plane_conditioned_segmentation.common import read_json, sha256_file, write_json
from plane_conditioned_segmentation.config import load_protocol
from plane_conditioned_segmentation.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location('condition_driver', REPO_ROOT / 'tools/run_condition_ablation.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_three_treatments_only_change_condition_with_same_operational_budget(tmp_path):
    original = read_json(REPO_ROOT / 'configs/sixplane_radam_protocol.json')
    saved = copy.deepcopy(original)
    normalized = []
    for name, condition in driver.TREATMENTS.items():
        plan = driver.training_plan(original, condition)
        assert plan['new_training_models'] == ['film'] and plan['test_evaluation_permitted'] is False
        assert plan['schedulefree']['max_epochs'] == 500 and plan['schedulefree']['patience'] == 30
        path = tmp_path / f'{name}.json'
        write_json(path, plan)
        load_protocol(path)
        del plan['condition_training']
        normalized.append(plan)
    assert all(plan == normalized[0] for plan in normalized)
    assert original == saved


@pytest.mark.parametrize('failure', [None, 'process', 'source'])
def test_driver_runs_only_six_validation_training_stages_and_stops_on_error(tmp_path, monkeypatch, failure):
    output = tmp_path / 'audit'
    (output / 'baseline').mkdir(parents=True)
    write_json(output / 'baseline/summary.json', dict(complete=True, manifest=dict(split='val')))
    code = tmp_path / 'source.py'
    code.write_text('source', encoding='utf-8')
    treatments = {}
    for key in driver.TREATMENTS:
        root = tmp_path / key
        root.mkdir()
        write_json(root / 'study_plan.json', {})
        treatments[key] = dict(experiment=key, protocol_sha256=sha256_file(root / 'study_plan.json'))
    record = dict(treatments=treatments, source_sha256={'source.py': sha256_file(code)}, pending=['human'])
    write_json(output / 'ablation_plan.json', record)
    monkeypatch.setattr(driver, 'experiment_root', lambda name: tmp_path / name)
    monkeypatch.setattr(driver, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(driver, 'comparison', lambda *args: None)
    called = []

    def launch(args, **kw):
        called.append(args)
        assert kw['check'] and kw['env']['PCS_EXPERIMENT'] in treatments
        assert 'stage6-evaluate' not in args
        if failure == 'process':
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(driver.subprocess, 'run', launch)
    if failure == 'source':
        code.write_text('changed', encoding='utf-8')
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            driver.execute(output)
        assert len(called) == (1 if failure == 'process' else 0)
        assert read_json(output / 'status.json')['state'] == 'stopped'
    else:
        driver.execute(output)
        assert len(called) == 6
        for args in called[1::2]:
            assert 'plane_conditioned_segmentation.evaluation.sensitivity' in args
        assert read_json(output / 'status.json')['state'] == 'screening_complete'
    assert not (output / 'ablation_process.lock').exists()


@pytest.mark.parametrize('mismatch', [False, True])
def test_comparison_requires_same_classifier_and_validation_cohort(tmp_path, monkeypatch, mismatch):
    manifest = dict(split='val', validation_ids_sha256='ids', cohort_sha256='cohort', classifier_sha256='fixed')
    cases = dict(
        film_oracle=dict(scores=dict(macro_dice=0.8)),
        film_estimated_soft=dict(scores=dict(macro_dice=0.75), failures=dict(dice_below_0_2=10)),
    )
    record = dict(treatments={}, pending=['human'])
    monkeypatch.setattr(driver, 'experiment_root', lambda name: tmp_path / name)
    for key in driver.TREATMENTS:
        out = tmp_path / f'validation_{key}'
        out.mkdir()
        m = {**manifest, 'classifier_sha256': 'changed'} if mismatch and key == 'noise' else manifest
        write_json(out / 'summary.json', dict(manifest=m, complete=True, conditions=cases))
        model = tmp_path / key / 'full/film'
        model.mkdir(parents=True)
        write_json(model / 'result.json', dict(learning_sufficiency_pass=True))
        record['treatments'][key] = dict(experiment=key)
    if mismatch:
        with pytest.raises(ValueError, match='identical validation'):
            driver.comparison(tmp_path, record)
        assert not (tmp_path / 'comparison.json').exists()
    else:
        driver.comparison(tmp_path, record)
        result = read_json(tmp_path / 'comparison.json')
        assert result['automatic_adoption'] is False
        assert not any(r['passes_exploratory_screen'] for r in result['treatments'].values())
