import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import test_registry as registry_fixtures
from plane_conditioned_segmentation import cli
from plane_conditioned_segmentation.data import dataset as data
from plane_conditioned_segmentation.models import classifier as classification
from plane_conditioned_segmentation.models import segmentation as model
from plane_conditioned_segmentation.paths import REPO_ROOT
from plane_conditioned_segmentation.pipeline import report, stages, verification


@pytest.fixture
def fake_run(tmp_path):
    root = tmp_path / 'runs/fixture'
    registry_fixtures.RegistryTests().fixture(root)
    summary = dict(
        full_training_models=11,
        seed=17,
        test_segmentation_images=7,
        test_classification_images=7,
        scores={'synthetic_oracle': dict(macro_dice=0.5, dice=0.5, macro_global_dice=0.5)},
        classifier_accuracy=0.8,
        classifier_macro_accuracy=0.8,
        class_names=[str(i) for i in range(7)],
        baseline_macro_dice=0.1,
    )
    (root / 'final_evaluation/summary.json').write_text(json.dumps(summary))
    return root


def test_private_report_generation_is_data_driven_and_never_overwrites(monkeypatch, fake_run):
    root = fake_run
    path = root / 'final_evaluation/REPORT.md'
    path.unlink()
    monkeypatch.setattr(report, 'ROOT', root)
    monkeypatch.setattr(stages, 'REPO_ROOT', root.parent.parent)
    monkeypatch.setattr(cli, 'experiment_root', lambda name: root)
    monkeypatch.setenv('PCS_EXPERIMENT', 'fixture')
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert cli.main(['stage7-report', '--plan']) == 0
    assert not path.exists()
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(['stage7-report', '--execute']) == 0
    assert 'Classifier accuracy 0.800000' in path.read_text()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(AssertionError):
        report.main()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_verify_lock_and_aggregate_tampering_refuse(monkeypatch, fake_run):
    monkeypatch.setattr(verification, 'REPO_ROOT', fake_run.parent.parent)
    assert verification.verify(fake_run)['private_aggregate_results_identical'] is None
    folder = fake_run / 'aggregates'
    folder.mkdir()
    summary = json.loads((fake_run / 'final_evaluation/summary.json').read_text())
    summary['classifier_accuracy'] = 0.1
    (folder / 'sevenplane_seed17_aggregates.json').write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        verification.verify(fake_run)
    summary['classifier_accuracy'] = 0.8
    (folder / 'sevenplane_seed17_aggregates.json').write_text(json.dumps(summary))
    assert verification.verify(fake_run)['private_aggregate_results_identical']
    (fake_run / 'full/plain/best.pt').write_bytes(b'tampered')
    with pytest.raises(AssertionError):
        verification.verify(fake_run)


def test_verify_real_cache_and_real_untrained_checkpoint_reload(monkeypatch, tmp_path):
    # Actual ZIP preparation, loader, torch serialization and model forwards; no fake model/loaders.
    import test_portability as fixtures
    from plane_conditioned_segmentation.data import preparation
    from plane_conditioned_segmentation.training import runtime as core

    base = tmp_path / 'source'
    base.mkdir()
    manifest, archives = fixtures.PortableDataTests().fixtures(base)
    root = tmp_path / 'run'
    cache = tmp_path / 'cache'
    preparation.prepare(
        manifest,
        REPO_ROOT / 'configs/data_definition.json',
        root,
        cache,
        archives,
        REPO_ROOT / 'configs/sevenplane_protocol.json',
    )
    monkeypatch.setattr(data, 'DATA', cache)
    monkeypatch.setattr(data, 'ROOT', root)
    monkeypatch.setattr(data, 'COHORT', json.loads((root / 'cohort.json').read_text()))
    checkpoints = {}
    for stage, names in stages.STAGES.items():
        for name in names:
            folder = root / 'full' / name
            folder.mkdir(parents=True)
            path = folder / 'best.pt'
            if name == 'classifier':
                m = classification.Classifier(initialize=False)
                torch.save(dict(state=m.state_dict()), path)
            else:
                target = int(name.split('_')[1]) if name.startswith('expert_') else None
                m = model.Model('expert' if target is not None else name, target, initialize=False)
                core.checkpoint(path, m, epoch=0)
            checkpoints[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            del m
    (root / 'FINAL_MODEL_LOCK.json').write_text(json.dumps(dict(checkpoints=checkpoints)))
    (root / 'final_evaluation').mkdir()
    (root / 'final_evaluation/summary.json').write_text(
        json.dumps(dict(full_training_models=11, seed=17, test_segmentation_images=7, scores={}))
    )
    result = verification.verify(root, inference=True, audit_dir=tmp_path / 'audit')
    assert result['classifier_forwards'] == 2 and result['split_counts'] == [7, 7, 7]
    assert len(result['shared_model_reload_comparisons']) == 6 and all(
        r['max_reload_logit_difference'] == 0 for r in result['shared_model_reload_comparisons']
    )
    assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest for path, digest in checkpoints.items())


def test_source_guard_isolated_in_child_process(fake_run):
    repo = fake_run.parent.parent
    forbidden = repo / 'forbidden_original'
    forbidden.mkdir()
    # CPython audit hooks are otherwise opaque to tracing; this is test-only instrumentation.
    code = "from pathlib import Path; import sys; from plane_conditioned_segmentation.pipeline import verification as v; v.REPO_ROOT=Path(sys.argv[1]); original_register=sys.addaudithook; sys.addaudithook=lambda hook: (setattr(hook,'__cantrace__',True), original_register(hook))[-1]; r=v.verify(Path(sys.argv[2]),forbid_source=Path(sys.argv[3])); assert r['source_access_count']==0; print('source guard active')"
    result = subprocess.run(
        [sys.executable, '-I', '-c', code, str(repo), str(fake_run), str(forbidden)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert 'source guard active' in result.stdout


def test_cli_status_registration_and_verify_dispatch(monkeypatch, fake_run):
    monkeypatch.setattr(stages, 'REPO_ROOT', fake_run.parent.parent)
    monkeypatch.setattr(cli, 'experiment_root', lambda name: fake_run)
    monkeypatch.setenv('PCS_EXPERIMENT', 'fixture')

    def call(args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cli.main(args) == 0
        return json.loads(out.getvalue())

    assert call(['status'])['complete']
    assert call(['register-existing'])['read_only']
    assert call(['register-existing', '--execute'])['complete']
    monkeypatch.setattr(
        verification, 'verify', lambda root, inference, forbid: dict(inference=inference, root=str(root))
    )
    assert call(['verify', '--inference'])['inference'] is True


@pytest.mark.parametrize('kind', ['unregistered_summary', 'registered_tamper'])
def test_cli_reuse_rejects_saved_protocol_or_registered_hash_change(monkeypatch, fake_run, kind):
    monkeypatch.setattr(stages, 'REPO_ROOT', fake_run.parent.parent)
    if kind == 'registered_tamper':
        from plane_conditioned_segmentation.pipeline.registry import registration

        registration(fake_run, True)
    path = fake_run / 'final_evaluation/summary.json'
    summary = json.loads(path.read_text())
    summary['seed'] = 99 if kind == 'unregistered_summary' else 17
    summary['classifier_accuracy'] = 0.2
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        stages.evaluate_stage(fake_run, True)
