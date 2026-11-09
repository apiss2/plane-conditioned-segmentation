"""Synthetic final bundles and archival paths; no private experiment data is read."""

import json
from types import SimpleNamespace

import pytest

from plane_conditioned_segmentation import paths
from plane_conditioned_segmentation.common import sha256_file
from plane_conditioned_segmentation.pipeline import results


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'runs/final'
    models = []
    files = {}
    for name in results.MODEL_NAMES:
        folder = root / 'full' / name
        folder.mkdir(parents=True)
        (folder / 'best.pt').write_bytes(('synthetic checkpoint ' + name).encode())
        for filename in ['result.json', 'history.json', 'configuration.json']:
            write(folder / filename, {})
        checkpoint = f'full/{name}/best.pt'
        models.append(
            dict(
                name=name,
                kind='classifier' if name == 'classifier' else 'segmentation',
                target=int(name.split('_')[1]) if name.startswith('expert_') else None,
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(root / checkpoint),
            )
        )
    for filename in [
        'study_plan.json',
        'confirmed_targets.json',
        'cohort.json',
        'data_references.json',
        'final_evaluation/summary.json',
        'final_evaluation/REPORT.md',
    ]:
        write(root / filename, {})
    write(
        root / 'FINAL_MODEL_LOCK.json',
        dict(
            checkpoints={r['checkpoint']: r['checkpoint_sha256'] for r in models},
            study_plan_sha256=sha256_file(root / 'study_plan.json'),
        ),
    )
    write(root / 'provenance/original_study_plan.json', {})
    original_lock = json.loads((root / 'FINAL_MODEL_LOCK.json').read_text())
    write(root / 'provenance/original_model_lock.json', original_lock)
    write(
        root / 'study_plan.json', dict(artifact_purpose='read_only_final_results', training_or_tuning_permitted=False)
    )
    current_lock = {**original_lock, 'study_plan_sha256': sha256_file(root / 'study_plan.json')}
    write(root / 'FINAL_MODEL_LOCK.json', current_lock)
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob('*') if p.is_file()}
    write(root / results.MANIFEST, dict(schema_version=1, read_only=True, models=models, artifacts_sha256=files))
    return root


def test_final_status_and_plan_are_ten_model_read_only_results(bundle):
    for command in ['status', 'plan', 'verify', 'stage6-evaluate', 'stage7-report']:
        state = results.run_command(bundle, SimpleNamespace(command=command, execute=True))
        assert state['model_count'] == 10
        assert state['classifier_count'] == 1
        assert state['segmentation_count'] == 9
        assert state['read_only'] and state['complete']
    assert not (bundle / 'comparison_process.lock').exists()


@pytest.mark.parametrize(
    'command',
    [
        'resume-model',
        'prepare-continuation',
        'prepare-radam',
        'prepare-data',
        'initialize-models',
        'register-existing',
        'stage1-classifier',
        'stage2-experts',
        'stage3-baseline',
        'stage4-film',
        'stage5-condconv',
    ],
)
def test_final_training_and_mutation_commands_are_rejected(bundle, command):
    with pytest.raises(ValueError, match='read-only'):
        results.run_command(bundle, SimpleNamespace(command=command, execute=True))


def test_corrupt_final_weight_is_not_silently_reused(bundle):
    (bundle / 'full/plain/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        results.validate(bundle)


def test_missing_final_artifact_is_not_recreated(bundle):
    (bundle / 'full/plain/history.json').unlink()
    with pytest.raises(FileNotFoundError):
        results.validate(bundle)


def test_wrong_model_set_is_rejected(bundle):
    path = bundle / results.MANIFEST
    record = json.loads(path.read_text())
    record['models'][0]['name'] = 'duplicate'
    write(path, record)
    with pytest.raises(ValueError, match='one classifier and nine'):
        results.validate(bundle)


@pytest.mark.parametrize('relative', ['../archive/old.pt', '/tmp/model.pt', 'C:/model.pt', 'bad\\model.pt'])
def test_manifest_artifact_path_escape_is_rejected(bundle, relative):
    with pytest.raises(ValueError, match='safe run-relative'):
        results._artifact_path(bundle, relative)


def test_cli_routes_final_bundle_without_training_handlers(bundle, monkeypatch, capsys):
    from plane_conditioned_segmentation.cli import main

    monkeypatch.setenv('PCS_RUNS_DIR', str(bundle.parent))
    assert main(['--experiment', 'final', 'status']) == 0
    assert json.loads(capsys.readouterr().out)['segmentation_count'] == 9
    with pytest.raises(SystemExit) as error:
        main(['--experiment', 'final', 'stage3-baseline', '--execute'])
    assert error.value.code == 2


def test_archive_lookup_and_external_data_references_keep_original_meaning(tmp_path, monkeypatch):
    base = tmp_path / 'runs'
    archived = base / 'archive/older'
    archived.mkdir(parents=True)
    write(base / 'archive/ARCHIVE_LAYOUT.json', {'schema_version': 1})
    monkeypatch.setenv('PCS_RUNS_DIR', str(base))
    for variable in ['PCS_DATA_DIR', 'PCS_ARCHIVES_DIR', 'PCS_INITIALIZATION_DIR']:
        monkeypatch.delenv(variable, raising=False)
    refs = dict(
        immutable_data_source='../../data/shared/cache',
        source_archives='../../data/shared/zips',
        initialization_source='../../data/shared/initialization',
    )
    write(archived / 'data_references.json', refs)
    assert paths.experiment_root('older') == archived
    assert paths.data_root(archived, refs).resolve() == tmp_path / 'data/shared/cache'
    assert paths.archives_root(archived, refs).resolve() == tmp_path / 'data/shared/zips'
    assert paths.initialization_root(archived).resolve() == tmp_path / 'data/shared/initialization'
    refs['immutable_data_source'] = 'local_cache'
    assert paths.data_root(archived, refs).resolve() == archived / 'local_cache'
    (base / 'older').mkdir()
    assert paths.experiment_root('older') == base / 'older'


def test_archive_execution_is_rejected_before_handler(tmp_path, monkeypatch):
    from plane_conditioned_segmentation.cli import main

    base = tmp_path / 'runs'
    (base / 'archive/older').mkdir(parents=True)
    write(base / 'archive/ARCHIVE_LAYOUT.json', {'schema_version': 1})
    monkeypatch.setenv('PCS_RUNS_DIR', str(base))
    with pytest.raises(SystemExit) as error:
        main(['--experiment', 'older', 'stage1-classifier', '--execute'])
    assert error.value.code == 2
