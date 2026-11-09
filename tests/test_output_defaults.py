"""Output selection tests use synthetic paths only, never saved research data."""

from plane_conditioned_segmentation import paths
from plane_conditioned_segmentation.cli import build_parser


def test_default_selects_latest_completed_experiment(monkeypatch, tmp_path):
    monkeypatch.delenv('PCS_EXPERIMENT', raising=False)
    monkeypatch.delenv('PCS_RUNS_DIR', raising=False)
    monkeypatch.setattr(paths, 'REPO_ROOT', tmp_path)
    expected = 'sixplane_radam_sf_continue500_20261105'
    assert paths.DEFAULT_EXPERIMENT == expected
    assert paths.experiment_root() == tmp_path / 'runs' / expected
    assert build_parser().parse_args(['status']).experiment == expected
    assert not (tmp_path / 'runs').exists()


def test_default_output_is_relative_to_repository_not_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv('PCS_RUNS_DIR', raising=False)
    repository = tmp_path / 'repo'
    other = tmp_path / 'elsewhere'
    other.mkdir()
    monkeypatch.setattr(paths, 'REPO_ROOT', repository)
    monkeypatch.chdir(other)
    assert paths.experiment_root('new_run') == repository / 'runs' / 'new_run'
    assert not repository.exists()


def test_explicit_experiment_and_storage_overrides_remain_available(monkeypatch, tmp_path):
    monkeypatch.setenv('PCS_RUNS_DIR', str(tmp_path))
    monkeypatch.setenv('PCS_EXPERIMENT', 'environment_run')
    assert paths.experiment_root() == tmp_path / 'environment_run'
    assert paths.experiment_root('older_run') == tmp_path / 'older_run'
    assert build_parser().parse_args(['--experiment', 'older_run', 'status']).experiment == 'older_run'
