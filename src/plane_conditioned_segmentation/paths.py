"""Run outputs and private input data; environment is read before run-module imports."""

import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = 'final'


def configured_path(variable, default):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def experiment_root(name=None):
    name = name or os.environ.get('PCS_EXPERIMENT', DEFAULT_EXPERIMENT)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Experiment must be a simple name, not a path')
    base = configured_path('PCS_RUNS_DIR', REPO_ROOT / 'runs')
    current = base / name
    archived = base / 'archive' / name
    return archived if not current.exists() and is_archived_run(archived) and archived.is_dir() else current


def is_archived_run(root):
    return root.parent.name == 'archive' and (root.parent / 'ARCHIVE_LAYOUT.json').is_file()


def _reference_base(root, reference):
    # Archived records keep their original bytes, including ../../data references.
    # Resolve those records from their original logical runs/<name> location.
    external = Path(reference).parts[:2] == ('..', '..')
    return root.parent.parent / root.name if is_archived_run(root) and external else root


def prepared_data_root(root):
    return configured_path('PCS_DATA_DIR', REPO_ROOT / 'data' / root.name / 'cache')


def data_root(root, reference):
    return configured_path(
        'PCS_DATA_DIR', _reference_base(root, reference['immutable_data_source']) / reference['immutable_data_source']
    )


def archives_root(root, reference):
    return configured_path(
        'PCS_ARCHIVES_DIR', _reference_base(root, reference['source_archives']) / reference['source_archives']
    )


def initialization_root(root):
    refs = root / 'data_references.json'
    if refs.exists():
        reference = json.loads(refs.read_text(encoding='utf-8'))
        if reference.get('initialization_source'):
            default = _reference_base(root, reference['initialization_source']) / reference['initialization_source']
        else:
            default = data_root(root, reference).parent / 'initialization'
    else:
        default = prepared_data_root(root).parent / 'initialization'
    return configured_path('PCS_INITIALIZATION_DIR', default)
