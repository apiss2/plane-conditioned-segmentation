"""Run outputs and private input data; environment is read before run-module imports."""

import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = 'sixplane_radam_sf_continue500_20261105'


def configured_path(variable, default):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def experiment_root(name=None):
    name = name or os.environ.get('PCS_EXPERIMENT', DEFAULT_EXPERIMENT)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Experiment must be a simple name, not a path')
    return configured_path('PCS_RUNS_DIR', REPO_ROOT / 'runs') / name


def prepared_data_root(root):
    return configured_path('PCS_DATA_DIR', REPO_ROOT / 'data' / root.name / 'cache')


def data_root(root, reference):
    return configured_path('PCS_DATA_DIR', root / reference['immutable_data_source'])


def archives_root(root, reference):
    return configured_path('PCS_ARCHIVES_DIR', root / reference['source_archives'])


def initialization_root(root):
    refs = root / 'data_references.json'
    if refs.exists():
        reference = json.loads(refs.read_text(encoding='utf-8'))
        if reference.get('initialization_source'):
            default = root / reference['initialization_source']
        else:
            default = data_root(root, reference).parent / 'initialization'
    else:
        default = prepared_data_root(root).parent / 'initialization'
    return configured_path('PCS_INITIALIZATION_DIR', default)
