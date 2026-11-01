"""Provide synthetic import metadata before collection; never consume local research assets."""

import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

_repo = Path(__file__).resolve().parents[1]
_temporary = tempfile.TemporaryDirectory(prefix='pcs-pytest-metadata-')
_base = Path(_temporary.name)
_run = _base / 'runs/unit_import_metadata'
_run.mkdir(parents=True)
_cache = _base / 'cache'
_cache.mkdir()
_initialization = _base / 'initialization'
_initialization.mkdir()
_saved = {k: v for k, v in os.environ.items() if k.startswith('PCS_')}
for _key in list(_saved):
    os.environ.pop(_key, None)
os.environ['PCS_RUNS_DIR'] = str(_base / 'runs')
os.environ['PCS_EXPERIMENT'] = 'unit_import_metadata'
_definition = json.loads((_repo / 'configs/data_definition.json').read_text(encoding='utf-8'))
_cohort = dict(allowed_indices=dict(train=[], val=[], test=[]))


def _write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


_write(_cache / 'manifest.json', [])
np.save(_cache / 'split.npy', np.array([], dtype=np.int64))
_write(_run / 'cohort.json', _cohort)
_write(_run / 'confirmed_targets.json', dict(targets=_definition['targets']))
_write(_initialization / 'encoder_weights_metadata.json', dict(gray_mean=0.5, gray_std=0.25))
_sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
_write(
    _run / 'data_references.json',
    dict(
        immutable_data_source='../../cache',
        source_archives='../../archives',
        initialization_source='../../initialization',
        manifest_sha256=_sha(_cache / 'manifest.json'),
        split_sha256=_sha(_cache / 'split.npy'),
        cohort_sha256=_sha(_run / 'cohort.json'),
    ),
)
_forbidden = [
    os.path.normcase(str(p.resolve()))
    for p in [_repo / 'runs', _repo / 'data', _repo / 'archive', _repo.parent / 'fetal_sevenplane_smp_20261031']
]


def _deny_research_assets(event, args):
    if event in ['socket.connect', 'socket.getaddrinfo']:
        raise PermissionError('Tests cannot access the network')
    if event in ['open', 'os.listdir', 'os.scandir'] and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if any(path == p or path.startswith(p + os.sep) for p in _forbidden):
            raise PermissionError('Unit tests cannot read original/local research assets')


sys.addaudithook(_deny_research_assets)


@pytest.fixture(autouse=True)
def isolated_random_and_native_threads():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    torch_threads = torch.get_num_threads()
    cv_threads = cv2.getNumThreads()
    random.seed(19)
    np.random.seed(19)
    torch.random.default_generator.manual_seed(19)
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.set_num_threads(torch_threads)
        cv2.setNumThreads(cv_threads)


def pytest_unconfigure(config):
    for key in list(os.environ):
        if key.startswith('PCS_'):
            os.environ.pop(key, None)
    os.environ.update(_saved)
    _temporary.cleanup()


_behavior_records = {}


def record_behavior(key, value):
    canonical = json.loads(json.dumps(value, sort_keys=True))
    if key in _behavior_records:
        assert _behavior_records[key] == canonical, 'Behavior changed within a run: ' + key
    _behavior_records[key] = canonical


def pytest_addoption(parser):
    parser.addoption('--behavior-transcript', default=None, help='Private deterministic behavior fingerprint output')


def pytest_sessionfinish(session, exitstatus):
    value = session.config.getoption('--behavior-transcript')
    if value:
        target = Path(value).resolve()
        if not target.is_relative_to((_repo / '.local').resolve()):
            raise ValueError('Transcript must stay in private audit directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_behavior_records, sort_keys=True, indent=2), encoding='utf-8')
