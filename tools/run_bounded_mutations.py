"""Six intentional faults in temporary source copies; never edit the working source."""

import ast
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = repo / '.local/behavior_checks'
out.mkdir(exist_ok=True)
cases = [
    (
        'inclusive_threshold',
        'evaluation/metrics.py',
        'pred = (z.float().sigmoid() > threshold) & v.bool()',
        'pred = (z.float().sigmoid() >= threshold) & v.bool()',
        'test_strict_half_probability_threshold',
    ),
    (
        'padding_leak',
        'evaluation/metrics.py',
        'pred = (z.float().sigmoid() > threshold) & v.bool()',
        'pred = z.float().sigmoid() > threshold',
        'test_padding_never_counts_as_fp_or_fn',
    ),
    (
        'wrong_negative_average',
        'training/losses.py',
        'pixels * (NUM_TARGETS - 1)',
        'pixels * NUM_TARGETS',
        'test_zero_logit_loss_has_closed_form_oracle',
    ),
    (
        'strict_gate_false_negative',
        'training/sufficiency.py',
        "train['macro_dice'] >= gate.train_dice",
        "train['macro_dice'] > gate.train_dice",
        'test_gate_ieee_boundary_contract',
    ),
    (
        'gate_always_pass_false_positive',
        'training/sufficiency.py',
        'passed = (',
        'passed = True or (',
        'test_gate_ieee_boundary_contract',
    ),
    (
        'stale_registration_accepted',
        'pipeline/registry.py',
        'if previous != record:',
        'if False:',
        'test_registration_plan_register_replay_tamper_state_machine',
    ),
]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original = {p.relative_to(repo).as_posix(): sha(p) for p in (repo / 'src/plane_conditioned_segmentation').rglob('*.py')}
rows = []
with tempfile.TemporaryDirectory(prefix='pcs-bounded-mutations-') as td:
    base = Path(td)
    for index, (name, file, old, new, test) in enumerate([('control', None, None, None, None), *cases]):
        folder = base / name
        src = folder / 'src'
        shutil.copytree(
            repo / 'src/plane_conditioned_segmentation',
            src / 'plane_conditioned_segmentation',
            ignore=shutil.ignore_patterns('__pycache__'),
        )
        # The copied package resolves REPO_ROOT to this folder, so it needs the configs beside it.
        shutil.copytree(repo / 'configs', folder / 'configs')
        if file:
            path = src / 'plane_conditioned_segmentation' / file
            text = path.read_text(encoding='utf-8')
            assert text.count(old) == 1, (name, text.count(old))
            text = text.replace(old, new)
            ast.parse(text)
            path.write_text(text, encoding='utf-8')
        selectors = (
            [str(repo / 'tests/test_behavior_contracts.py') + '::' + test]
            if test
            else [
                str(repo / 'tests/test_behavior_contracts.py') + '::' + t for t in dict.fromkeys(c[-1] for c in cases)
            ]
        )
        xml = out / (name + '.xml')
        log = out / (name + '.log')
        bootstrap = "import sys;sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8');from pathlib import Path;sys.path.insert(0,sys.argv[1]);import plane_conditioned_segmentation as p;assert Path(p.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve());import pytest;raise SystemExit(pytest.main(sys.argv[2:]))"
        command = [
            sys.executable,
            '-I',
            '-c',
            bootstrap,
            str(src),
            *selectors,
            '--no-cov',
            '-p',
            'no:cacheprovider',
            '-q',
            '--tb=short',
            '--basetemp',
            str(folder / 'scratch'),
            '--junitxml',
            str(xml),
        ]
        env = {k: v for k, v in os.environ.items() if not k.startswith('COVERAGE_')}
        env['PCS_DATA_DIR'] = str(folder / 'poisoned_external_data')
        env['PCS_INITIALIZATION_DIR'] = str(folder / 'poisoned_external_weights')
        result = subprocess.run(
            command, cwd=repo, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60
        )
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        suite = next(ET.parse(xml).getroot().iter('testsuite'))
        errors = int(suite.attrib['errors'])
        failures = int(suite.attrib['failures'])
        if not file:
            assert result.returncode == 0 and errors == failures == 0, result.stdout + result.stderr
        killed = bool(file and result.returncode == 1 and errors == 0 and failures > 0)
        rows.append(
            dict(
                name=name,
                function_test=test,
                exit_code=result.returncode,
                assertion_failures=failures,
                collection_or_runtime_errors=errors,
                killed=killed,
                working_source_modified=False,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
        if file and not killed:
            raise AssertionError('Mutation survived or failed due to infrastructure: ' + name)
assert original == {
    p.relative_to(repo).as_posix(): sha(p) for p in (repo / 'src/plane_conditioned_segmentation').rglob('*.py')
}
record = dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    control_passed=True,
    mutations=len(cases),
    killed=sum(r['killed'] for r in rows),
    rows=rows,
    temporary_source_copies_removed=True,
    working_source_sha256_unchanged=True,
    scope='Six hand-selected semantic faults only; not an exhaustive mutation score. Original inputs are synthetic temporary fixtures with fixed RNG/clock and no network.',
)
(out / 'mutations.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
