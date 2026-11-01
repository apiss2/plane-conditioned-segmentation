import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = repo / '.local/behavior_checks'
config = out / 'coverage.toml'
config.write_text(
    """[tool.coverage.run]
branch=true
source=["src/plane_conditioned_segmentation"]
relative_files=true
data_file=".local/behavior_checks/.coverage"
patch=["subprocess"]
[tool.coverage.report]
exclude_lines=[]
exclude_also=[]
show_missing=true
""",
    encoding='utf-8',
)
source_hashes = {
    p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (repo / 'src/plane_conditioned_segmentation').rglob('*.py')
}
record = dict(
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_sha256=source_hashes, runs=[]
)
for order in ['normal', 'reverse']:
    xml = out / (order + '_tests.xml')
    log = out / (order + '.log')
    transcript = out / (order + '_observations.json')
    if order == 'normal':
        command = [
            sys.executable,
            '-m',
            'pytest',
            '--cov=src/plane_conditioned_segmentation',
            '--cov-branch',
            '--cov-config',
            str(config),
            '--cov-report=json:' + str(out / 'after_coverage.json'),
            '--cov-report=term',
            '--junitxml',
            str(xml),
            '--behavior-transcript=' + str(transcript),
            '-q',
            '--tb=short',
        ]
    else:
        bootstrap = "import pytest,sys\nsys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')\nclass Reverse:\n def pytest_collection_modifyitems(self,items):items.reverse()\nraise SystemExit(pytest.main(['--no-cov','-q','--tb=short','--junitxml',sys.argv[1],'--behavior-transcript='+sys.argv[2]],plugins=[Reverse()]))"
        command = [sys.executable, '-I', '-c', bootstrap, str(xml), str(transcript)]
    env = os.environ.copy()
    env.update(
        PCS_DATA_DIR=str(out / 'must_not_be_used_data'), PCS_INITIALIZATION_DIR=str(out / 'must_not_be_used_weights')
    )
    env['RUST_LOG'] = 'warn'
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, encoding='utf-8', timeout=180)
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    print(result.stdout[-2600:], flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    suites = ET.parse(xml).getroot()
    cases = []
    for case in suites.iter('testcase'):
        status = (
            'failed'
            if case.find('failure') is not None or case.find('error') is not None
            else 'skipped'
            if case.find('skipped') is not None
            else 'passed'
        )
        cases.append((case.attrib.get('classname'), case.attrib['name'], status))
    assert all(c[2] == 'passed' for c in cases)
    record['runs'].append(dict(order=order, exit_code=0, junit_cases=len(cases), cases=sorted(cases)))
    if order == 'normal':
        shutil.copy2(out / '.coverage', out / 'after.coverage')
    assert source_hashes == {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (repo / 'src/plane_conditioned_segmentation').rglob('*.py')
    }
assert record['runs'][0]['cases'] == record['runs'][1]['cases']
observations = json.loads((out / 'normal_observations.json').read_text())
assert observations == json.loads((out / 'reverse_observations.json').read_text())
record.update(
    completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    same_test_outcomes_under_reverse_order=True,
    same_behavior_fingerprints_under_reverse_order=True,
    observed_contracts=len(observations),
    hostile_external_PCS_paths_cleared=True,
    source_unchanged_during_both_runs=True,
)
(out / 'repeatability.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in record.items() if k not in ['source_sha256', 'runs']}, ensure_ascii=False))
