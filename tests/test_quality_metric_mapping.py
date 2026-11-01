"""Verify metric ownership on decorated, nested, one-line and async functions."""

import subprocess
import sys
from pathlib import Path

from plane_conditioned_segmentation.paths import REPO_ROOT


def test_cc_crap_mapping_does_not_credit_definition_as_body_execution(tmp_path):
    source = tmp_path / 'scopes.py'
    source.write_text("""import asyncio
def identity(f): return f
@identity
def decorated():
    return 2
def unused(): return 3
def outer():
    def nested(): return 4
    return nested
async def executed_async():
    return 5
async def unawaited(): return 6
decorated()
outer()
asyncio.run(executed_async())
unawaited().close()
""")
    code = """import importlib.util,json,sys
from pathlib import Path
import coverage
spec=importlib.util.spec_from_file_location('quality_tool',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=Path(sys.argv[2]);out=p.parent/'coverage.json';db=p.parent/'coverage.db'
c=coverage.Coverage(data_file=str(db),branch=True,config_file=False,include=[str(p)]);c.start();exec(compile(p.read_text(),str(p),'exec'),{});c.stop();c.save();c.json_report(outfile=str(out))
r=json.loads(out.read_text());file=next(iter(r['files'].values()));rows={x['function']:x for x in m.function_rows(p,db,file)}
assert rows['decorated']['line_coverage']==1
assert rows['unused']['line_coverage']==0 and rows['unused']['crap']==2
assert rows['outer']['line_coverage']==1 and rows['outer.nested']['line_coverage']==0
assert rows['executed_async']['line_coverage']==1 and rows['unawaited']['line_coverage']==0
assert not any(x['ambiguous_entry_collisions'] for x in rows.values())
print('decorator/nested/one-line/async function mapping passed')
"""
    result = subprocess.run(
        [sys.executable, '-I', '-c', code, str(REPO_ROOT / 'tools/quality_metrics.py'), str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
