"""Process-level helpers for training jobs: seeding, event log, resource guard and checkpoints."""

import json
import os
import random
import subprocess

import numpy as np
import torch

from ..common import SEED, num_targets, utc_now, write_json
from ..config import load_runtime
from ..paths import REPO_ROOT, experiment_root
from .losses import LOSS

ROOT = experiment_root()
RUNTIME = load_runtime()
torch.set_num_threads(RUNTIME.torch_threads)


def seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def event(kind, **kw):
    record = dict(utc=utc_now(), kind=kind, **kw)
    print(json.dumps(record), flush=True)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def validation_due(epoch, epochs, every):
    """Validate after the first epoch, every ``every`` epochs and the final epoch (0-based ``epoch``)."""
    return epoch == 0 or (epoch + 1) % every == 0 or epoch + 1 == epochs


def _python_processes():
    cmd = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','plane-seg.exe') } "
        '| Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'
    )
    r = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, errors='replace')
    assert r.returncode == 0, r.stderr
    processes = json.loads(r.stdout) if r.stdout.strip() else []
    return [processes] if isinstance(processes, dict) else processes


def _other_training(processes):
    """Pipeline processes other than this one and its own launcher ancestors."""
    by_id = {p['ProcessId']: p for p in processes}
    allowed = {os.getpid()}
    pid = os.getpid()
    while pid in by_id and by_id[pid]['ParentProcessId'] not in allowed:
        pid = by_id[pid]['ParentProcessId']
        allowed.add(pid)
    markers = [str(REPO_ROOT), *RUNTIME.pipeline_process_markers, str(ROOT)]
    return [
        p for p in processes if p['ProcessId'] not in allowed and any(t in (p['CommandLine'] or '') for t in markers)
    ]


def resources():
    """Refuse to start while other pipeline jobs run or the GPU lacks free memory."""
    active = _other_training(_python_processes())
    assert not active, active
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
    )
    used, total, util = map(int, gpu.strip().splitlines()[0].split(','))
    assert total - used >= RUNTIME.min_free_gpu_mib, (used, total)
    state = dict(utc=utc_now(), gpu_used_mib=used, gpu_total_mib=total, gpu_utilization=util, other_training=active)
    write_json(ROOT / 'last_resource_check.json', state)
    return state


def checkpoint(path, m, **kw):
    torch.save(
        dict(state=m.state_dict(), mode=m.mode, target=m.target, target_count=num_targets(), loss_recipe=LOSS, **kw),
        path,
    )
