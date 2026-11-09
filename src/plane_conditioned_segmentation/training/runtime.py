"""Process-level helpers for training jobs: seeding, event log, resource guard and checkpoints."""

import json
import os
import random
import shlex
import subprocess
from pathlib import PureWindowsPath

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


def _pipeline_command(command):
    """Interpreter location alone says nothing about the work done by a Python process."""
    try:
        args = [a.strip('"') for a in shlex.split(command or '', posix=False)]
    except ValueError:
        return True  # Cannot classify a malformed command safely.
    if not args:
        return False
    executable = PureWindowsPath(args[0]).name.casefold()
    cli_args = args[1:] if executable in {'plane-seg', 'plane-seg.exe'} else None
    if '-m' in args:
        i = args.index('-m')
        module = args[i + 1] if i + 1 < len(args) else ''
        if module == 'plane_conditioned_segmentation.pipeline.report':
            return False
        if module == 'plane_conditioned_segmentation.cli':
            cli_args = args[i + 2 :]
    if cli_args is not None:
        i = 0
        while i < len(cli_args) and cli_args[i].startswith('--experiment'):
            i += 1 if '=' in cli_args[i] else 2
        command = cli_args[i] if i < len(cli_args) else None
        cpu_commands = {
            'stage7-report',
            'prepare-data',
            'prepare-radam',
            'prepare-sixplane',
            'prepare-continuation',
            'initialize-models',
            'register-existing',
            'status',
            'plan',
            'check-data',
            'verify',
        }
        return '--execute' in cli_args and command not in cpu_commands
    payload = ' '.join(args[1:]).casefold()
    markers = [*RUNTIME.pipeline_process_markers, 'run_sixplane_comparison.py']
    return any(marker.casefold() in payload for marker in markers)


def _gpu_process_ids():
    """CUDA/WDDM process membership catches GPU Python jobs with unknown script names."""
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    assert all(line.isdecimal() for line in lines), 'Unrecognized GPU process list; no stage started'
    return {int(line) for line in lines}


def _other_training(processes, gpu_pids=()):
    """Pipeline processes other than this one and its own launcher ancestors."""
    by_id = {p['ProcessId']: p for p in processes}
    allowed = {os.getpid()}
    pid = os.getpid()
    while pid in by_id and by_id[pid]['ParentProcessId'] not in allowed:
        pid = by_id[pid]['ParentProcessId']
        allowed.add(pid)
    return [
        p
        for p in processes
        if p['ProcessId'] not in allowed and (p['ProcessId'] in gpu_pids or _pipeline_command(p.get('CommandLine')))
    ]


def resources():
    """Refuse to start while other pipeline jobs run or the GPU lacks free memory."""
    processes = _python_processes()
    active = _other_training(processes)
    assert not active, active
    active = _other_training(processes, _gpu_process_ids())
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
