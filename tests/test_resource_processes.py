"""Deterministic Windows process observations; no host job is launched or terminated."""

import subprocess
import types

import pytest

from plane_conditioned_segmentation.training import runtime

PYTHON = r'C:\Users\Example User\repo\plane-conditioned-segmentation\.venv\Scripts\python.exe'


@pytest.mark.parametrize(
    'command,blocked',
    [
        (f'"{PYTHON}" tmp\\build_report_pdf.py', False),
        (f'"{PYTHON}" "C:\\Example User\\plane-conditioned-segmentation\\tmp\\build_report_pdf.py"', False),
        (f'"{PYTHON}" -m plane_conditioned_segmentation.pipeline.report', False),
        (f'"{PYTHON}" -m plane_conditioned_segmentation.cli --experiment demo stage7-report --execute', False),
        ('"C:\\Example User\\plane-seg.exe" --experiment=demo stage7-report --execute', False),
        (f'"{PYTHON}" tools\\run_sixplane_comparison.py', True),
        (f'"{PYTHON}" "C:\\Example User\\tools\\run_sixplane_comparison.py"', True),
        (
            f'"{PYTHON}" -m plane_conditioned_segmentation.cli --experiment demo resume-model --model film --execute',
            True,
        ),
        ('plane-seg --experiment stage7-report stage4-film --execute', True),
        ('plane-seg --experiment stage3-baseline stage7-report --execute', False),
        ('plane-seg stage6-evaluate --execute', True),
        ('plane-seg stage3-baseline --plan', False),
        ('plane-seg prepare-continuation --execute', False),
        (f'"{PYTHON}" seven_pipeline.py', True),
        (f'"{PYTHON}" src\\plane_conditioned_segmentation\\training\\segmentation.py', True),
        ('python "unclosed', True),
        ('', False),
    ],
)
def test_entrypoint_classification_uses_work_not_interpreter_folder(command, blocked):
    assert runtime._pipeline_command(command) is blocked


def process(pid, parent, command):
    return dict(ProcessId=pid, ParentProcessId=parent, CommandLine=command)


def test_gpu_membership_overrides_cpu_report_name_and_keeps_own_ancestors(monkeypatch):
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 100)
    rows = [
        process(100, 90, 'python self.py'),
        process(90, 0, 'python tools/run_sixplane_comparison.py'),
        process(200, 1, f'"{PYTHON}" tmp\\build_report_pdf.py'),
        process(201, 200, f'"{PYTHON}" tmp\\build_report_pdf.py'),
        process(300, 1, 'python unrelated_name.py'),
    ]
    assert runtime._other_training(rows) == []
    assert [p['ProcessId'] for p in runtime._other_training(rows, {100, 90, 300})] == [300]
    assert [p['ProcessId'] for p in runtime._other_training(rows, {200})] == [200]


@pytest.mark.parametrize('output,expected', [('', set()), ('123\n456\n123\n', {123, 456})])
def test_gpu_pid_list_is_parsed_without_relying_on_process_names(monkeypatch, output, expected):
    monkeypatch.setattr(runtime.subprocess, 'check_output', lambda *a, **k: output)
    assert runtime._gpu_process_ids() == expected


def test_unreadable_or_malformed_gpu_list_never_permits_new_training(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, 'check_output', lambda *a, **k: 'N/A')
    with pytest.raises(AssertionError):
        runtime._gpu_process_ids()

    def unavailable(*a, **k):
        raise subprocess.CalledProcessError(1, a)

    monkeypatch.setattr(runtime.subprocess, 'check_output', unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        runtime._gpu_process_ids()


def test_cpu_report_is_allowed_but_real_gpu_pid_is_rejected_before_status_write(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 100)
    rows = [process(100, 0, 'python self.py'), process(200, 1, f'"{PYTHON}" tmp\\build_report_pdf.py')]
    monkeypatch.setattr(runtime, '_python_processes', lambda: rows)
    monkeypatch.setattr(
        runtime.subprocess,
        'check_output',
        lambda args, **kw: '' if '--query-compute-apps=pid' in args else '1500,12288,11',
    )
    assert runtime.resources()['other_training'] == []
    original = (tmp_path / 'last_resource_check.json').read_bytes()
    monkeypatch.setattr(
        runtime.subprocess,
        'check_output',
        lambda args, **kw: '200' if '--query-compute-apps=pid' in args else '1500,12288,11',
    )
    with pytest.raises(AssertionError):
        runtime.resources()
    assert (tmp_path / 'last_resource_check.json').read_bytes() == original
