import hashlib
import json

import pytest

from plane_conditioned_segmentation.paths import REPO_ROOT
from plane_conditioned_segmentation.pipeline.provisional import prepare
from plane_conditioned_segmentation.pipeline.stages import prerequisites, stage_models


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


def completed(root, name):
    folder = root / 'full' / name
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(('weights-' + name).encode())
    write(folder / 'result.json', dict(seed=17, macro_accuracy=0.9, learning_sufficiency_pass=True))


def test_shared_first_order_releases_baseline_before_experts(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    plan = json.loads((REPO_ROOT / 'configs/sixplane_radam_protocol.json').read_text())
    write(root / 'study_plan.json', plan)
    write(root / 'confirmed_targets.json', dict(targets=[dict(id=i) for i in range(6)]))
    assert list(stage_models(root)) == [
        'stage1-classifier',
        'stage3-baseline',
        'stage4-film',
        'stage5-condconv',
        'stage2-experts',
    ]
    with pytest.raises(ValueError, match='classifier'):
        prerequisites(root, 'stage3-baseline')
    completed(root, 'classifier')
    prerequisites(root, 'stage3-baseline')
    assert not (root / 'full/expert_0').exists()
    with pytest.raises(ValueError, match='plain'):
        prerequisites(root, 'stage4-film')
    completed(root, 'plain')
    completed(root, 'film')
    completed(root, 'condconv')
    prerequisites(root, 'stage2-experts')
    plan['training_stage_order'] = ['stage1-classifier'] * 5
    write(root / 'study_plan.json', plan)
    with pytest.raises(ValueError, match='order'):
        stage_models(root)


def test_provisional_preserves_sources_copies_hashes_and_refuses_tamper(tmp_path, monkeypatch):
    monkeypatch.setenv('PCS_RUNS_DIR', str(tmp_path))
    root = tmp_path / 'new'
    old = tmp_path / 'old'
    root.mkdir()
    old.mkdir()
    plan = json.loads((REPO_ROOT / 'configs/sixplane_radam_protocol.json').read_text())
    plan.update(provisional_reference_experts='old', segmentation_counts=[6] * 3, classification_counts=[6] * 3)
    refs = dict(
        manifest_sha256='m',
        split_sha256='s',
        cohort_sha256='c',
        immutable_data_source='../cache',
        source_archives='../archives',
        initialization_source='../init',
    )
    for folder in [root, old]:
        write(folder / 'study_plan.json', plan)
        write(folder / 'data_references.json', refs)
        write(folder / 'confirmed_targets.json', dict(targets=[dict(id=i) for i in range(6)]))
        write(folder / 'cohort.json', dict(allowed_indices=dict(train=[1], val=[2], test=[3])))
    for name in ['classifier', 'plain', 'film', 'condconv']:
        completed(root, name)
    for i in range(6):
        completed(old, f'expert_{i}')
    protected = {str(p): p.read_bytes() for folder in [root, old] for p in folder.rglob('*') if p.is_file()}
    destination = prepare(root)
    audit = json.loads((destination / 'provisional_model_provenance.json').read_text())
    assert len([p for p in audit['copied_artifacts_sha256'] if p.endswith('best.pt')]) == 10
    assert all(
        hashlib.sha256((destination / p).read_bytes()).hexdigest() == h
        for p, h in audit['copied_artifacts_sha256'].items()
    )
    assert all(__import__('pathlib').Path(p).read_bytes() == data for p, data in protected.items())
    assert prepare(root) == destination
    (destination / 'full/expert_0/best.pt').write_bytes(b'tamper')
    with pytest.raises(ValueError, match='changed'):
        prepare(root)
