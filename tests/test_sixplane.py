import json

import pytest
import torch

from plane_conditioned_segmentation.common import full_model_count, num_targets
from plane_conditioned_segmentation.evaluation.metrics import metric_row, metrics, summarize
from plane_conditioned_segmentation.models.classifier import Classifier
from plane_conditioned_segmentation.models.segmentation import Model
from plane_conditioned_segmentation.pipeline.stages import stage_models, stage_plan
from plane_conditioned_segmentation.training.losses import loss


@pytest.fixture
def six_run(monkeypatch, tmp_path):
    root = tmp_path / 'runs/six'
    root.mkdir(parents=True)
    (root / 'confirmed_targets.json').write_text(json.dumps(dict(targets=[dict(id=i) for i in range(6)])))
    monkeypatch.setenv('PCS_RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setenv('PCS_EXPERIMENT', 'six')
    return root


def test_six_target_stage_excludes_csp_and_preserves_seven(tmp_path, six_run):
    assert num_targets() == 6 and full_model_count() == 10
    assert stage_models(six_run)['stage2-experts'] == tuple(f'expert_{i}' for i in range(6))
    assert sum(map(len, stage_models(six_run).values())) == 10
    with pytest.raises(ValueError, match='outside'):
        stage_plan(six_run, 'stage2-experts', 6)
    seven = tmp_path / 'seven'
    seven.mkdir()
    (seven / 'confirmed_targets.json').write_text(json.dumps(dict(targets=[dict(id=i) for i in range(7)])))
    assert num_targets(seven) == 7 and full_model_count(seven) == 11
    assert len(stage_models(seven)['stage2-experts']) == 7


def test_six_target_real_training_paths_and_roundtrip(six_run, tmp_path):
    torch.manual_seed(17)
    image = torch.rand(2, 1, 64, 64)
    condition = torch.eye(6)[[0, 5]]
    truth = torch.zeros(2, 6, 64, 64)
    truth[0, 0, 20:35, 22:40] = 1
    truth[1, 5, 24:38, 20:36] = 1
    support = torch.ones(2, 1, 64, 64)
    for mode in ['plain', 'film', 'condconv', 'expert']:
        model = Model(mode, 0 if mode == 'expert' else None, initialize=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        before = model.network.segmentation_head[0].weight.detach().clone()
        z = model(image, condition)
        assert z.shape == (2, 1 if mode == 'expert' else 6, 64, 64)
        batch_truth = truth.clone()
        batch_plane = torch.tensor([0, 5])
        if mode == 'expert':
            batch_truth[1, 0] = batch_truth[1, 5]
            batch_truth[1, 5] = 0
            batch_plane[1] = 0
        value = loss(z, batch_truth, support, batch_plane, 2)
        assert torch.isfinite(value)
        value.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before, model.network.segmentation_head[0].weight)
        model.eval()
        path = tmp_path / f'{mode}.pt'
        torch.save(model.state_dict(), path)
        reloaded = Model(mode, 0 if mode == 'expert' else None, initialize=False).eval()
        reloaded.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            assert torch.equal(model(image, condition), reloaded(image, condition))
    classifier = Classifier(initialize=False)
    prediction = classifier(image)
    assert prediction.shape == (2, 6)
    opt = torch.optim.SGD(classifier.parameters(), lr=1e-3)
    before = classifier.head.weight.detach().clone()
    torch.nn.functional.cross_entropy(prediction, torch.tensor([0, 5])).backward()
    opt.step()
    assert not torch.equal(before, classifier.head.weight)


def test_six_target_fp_denominator_and_macro_contract(six_run):
    z = torch.full((1, 6, 2, 2), -10.0)
    y = torch.zeros_like(z)
    y[0, 0, 0, 0] = 1
    z[0, 0, 0, 0] = 10
    z[0, 5, 1, 1] = 10
    r = metrics(z, y, torch.ones(1, 1, 2, 2), torch.tensor([0]), threshold=0.5)
    assert r['dice'][0] == 1
    assert r['offtarget_fp_pixels'][0] == 1
    assert r['offtarget_fp_rate'][0] == pytest.approx(1 / (5 * 4))
    assert r['global_dice'][0] == pytest.approx(2 / 3)
    for bad in [6, 7]:
        with pytest.raises(ValueError):
            metrics(z, y, torch.ones(1, 1, 2, 2), torch.tensor([bad]), threshold=0.5)


def test_macro_has_six_equal_targets_no_csp(six_run):
    rows = []
    for p in range(6):
        truth = torch.zeros(1, 6, 2, 2)
        truth[0, p, 0, 0] = 1
        logits = torch.full_like(truth, -10.0)
        if p != 5:
            logits[0, p, 0, 0] = 10
        result = metrics(logits, truth, torch.ones(1, 1, 2, 2), torch.tensor([p]), threshold=0.5)
        rows.append(metric_row(result, 0, p, p))
    summary = summarize(rows)
    assert set(summary['per_target']) == {str(i) for i in range(6)}
    assert summary['macro_dice'] == pytest.approx(5 / 6, abs=1e-6)


def test_reuse_only_missing_expert_refuses_before_creating_run(six_run):
    from plane_conditioned_segmentation.config import PROTOCOL_TEMPLATE
    from plane_conditioned_segmentation.pipeline.stages import execute_stage

    plan = json.loads(PROTOCOL_TEMPLATE.read_text())
    plan['reuse_experts_only'] = True
    (six_run / 'study_plan.json').write_text(json.dumps(plan))
    folder = six_run / 'full/classifier'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'classifier fixture')
    (folder / 'result.json').write_text(json.dumps(dict(seed=17, macro_accuracy=0.9)))
    with pytest.raises(ValueError, match='reuse only'):
        execute_stage(six_run, 'stage2-experts')
    assert not (six_run / 'full/expert_0').exists()


def test_reused_weight_tampering_is_rejected_without_final_lock(six_run):
    import hashlib

    from plane_conditioned_segmentation.pipeline.stages import existing_model

    folder = six_run / 'full/expert_0'
    folder.mkdir(parents=True)
    weight = folder / 'best.pt'
    weight.write_bytes(b'original selected checkpoint')
    (folder / 'result.json').write_text(json.dumps(dict(seed=17, learning_sufficiency_pass=True)))
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    (six_run / 'expert_reuse_audit.json').write_text(
        json.dumps(dict(reused_experts=[dict(model='expert_0', checkpoint_sha256=digest)]))
    )
    assert existing_model(six_run, 'expert_0')['action'] == 'reuse'
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        existing_model(six_run, 'expert_0')
