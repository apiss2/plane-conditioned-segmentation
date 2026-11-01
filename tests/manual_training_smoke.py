import datetime
import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from plane_conditioned_segmentation.config import PROTOCOL_TEMPLATE, load_protocol
from plane_conditioned_segmentation.models.classifier import Classifier, load_classifier
from plane_conditioned_segmentation.models.segmentation import ROOT, Model, load
from plane_conditioned_segmentation.training.optim import protocol_optimizer, update
from plane_conditioned_segmentation.training.runtime import checkpoint

PROTOCOL = load_protocol(PROTOCOL_TEMPLATE)
from plane_conditioned_segmentation.paths import REPO_ROOT

torch.set_num_threads(2)
assert torch.cuda.is_available()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


protected = [ROOT / 'FINAL_MODEL_LOCK.json', ROOT / 'final_evaluation/summary.json']
protected += list((ROOT / 'full').glob('*/best.pt')) + list((ROOT / 'full').glob('*/result.json'))
before = {str(p): digest(p) for p in protected}
rows = []
with tempfile.TemporaryDirectory(prefix='pcs-synthetic-step-') as td:
    for mode in ['classifier', 'expert', 'plain', 'film', 'condconv']:
        torch.manual_seed(17)
        a = torch.randint(0, 256, (2, 1, 96, 96), dtype=torch.uint8)
        p = torch.tensor([6, 6]) if mode == 'expert' else torch.tensor([0, 3])
        y = torch.zeros(2, 7, 96, 96)
        y[torch.arange(2), p, 24:64, 20:70] = 1
        v = torch.ones(2, 1, 96, 96)
        m = (Classifier() if mode == 'classifier' else Model(mode, 6 if mode == 'expert' else None)).cuda().train()
        opt = (
            torch.optim.AdamW(
                [dict(params=m.encoder.parameters(), lr=1e-5), dict(params=m.head.parameters(), lr=0.001)],
                weight_decay=1e-4,
            )
            if mode == 'classifier'
            else protocol_optimizer(m, PROTOCOL)
        )
        params = {n: z.detach().cpu().clone() for n, z in m.named_parameters()}
        scaler = torch.amp.GradScaler('cuda')
        skipped = 0
        for attempt in range(16):
            if mode == 'classifier':
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    z = m(a.cuda().float() / 255)
                ell = F.cross_entropy(z.float(), p.cuda(), reduction='sum') / len(a)
                assert torch.isfinite(ell)
                scaler.scale(ell).backward()
                scaler.unscale_(opt)
                grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
                old = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                result = dict(loss=float(ell.detach()), applied=scaler.get_scale() >= old, grad_norm=float(grad))
            else:
                result = update(
                    m,
                    opt,
                    scaler,
                    a,
                    y,
                    v,
                    p,
                    microbatch=PROTOCOL.full_segmentation_rule.microbatch,
                    grad_clip_norm=PROTOCOL.optimization.grad_clip_norm,
                )
            if result['applied']:
                break
            skipped += 1
            assert all(torch.equal(params[n], z.detach().cpu()) for n, z in m.named_parameters()), (
                'AMP skip mutated parameters'
            )
        changed = [n for n, z in m.named_parameters() if not torch.equal(params[n], z.detach().cpu())]
        assert result['applied'] and changed and math.isfinite(result['loss']) and math.isfinite(result['grad_norm']), (
            result
        )
        assert all(torch.isfinite(z).all().item() for z in m.parameters())
        output = Path(td) / (mode + '.pt')
        if mode == 'classifier':
            torch.save(dict(state=m.state_dict()), output)
            restored = load_classifier(output)
        else:
            checkpoint(output, m, synthetic_qa=True)
            restored = load(output)
        assert all(torch.equal(z.detach().cpu(), restored.state_dict()[n]) for n, z in m.state_dict().items())
        rows.append(
            dict(
                mode=mode,
                target=6 if mode == 'expert' else None,
                batch=2,
                height=96,
                width=96,
                **result,
                attempts=attempt + 1,
                initial_amp_skips=skipped,
                final_amp_scale=scaler.get_scale(),
                changed_parameter_tensors=len(changed),
                checkpoint_roundtrip=True,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
        del m, opt, scaler, params, restored
        if mode == 'classifier':
            del z, ell, grad
        gc.collect()
        torch.cuda.empty_cache()
assert before == {str(p): digest(p) for p in protected}
result = dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    device=torch.cuda.get_device_name(),
    data='synthetic only; no train/valid/test samples',
    paths=rows,
    protected_files_unchanged=len(before),
    temporary_checkpoint_outputs_removed=True,
    limits='One synthetic optimizer step per architecture; not a full training/convergence or quality-gate validation.',
)
(REPO_ROOT / '.local/training_smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('All five real forward/backward/AdamW AMP steps and checkpoint roundtrips passed; production hashes unchanged.')
