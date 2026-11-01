"""Plane-classifier training loop."""

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..common import NUM_TARGETS, SEED, STOP_FILE
from ..common import write_json as dump
from ..config import load_protocol
from ..data.batch import to_unit
from ..data.dataset import ids_for, loader
from ..evaluation.classification import evaluate
from ..models.classifier import Classifier
from ..models.segmentation import ROOT
from .optim import amp_step
from .runtime import event, resources, seed, validation_due


def _stop_requested():
    return (ROOT / STOP_FILE).exists()


def _update(m, opt, scaler, a, p, microbatch, grad_clip_norm):
    """One effective-batch cross-entropy step accumulated over microbatches."""
    opt.zero_grad(set_to_none=True)
    value = 0
    for j in range(0, len(a), microbatch):
        part = slice(j, j + microbatch)
        with torch.autocast('cuda', dtype=torch.float16):
            z = m(to_unit(a[part]))
        loss = F.cross_entropy(z.float(), p[part].cuda(), reduction='sum') / len(a)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        value += float(loss.detach())
    _, applied = amp_step(m, opt, scaler, grad_clip_norm)
    return value, applied


def train_classifier():
    assert not _stop_requested(), 'Stop requested; no classifier start'
    resources()
    seed()
    protocol = load_protocol(ROOT / 'study_plan.json')
    epochs = protocol.classifier_epochs
    rule = protocol.classifier_rule
    optimization = protocol.optimization
    out = ROOT / 'full/classifier'
    out.mkdir(parents=True, exist_ok=False)
    train = ids_for(0, classification=True)
    valid = ids_for(1, classification=True)
    m = Classifier().cuda()
    opt = torch.optim.AdamW(
        [
            dict(params=m.encoder.parameters(), lr=protocol.learning_rates.encoder),
            dict(params=m.head.parameters(), lr=protocol.learning_rates.decoder),
        ],
        weight_decay=optimization.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=optimization.cosine_eta_min)
    scaler = torch.amp.GradScaler('cuda')
    dl, sampler = loader(train, batch=rule.effective_batch, aug=True)
    history = []
    best = -1
    applied = attempted = 0
    start = time.perf_counter()
    dump(
        out / 'configuration.json',
        dict(
            outputs=NUM_TARGETS,
            seed=SEED,
            epochs=epochs,
            train_images=len(train),
            valid_images=len(valid),
            segmentation_missing_images_kept_for_known_plane_label=True,
            effective_batch=rule.effective_batch,
            microbatch=rule.microbatch,
        ),
    )
    event('CLASSIFIER_START', epochs=epochs, train_images=len(train), valid_images=len(valid))
    for epoch in range(epochs):
        assert not _stop_requested()
        sampler.epoch = epoch
        m.train()
        losses = []
        seen_epoch = []
        for a, _y, _v, p, idx in dl:
            seen_epoch.extend(int(i) for i in idx)
            value, step_applied = _update(m, opt, scaler, a, p, rule.microbatch, optimization.grad_clip_norm)
            applied += step_applied
            attempted += 1
            losses.append(value)
            if _stop_requested():
                torch.save(
                    dict(
                        state=m.state_dict(),
                        optimizer=opt.state_dict(),
                        scaler=scaler.state_dict(),
                        epoch=epoch,
                        partial_epoch=True,
                    ),
                    out / 'interrupted_partial.pt',
                )
                raise RuntimeError('Stop requested; classifier partial saved')
        assert sorted(seen_epoch) == sorted(train.tolist()), 'Epoch did not use every training image exactly once'
        sched.step()
        row = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)),
            attempted=attempted,
            applied=applied,
            seconds=time.perf_counter() - start,
        )
        if validation_due(epoch, epochs, optimization.validation_every):
            val = evaluate(m, valid)
            row.update(validation_accuracy=val['accuracy'], validation_macro_accuracy=val['macro_accuracy'])
            if val['macro_accuracy'] > best:
                best = val['macro_accuracy']
                torch.save(dict(state=m.state_dict(), epoch=epoch + 1), out / 'best.pt')
            event('CLASSIFIER_EPOCH', **row)
        history.append(row)
        dump(out / 'history.json', history)
        torch.save(
            dict(
                state=m.state_dict(),
                optimizer=opt.state_dict(),
                schedule=sched.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch + 1,
                history=history,
            ),
            out / 'resume.pt',
        )
    m.load_state_dict(torch.load(out / 'best.pt', weights_only=True)['state'])
    val = evaluate(m, valid)
    dump(out / 'validation.json', val)
    result = dict(
        seed=SEED,
        epochs=epochs,
        accuracy=val['accuracy'],
        macro_accuracy=val['macro_accuracy'],
        attempted=attempted,
        applied=applied,
        seconds=time.perf_counter() - start,
        parameters=sum(p.numel() for p in m.parameters()),
        test_evaluated=False,
    )
    dump(out / 'result.json', result)
    del m, opt, sched, scaler, dl
    gc.collect()
    torch.cuda.empty_cache()
    if result['macro_accuracy'] < rule.min_validation_macro_accuracy:
        raise RuntimeError(('Classifier validation insufficient; review before next model', result))
    return result
