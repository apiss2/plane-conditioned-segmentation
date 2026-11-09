"""Segmentation training loop for shared models and binary experts."""

import gc
import hashlib
import math
import time

import numpy as np
import torch

from ..common import SEED, STOP_FILE, num_targets, read_json
from ..common import write_json as dump
from ..config import load_protocol
from ..data.dataset import ROOT, Batches, ids_for, loader
from ..evaluation.metrics import evaluate
from ..evaluation.visualize import visualize
from ..models.segmentation import Model
from .losses import LOSS
from .optim import protocol_optimizer, update
from .runtime import checkpoint, event, resources, seed, validation_due
from .sufficiency import assess


def stop_check():
    assert not (ROOT / STOP_FILE).exists(), 'Stop requested; no subsequent stage'


def _plot_learning_curves(run):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hist = read_json(run / 'history.json')
    checked = [r for r in hist if 'validation' in r]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r['epoch'] for r in hist], [r['loss'] for r in hist])
    axes[0].set_title('Augmented train loss')
    for split, style in [('train', '--'), ('validation', '-')]:
        for metric in ['macro_dice', 'macro_global_dice']:
            axes[1].plot(
                [r['epoch'] for r in checked], [r[split][metric] for r in checked], style, label=split + ' ' + metric
            )
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / 'learning_curves.png', dpi=130)
    plt.close(fig)


def review(m, run, score):
    """Save worst/median validation examples per target and the learning curves."""
    selected = []
    for p in sorted(set(r['plane'] for r in score['rows'])):
        rows = sorted([r for r in score['rows'] if r['plane'] == p], key=lambda r: r['dice'])
        selected.extend([rows[0]['index'], rows[len(rows) // 2]['index']])
    visualize(m, selected, run / 'validation_worst_and_median.png')
    _plot_learning_curves(run)


def _ids_sha256(ids):
    return hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()


def _check_stall(label, history, row, epoch, guard):
    if epoch + 1 < guard.start_epoch or 'train' not in row:
        return
    for cid, current in row['train']['per_target'].items():
        past = [
            r['train']['per_target'][cid]['dice']
            for r in history
            if 'train' in r and r['epoch'] <= epoch + 1 - guard.window
        ]
        if current['dice'] < guard.max_dice and current['dice'] - max(past, default=0) < guard.min_gain:
            raise RuntimeError(('Training stalled for target', label, cid, current))


def train_seg(label, mode, target=None):
    stop_check()
    resources()
    protocol = load_protocol(ROOT / 'study_plan.json')
    rule = protocol.full_segmentation_rule
    optimization = protocol.optimization
    planes = None if target is None else [target]
    train = ids_for(0, planes)
    valid = ids_for(1, planes)
    steps = len(Batches(train, rule.effective_batch, True))
    epochs = max(rule.min_epochs, math.ceil(rule.min_attempted_updates / steps))
    run = ROOT / 'full' / label
    run.mkdir(parents=True, exist_ok=False)
    config = dict(
        label=label,
        mode=mode,
        target=target,
        output_channels=1 if target is not None else num_targets(),
        seed=SEED,
        epochs=epochs,
        train_images=len(train),
        valid_images=len(valid),
        steps_per_epoch=steps,
        train_ids_sha256=_ids_sha256(train),
        valid_ids_sha256=_ids_sha256(valid),
        loss=LOSS,
        encoder_lr=protocol.learning_rates.encoder,
        decoder_lr=protocol.learning_rates.decoder,
        effective_batch=rule.effective_batch,
        microbatch=rule.microbatch,
    )
    dump(run / 'configuration.json', config)
    seed()
    m = Model(mode, target).cuda()
    opt = protocol_optimizer(m, protocol)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=optimization.cosine_eta_min)
    scaler = torch.amp.GradScaler('cuda')
    dl, sampler = loader(train, batch=rule.effective_batch, aug=True)
    history = []
    best = -1
    best_epoch = 0
    attempted = applied = 0
    start = time.perf_counter()
    event('TRAIN_START', **config)
    for epoch in range(epochs):
        stop_check()
        sampler.epoch = epoch
        m.train()
        losses = []
        skips = 0
        seen_epoch = []
        for a, y, v, p, idx in dl:
            seen_epoch.extend(int(i) for i in idx)
            r = update(
                m, opt, scaler, a, y, v, p, microbatch=rule.microbatch, grad_clip_norm=optimization.grad_clip_norm
            )
            attempted += 1
            applied += r['applied']
            skips += not r['applied']
            losses.append(r['loss'])
            if (ROOT / STOP_FILE).exists():
                checkpoint(
                    run / 'interrupted_partial.pt',
                    m,
                    optimizer=opt.state_dict(),
                    schedule=sched.state_dict(),
                    scaler=scaler.state_dict(),
                    epoch=epoch,
                    partial_epoch=True,
                    completed_batches=len(losses),
                    attempted=attempted,
                    applied=applied,
                )
                raise RuntimeError('Stop marker; partial saved')
        assert sorted(seen_epoch) == sorted(train.tolist()), 'Epoch did not use every training image exactly once'
        sched.step()
        row = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)),
            attempted=attempted,
            applied=applied,
            amp_skips=skips,
            seconds=time.perf_counter() - start,
        )
        if validation_due(epoch, epochs, optimization.validation_every):
            val = evaluate(m, valid)
            raw = evaluate(m, train)
            row.update(validation=val['summary'], train=raw['summary'])
            quality = val['summary']['selection']
            if quality > best:
                best = quality
                best_epoch = epoch + 1
                checkpoint(run / 'best.pt', m, epoch=epoch + 1, selection=quality)
            event('EPOCH', label=label, **row)
        history.append(row)
        dump(run / 'history.json', history)
        checkpoint(
            run / 'resume.pt',
            m,
            epoch=epoch + 1,
            optimizer=opt.state_dict(),
            schedule=sched.state_dict(),
            scaler=scaler.state_dict(),
            history=history,
            attempted=attempted,
            applied=applied,
        )
        _check_stall(label, history, row, epoch, protocol.stall_guard)
    m.load_state_dict(torch.load(run / 'best.pt', weights_only=True)['state'])
    val = evaluate(m, valid)
    raw = evaluate(m, train)
    dump(run / 'validation.json', val)
    dump(run / 'train_metrics.json', raw)
    review(m, run, val)
    assessment = assess(history, protocol.sufficiency_gate)
    last_gain = assessment['last20percent_selection_gain']
    sufficient = assessment['pass_gate']
    dump(run / 'terminal_learning_assessment.json', assessment)
    result = dict(
        **config,
        best_epoch=best_epoch,
        attempted=attempted,
        applied=applied,
        train=raw['summary'],
        validation=val['summary'],
        last20percent_selection_gain=last_gain,
        learning_sufficiency_pass=sufficient,
        learning_sufficiency_basis='terminal_training_history',
        terminal_learning_assessment=assessment,
        parameters=sum(p.numel() for p in m.parameters()),
        seconds=time.perf_counter() - start,
        test_evaluated=False,
    )
    dump(run / 'result.json', result)
    event('TRAIN_DONE', **result)
    del m, opt, sched, scaler, dl
    gc.collect()
    torch.cuda.empty_cache()
    if not sufficient:
        raise RuntimeError(
            ('Learning sufficiency requires review before next model/test', label, result['train'], last_gain)
        )
    return result
