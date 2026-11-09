"""Unified classifier/expert/shared Schedule-Free loop. Legacy AdamW artifacts remain untouched."""

import dataclasses
import gc
import hashlib
import json
import time

import numpy as np
import torch

from ..common import SEED, STOP_FILE, num_targets, read_json, sha256_file, write_json
from ..config import load_protocol
from ..data.dataset import ROOT, Batches, ids_for, loader
from ..evaluation.classification import evaluate as classify
from ..evaluation.metrics import evaluate as segment
from ..models.classifier import Classifier
from ..models.segmentation import Model
from .control import EarlyStopping, load_policy, stopping_override
from .losses import LOSS
from .optim import update
from .runtime import event, resources, seed
from .schedulefree_state import learning_rates, make_optimizer, restore_state, save_state, set_mode
from .sufficiency import assess


def _id_hash(ids):
    return hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()


def train(label, mode, target=None, *, resume=False):
    """One fresh model, or an explicitly requested compatible epoch/batch-boundary resume.

    No completed model is overwritten. Partial resumes use saved optimizer/scaler/RNG/tracker state,
    deterministic sampler+augmentation and skip the recorded completed batches. No test set is read.
    """
    if (ROOT / STOP_FILE).exists():
        raise RuntimeError('Stop marker exists; no new stage')
    resources()
    seed()
    plan_path = ROOT / 'study_plan.json'
    protocol = load_protocol(plan_path)
    policy = load_policy(plan_path, label)
    override = stopping_override(plan_path, label)
    if policy is None or protocol.optimization.validation_every != 1:
        raise ValueError('Explicit Schedule-Free policy and validation every epoch required')
    classifier = mode == 'classifier'
    rule = protocol.classifier_rule if classifier else protocol.full_segmentation_rule
    train_ids = ids_for(0, None if target is None else [target], classification=classifier)
    valid_ids = ids_for(1, None if target is None else [target], classification=classifier)
    out = ROOT / 'full' / label
    if resume:
        if (out / 'result.json').exists():
            raise ValueError('Completed model cannot be resumed or overwritten')
        if not (out / 'resume.pt').is_file() and not (out / 'interrupted_partial.pt').is_file():
            raise ValueError('No compatible checkpoint for explicit resume')
    else:
        out.mkdir(parents=True, exist_ok=False)
    model = (Classifier() if classifier else Model(mode, target)).cuda()
    groups = (
        [
            dict(params=model.encoder.parameters(), lr=protocol.learning_rates.encoder),
            dict(params=model.head.parameters(), lr=protocol.learning_rates.decoder),
        ]
        if classifier
        else [
            dict(params=model.network.encoder.parameters(), lr=protocol.learning_rates.encoder),
            dict(
                params=[p for n, p in model.named_parameters() if not n.startswith('network.encoder.')],
                lr=protocol.learning_rates.decoder,
            ),
        ]
    )
    optimizer = make_optimizer(groups, protocol, policy)
    scaler = torch.amp.GradScaler('cuda')
    dl, sampler = loader(train_ids, batch=rule.effective_batch, aug=True)
    tracker = EarlyStopping(policy)
    history, attempted, applied, elapsed = [], 0, 0, 0.0
    first_epoch, skip_batches = 0, 0
    config = dict(
        label=label,
        mode=mode,
        target=target,
        seed=SEED,
        output_channels=num_targets() if target is None else 1,
        train_images=len(train_ids),
        valid_images=len(valid_ids),
        steps_per_epoch=len(Batches(train_ids, rule.effective_batch, True)),
        train_ids_sha256=_id_hash(train_ids),
        valid_ids_sha256=_id_hash(valid_ids),
        loss='cross_entropy' if classifier else LOSS,
        effective_batch=rule.effective_batch,
        microbatch=rule.microbatch,
        epochs=policy.max_epochs,
        encoder_lr=protocol.learning_rates.encoder,
        decoder_lr=protocol.learning_rates.decoder,
        weight_decay=protocol.optimization.weight_decay,
        policy=dataclasses.asdict(policy),
        scheduler=None,
        weight_decay_semantics='RAdamScheduleFree L2 penalty at training y',
        encoder_batchnorm='frozen running statistics',
        protocol_sha256=sha256_file(plan_path),
        optimizer_name='RAdamScheduleFree',
        monitor='validation_macro_recall' if classifier else 'validation_average_target_and_global_macro_Dice',
    )
    config = json.loads(json.dumps(config))
    if resume:
        saved_config = read_json(out / 'configuration.json')
        policy_transition = saved_config != config
        if policy_transition:
            adjusted = {**saved_config, 'policy': {**saved_config['policy'], 'min_epochs': 1, 'min_applied_updates': 0}}
            record = (override or {}).get('resume_records', {}).get(label, {})
            if adjusted != config or record.get('configuration_sha256') != sha256_file(out / 'configuration.json'):
                raise ValueError('Resume configuration differs beyond explicit minimum-removal transition')
        candidates = [p for p in [out / 'resume.pt', out / 'interrupted_partial.pt'] if p.is_file()]
        path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        ck = restore_state(path, model, optimizer, protocol_sha256=config['protocol_sha256'], label=label)
        scaler.load_state_dict(ck['scaler'])
        changing_policy = ck['early_stopping']['policy'] != dataclasses.asdict(policy)
        if changing_policy and (override or {}).get('resume_records', {}).get(label, {}).get(
            'checkpoint_sha256'
        ) != sha256_file(path):
            raise ValueError('Original transition checkpoint differs; no optimizer state modified')
        tracker.load_state_dict(ck['early_stopping'], allow_minimum_removal=bool(override))
        if policy_transition:
            transition_path = out / 'configuration_policy_transition.json'
            if not transition_path.exists():
                write_json(
                    transition_path,
                    dict(
                        original_configuration=saved_config,
                        effective_configuration=config,
                        original_configuration_sha256=sha256_file(out / 'configuration.json'),
                        resumed_checkpoint_sha256=sha256_file(path),
                        stopping_override_sha256=sha256_file(plan_path.parent / 'stopping_policy_override.json'),
                        preserved_validation_epochs=len(ck['history']),
                        preserved_completed_batches=ck['completed_batches'],
                    ),
                )
            effective_path = out / 'configuration_effective.json'
            if effective_path.exists() and read_json(effective_path) != config:
                raise ValueError('Effective transition configuration changed')
            if not effective_path.exists():
                write_json(effective_path, config)
        history, attempted, applied, elapsed = ck['history'], ck['attempted'], ck['applied'], ck['seconds']
        first_epoch, skip_batches = ck['next_epoch'], ck['completed_batches']
        resumed_losses, resumed_skips = ck.get('epoch_losses', []), ck.get('epoch_amp_skips', 0)
        if ck['target_count'] != num_targets() or ck['target'] != target:
            raise ValueError('Resume semantic target mismatch')
        event('TRAIN_RESUME', label=label, epoch=first_epoch + 1, completed_batches=skip_batches)
    else:
        write_json(out / 'configuration.json', config)
        event('CLASSIFIER_START' if classifier else 'TRAIN_START', **config)
    start = time.perf_counter()
    stop_reason = tracker.stop_reason()
    common = dict(
        label=label,
        mode=mode,
        target=target,
        target_count=num_targets(),
        loss_recipe='cross_entropy' if classifier else LOSS,
        protocol_sha256=config['protocol_sha256'],
        optimizer_name='RAdamScheduleFree',
    )

    def persist(path, next_epoch, completed_batches):
        save_state(
            path,
            model,
            optimizer,
            **common,
            resumable=True,
            epoch=len(history),
            next_epoch=next_epoch,
            completed_batches=completed_batches,
            scaler=scaler.state_dict(),
            history=history,
            attempted=attempted,
            applied=applied,
            seconds=elapsed + time.perf_counter() - start,
            early_stopping=tracker.state_dict(),
            epoch_losses=losses if completed_batches else [],
            epoch_amp_skips=skips if completed_batches else 0,
        )

    for epoch in range(first_epoch, policy.max_epochs):
        if stop_reason is not None:
            break
        sampler.epoch = epoch
        set_mode(model, optimizer, True)
        losses, skips, seen = (resumed_losses, resumed_skips, []) if skip_batches else ([], 0, [])
        # Resume RNG is restored after replaying the consumed deterministic data batches.
        replay_rng = torch.get_rng_state() if skip_batches else None
        replay_cuda = torch.cuda.get_rng_state_all() if skip_batches else None
        for batch_number, (a, y, v, p, idx) in enumerate(dl):
            seen.extend(int(i) for i in idx)
            if batch_number < skip_batches:
                continue
            if replay_rng is not None:
                torch.set_rng_state(replay_rng)
                torch.cuda.set_rng_state_all(replay_cuda)
                replay_rng = None
            if (ROOT / STOP_FILE).exists():
                persist(out / 'interrupted_partial.pt', epoch, batch_number)
                raise RuntimeError('Stop marker; Schedule-Free eval weights and optimizer resume state saved')
            if classifier:
                from .classifier import _update

                value, step_applied = _update(
                    model, optimizer, scaler, a, p, rule.microbatch, protocol.optimization.grad_clip_norm
                )
            else:
                row = update(
                    model,
                    optimizer,
                    scaler,
                    a,
                    y,
                    v,
                    p,
                    microbatch=rule.microbatch,
                    grad_clip_norm=protocol.optimization.grad_clip_norm,
                )
                value, step_applied = row['loss'], row['applied']
            losses.append(value)
            attempted += 1
            applied += int(step_applied)
            skips += int(not step_applied)
        skip_batches = 0
        if sorted(seen) != sorted(train_ids.tolist()):
            raise ValueError('Epoch did not cover every training image exactly once')
        set_mode(model, optimizer, False)
        row = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)) if losses else None,
            attempted=attempted,
            applied=applied,
            amp_skips=skips,
            seconds=elapsed + time.perf_counter() - start,
            learning_rates=learning_rates(optimizer),
        )
        if classifier:
            valid = classify(model, valid_ids)
            score = valid['macro_accuracy']
            row.update(validation_accuracy=valid['accuracy'], validation_macro_accuracy=score)
        else:
            valid = segment(model, valid_ids)
            raw = segment(model, train_ids)
            score = valid['summary']['selection']
            row.update(validation=valid['summary'], train=raw['summary'])
        decision = tracker.observe(float(score), epoch + 1, applied)
        row['early_stopping'] = decision
        if decision['selected']:
            save_state(
                out / 'best.pt', model, optimizer, **common, epoch=epoch + 1, selection=float(score), resumable=False
            )
        history.append(row)
        write_json(out / 'history.json', history)
        persist(out / 'resume.pt', epoch + 1, 0)
        event('CLASSIFIER_EPOCH' if classifier else 'EPOCH', label=label, **row)
        if decision['stop']:
            stop_reason = decision['reason']
            break
    # The resume optimizer remains tied to terminal parameters. Never transform selected-best weights with it.
    best = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['state'])
    model.eval()
    valid = classify(model, valid_ids) if classifier else segment(model, valid_ids)
    write_json(out / 'validation.json', valid)
    result = dict(
        **config,
        best_epoch=tracker.best_epoch,
        attempted=attempted,
        applied=applied,
        seconds=elapsed + time.perf_counter() - start,
        parameters=sum(p.numel() for p in model.parameters()),
        test_evaluated=False,
        stop_reason=stop_reason,
        early_stopping=tracker.state_dict(),
    )
    result['planned_max_epochs'] = policy.max_epochs
    result['stopping_policy_override_sha256'] = (
        sha256_file(plan_path.parent / 'stopping_policy_override.json') if override else None
    )
    result['epochs'] = len(history)
    if classifier:
        result.update(accuracy=valid['accuracy'], macro_accuracy=valid['macro_accuracy'])
    else:
        from .segmentation import review

        raw = segment(model, train_ids)
        write_json(out / 'train_metrics.json', raw)
        review(model, out, valid)
        assessment = assess(history, protocol.sufficiency_gate)
        write_json(out / 'terminal_learning_assessment.json', assessment)
        result.update(
            train=raw['summary'],
            validation=valid['summary'],
            last20percent_selection_gain=assessment['last20percent_selection_gain'],
            learning_sufficiency_pass=assessment['pass_gate'],
            learning_sufficiency_basis='terminal_training_history',
            terminal_learning_assessment=assessment,
        )
    write_json(out / 'result.json', result)
    event('CLASSIFIER_DONE' if classifier else 'TRAIN_DONE', **result)
    del model, optimizer, scaler, dl
    gc.collect()
    torch.cuda.empty_cache()
    if classifier and result['macro_accuracy'] < rule.min_validation_macro_accuracy:
        raise RuntimeError('Classifier validation quality gate requires review')
    if not classifier and not result['learning_sufficiency_pass']:
        raise RuntimeError(('Terminal learning sufficiency requires review; thresholds not changed', label))
    return result
