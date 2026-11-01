"""Read/hash checks and optional validation inference; no archived code dependency."""

import json
import os
import sys
from pathlib import Path

from ..common import FULL_MODEL_COUNT, LOCK_FILE, NUM_TARGETS, utc_now
from ..common import read_json as read
from ..common import sha256_file as sha
from ..config import load_runtime
from ..paths import REPO_ROOT

AGGREGATE_KEYS = [
    'scores',
    'classifier_accuracy',
    'classifier_macro_accuracy',
    'class_names',
    'test_segmentation_images',
    'test_classification_images',
    'full_training_models',
    'seed',
    'baseline_macro_dice',
]
SHARED_MODES = ['plain', 'film', 'condconv']


def _install_source_guard(forbid_source, blocked):
    """Audit hook that denies file access below ``forbid_source``; proves itself with a probe open."""
    forbidden = os.path.normcase(str(forbid_source.resolve()))

    def audit(event, args):
        if event not in ['open', 'os.listdir', 'os.scandir', 'os.chdir'] or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if path == forbidden or path.startswith(forbidden + os.sep):
            blocked.append(path)
            raise PermissionError('Original source blocked during verification')

    sys.addaudithook(audit)
    try:
        open(os.path.join(forbidden, '__guard_probe__'), 'rb')
    except PermissionError:
        pass
    else:
        raise ValueError('Source guard ineffective')
    blocked.clear()


def _aggregates_match(root, summary):
    """None when no private aggregate copy exists; otherwise every shared key must equal the summary."""
    aggregates = root / 'aggregates/sevenplane_seed17_aggregates.json'
    if not aggregates.exists():
        return None
    saved = read(aggregates)
    for key in AGGREGATE_KEYS:
        assert saved[key] == summary[key], key
    return True


def _inference_checks(root, plan):
    """CPU reload determinism of shared models and classifier sanity on one validation image per width."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from ..data.dataset import DATA, Images, ids_for
    from ..models.classifier import load_classifier
    from ..models.segmentation import load

    assert Path(sys.prefix).resolve() == REPO_ROOT / '.venv'
    counts = [len(ids_for(i)) for i in range(3)]
    assert counts == plan['segmentation_counts']
    ids = ids_for(1)
    widths = np.load(DATA / 'widths.npy')
    selected = [int(next(i for i in ids if widths[i] == w)) for w in sorted(set(widths[ids]))]
    torch.set_num_threads(load_runtime().torch_threads)
    comparisons = []
    with torch.inference_mode():
        for mode in SHARED_MODES:
            model = load(root / f'full/{mode}/best.pt', 'cpu')
            reloaded = load(root / f'full/{mode}/best.pt', 'cpu')
            for idx in selected:
                a, _y, _v, p, _ = Images([idx])[0]
                x = a[None].float() / 255
                c = F.one_hot(torch.tensor([p]), NUM_TARGETS).float()
                z = model(x, c)
                again = reloaded(x, c)
                delta = float((z - again).abs().max())
                assert delta < 1e-6 and torch.isfinite(z).all()
                comparisons.append(
                    dict(model=mode, width=a.shape[-1], shape=list(z.shape), max_reload_logit_difference=delta)
                )
            del model, reloaded
        classifier = load_classifier(root / 'full/classifier/best.pt', 'cpu')
        for idx in selected:
            a, *_ = Images([idx])[0]
            probs = classifier(a[None].float() / 255).softmax(1)
            assert (
                probs.shape == (1, NUM_TARGETS) and torch.isfinite(probs).all() and abs(float(probs.sum()) - 1) < 1e-5
            )
    return dict(
        split_counts=counts,
        shared_model_reload_comparisons=comparisons,
        classifier_forwards=len(selected),
        base_prefix=sys.base_prefix,
    )


def verify(root, inference=False, forbid_source=None, audit_dir=None):
    blocked = []
    if forbid_source:
        _install_source_guard(forbid_source, blocked)
    lock = read(root / LOCK_FILE)
    assert len(lock['checkpoints']) == FULL_MODEL_COUNT
    assert all(sha(root / path) == digest for path, digest in lock['checkpoints'].items())
    summary = read(root / 'final_evaluation/summary.json')
    plan = read(root / 'study_plan.json')
    aggregate_match = _aggregates_match(root, summary)
    result = dict(
        utc=utc_now(),
        experiment=root.name,
        all_11_checkpoint_hashes_match=True,
        private_aggregate_results_identical=aggregate_match,
        one_environment=str(REPO_ROOT / '.venv'),
        new_training=False,
        test_used=False,
        inference=inference,
    )
    if inference:
        result.update(_inference_checks(root, plan))
    assert not blocked, blocked
    result['source_access_count'] = len(blocked)
    folder = Path(audit_dir) if audit_dir is not None else REPO_ROOT / '.local'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
