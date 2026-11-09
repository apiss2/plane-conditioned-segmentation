"""Protocol constants and small stdlib-only JSON/hash helpers shared by every stage."""

import datetime
import hashlib
import json
from pathlib import Path

SEED = 17
SPLIT_NAMES = ('train', 'val', 'test')

LOCK_FILE = 'FINAL_MODEL_LOCK.json'
STOP_FILE = 'STOP_REQUESTED.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Write JSON atomically through a sibling temporary file."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def num_targets(root=None):
    """Target count of the selected run; historical seven-target runs remain readable."""
    from .paths import experiment_root

    root = Path(root) if root is not None else experiment_root()
    path = root / 'confirmed_targets.json'
    if not path.exists():
        return 7  # Historical/preparation callers without a run definition.
    targets = read_json(path)['targets']
    if not 2 <= len(targets) <= 7 or [t['id'] for t in targets] != list(range(len(targets))):
        raise ValueError('Run requires 2 through 7 contiguous ordered target IDs')
    return len(targets)


def full_model_count(root=None):
    return num_targets(root) + 4
