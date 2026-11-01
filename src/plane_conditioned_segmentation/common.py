"""Protocol constants and small stdlib-only JSON/hash helpers shared by every stage."""

import datetime
import hashlib
import json
from pathlib import Path

SEED = 17
NUM_TARGETS = 7
FULL_MODEL_COUNT = 11
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
