"""Cached arrays of the configured run: split/cohort selection, augmentation, width-bucketed loading."""

import os

from ..config import load_runtime, run_protocol

# Thread limits must be set before numpy/cv2/torch load their native runtimes.
RUNTIME = load_runtime()
_threads = str(RUNTIME.native_threads)
os.environ.update(
    NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS=_threads, MKL_NUM_THREADS=_threads, OPENBLAS_NUM_THREADS=_threads
)

import csv  # noqa: E402
import math  # noqa: E402

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402

from ..common import num_targets, SEED, SPLIT_NAMES, read_json, sha256_file  # noqa: E402
from ..paths import archives_root, data_root, experiment_root  # noqa: E402

ROOT = experiment_root()
PROTOCOL = run_protocol(ROOT)
SPEC = read_json(ROOT / 'confirmed_targets.json')
REF = read_json(ROOT / 'data_references.json')
DATA = data_root(ROOT, REF)
SOURCE_ARCHIVES = archives_root(ROOT, REF)
assert sha256_file(DATA / 'split.npy') == REF['split_sha256']
assert sha256_file(DATA / 'manifest.json') == REF['manifest_sha256']
COHORT = read_json(ROOT / 'cohort.json')
assert sha256_file(ROOT / 'cohort.json') == REF['cohort_sha256']

CLASSES = [r['name'] for r in SPEC['targets']]
# Source-cache plane/class IDs for each of the seven ordered targets.
OLD_PLANES = [r['source_plane_id'] for r in SPEC['targets']]
OLD_CLASSES = [r['source_cached_class_id'] for r in SPEC['targets']]
REMAP = {p: i for i, p in enumerate(OLD_PLANES)}
GEOMETRY_FIELDS = ('top', 'left', 'content_width', 'content_height', 'original_width', 'original_height')


def read_geometry(folder):
    """Letterbox geometry per cached image index."""
    with (folder / 'geometry.csv').open(newline='') as f:
        return {int(r['index']): {k: int(v) for k, v in r.items() if k in GEOMETRY_FIELDS} for r in csv.DictReader(f)}


def augment(settings=None):
    """Joint image/mask/support augmentation from the run protocol."""
    s = settings or PROTOCOL.augmentation
    return A.Compose(
        [
            A.HorizontalFlip(p=s.horizontal_flip_p),
            A.Affine(
                scale=s.affine_scale,
                translate_percent=s.affine_translate,
                rotate=s.affine_rotate_degrees,
                shear=(0, 0),
                keep_ratio=True,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=s.affine_p,
            ),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=s.contrast_limit, p=s.contrast_p),
            A.RandomGamma(gamma_limit=s.gamma_limit, p=s.gamma_p),
            A.MultiplicativeNoise(multiplier=s.noise_multiplier, per_channel=False, elementwise=True, p=s.noise_p),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=s.blur_sigma, p=s.blur_p),
        ],
        additional_targets={'valid': 'mask'},
        seed=SEED,
        strict=True,
    )


def ids_for(split, planes=None, classification=False):
    """Allowed cohort indices of ``split``; segmentation additionally requires a nonempty target mask."""
    plane = np.load(DATA / 'plane.npy')
    splits = np.load(DATA / 'split.npy')
    area = np.load(DATA / 'areas.npy')
    planes = list(range(num_targets())) if planes is None else list(planes)
    keep = np.zeros(len(plane), bool)
    for p in planes:
        keep |= (plane == OLD_PLANES[p]) & ((area[:, OLD_CLASSES[p]] > 0) | classification)
    allowed = np.isin(np.arange(len(plane)), COHORT['allowed_indices'][SPLIT_NAMES[split]])
    return np.where(keep & (splits == split) & allowed)[0]


def plane_ids(ids):
    return np.array([REMAP[int(p)] for p in np.load(DATA / 'plane.npy')[ids]])


class Images(Dataset):
    """Yields (image 1xHxW, masks 7xHxW, valid 1xHxW, target id, cache index); only the target mask is filled."""

    def __init__(self, ids, aug=False):
        self.ids = np.array(ids)
        self.aug = aug
        self.x = None

    def __len__(self):
        return len(self.ids)

    def _open(self):
        # Opened lazily so each DataLoader worker maps the arrays itself.
        cv2.setNumThreads(0)
        self.x = np.load(DATA / 'x.npy', mmap_mode='r')
        self.y = np.load(DATA / 'masks_packed.npy', mmap_mode='r')
        self.pp = np.load(DATA / 'plane.npy')
        self.widths = np.load(DATA / 'widths.npy')
        self.transform = augment()
        self.geom = read_geometry(DATA)

    def __getitem__(self, key):
        j, epoch = key if isinstance(key, tuple) else (key, 0)
        idx = int(self.ids[j])
        if self.x is None:
            self._open()
        p = REMAP[int(self.pp[idx])]
        w = int(self.widths[idx])
        height = self.x.shape[1]
        a = np.array(self.x[idx, :, :w])
        masks = np.zeros((height, w, num_targets()), np.uint8)
        masks[:, :, p] = np.unpackbits(self.y[idx, OLD_CLASSES[p], :, : w // 8], axis=-1)
        g = self.geom[idx]
        valid = np.zeros((height, w), np.uint8)
        valid[g['top'] : g['top'] + g['content_height'], g['left'] : g['left'] + g['content_width']] = 1
        if self.aug:
            # Fixed per (image, epoch), independent of worker scheduling.
            self.transform.set_random_seed((SEED + idx * 1009 + epoch * 1000003) % 2**32)
            r = self.transform(image=a, mask=masks, valid=valid)
            a, masks, valid = r['image'], r['mask'], r['valid']
        assert not (masks.astype(bool) & ~valid[:, :, None].astype(bool)).any()

        def tensor(array):
            return torch.from_numpy(np.ascontiguousarray(array))

        return tensor(a[None]), tensor(masks.transpose(2, 0, 1)), tensor(valid[None]), p, idx


class Batches(Sampler):
    """Width-homogeneous batches of (position, epoch); shuffled deterministically per epoch."""

    def __init__(self, ids, batch, shuffle):
        self.ids = np.array(ids)
        self.widths = np.load(DATA / 'widths.npy')[ids]
        self.batch = batch
        self.shuffle = shuffle
        self.epoch = 0

    def groups(self):
        rng = np.random.default_rng(SEED + self.epoch)
        out = []
        for w in np.unique(self.widths):
            local = np.where(self.widths == w)[0]
            if self.shuffle:
                rng.shuffle(local)
            out += [[(int(i), self.epoch) for i in local[j : j + self.batch]] for j in range(0, len(local), self.batch)]
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.groups())

    def __len__(self):
        return sum(math.ceil((self.widths == w).sum() / self.batch) for w in np.unique(self.widths))


def loader(ids, batch=None, aug=False):
    """Augmented loaders use worker processes; evaluation loaders stay in-process with the runtime batch size."""
    sampler = Batches(ids, RUNTIME.eval_batch_size if batch is None else batch, aug)
    workers = RUNTIME.train_loader_workers if aug else 0
    dl = DataLoader(
        Images(ids, aug),
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers else None,
    )
    return dl, sampler
