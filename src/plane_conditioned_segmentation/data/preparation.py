"""Explicit ZIP/polygon + fixed-manifest preparation; no split inference or downloads."""

import csv
import hashlib
import io
import os
import tempfile
from pathlib import Path, PurePosixPath

from ..common import num_targets, SEED, SPLIT_NAMES
from ..common import read_json as read
from ..common import sha256_file as sha
from ..common import write_json as dump
from ..config import protocol_from_dict

GEOMETRY_KEYS = (
    'input_width',
    'input_height',
    'content_width',
    'content_height',
    'left',
    'top',
    'original_width',
    'original_height',
)
PREPARATION = 'explicit_geometry_polygon_v1'
INITIALIZATION_FILES = (
    'encoder_weights_metadata.json',
    'encoder_mobilenet_v2_imagenet_gray.pt',
    f'fresh_common_initialization_seed{SEED}.pt',
)


def member(name):
    """Reject archive member names that could escape the archive or address another drive."""
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive member')
    return name


def prepare(manifest, definition, root, data, archives, protocol):
    with tempfile.TemporaryDirectory(prefix='pcs-prepare-') as spool:
        return _prepare(manifest, definition, root, data, archives, protocol, Path(spool))


def _validated_geometry(g, protocol):
    """Recorded letterbox geometry must place the content inside one of the protocol's input buckets."""
    if any(type(g.get(k)) is not int for k in GEOMETRY_KEYS):
        raise ValueError('Geometry values must be JSON integers')
    w, h = g['input_width'], g['input_height']
    cw, ch, left, top = g['content_width'], g['content_height'], g['left'], g['top']
    inside = 0 < cw <= w and 0 < ch <= h and 0 <= left <= w - cw and 0 <= top <= h - ch
    if (w, h) not in protocol.inputs or not inside:
        raise ValueError(f'Invalid recorded {protocol.input_height}-high bucket geometry')
    return w, cw, ch, left, top


def _rasterize(lines, classes, local, size, cw, ch, left, top):
    """Fill YOLO polygons, given in content coordinates, into one HxW mask per cached class."""
    import numpy as np
    from PIL import Image, ImageDraw

    masks = [Image.new('1', size) for _ in classes]
    for line in lines:
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        cid = int(values[0])
        xy = values[1:]
        if (
            values[0] != cid
            or not 0 <= cid < len(local)
            or len(xy) < 6
            or len(xy) % 2
            or not all(0 <= v <= 1 for v in xy)
        ):
            raise ValueError('Invalid YOLO polygon')
        points = [(xy[j] * (cw - 1) + left, xy[j + 1] * (ch - 1) + top) for j in range(0, len(xy), 2)]
        ImageDraw.Draw(masks[classes.index(local[cid])]).polygon(points, fill=1)
    return np.stack([np.asarray(m, dtype=np.uint8) for m in masks])


def _spool_record(i, row, byplane, spec, protocol, archives, seen, spool):
    """Validate one manifest row and spool its letterboxed image and packed masks; returns (width, areas)."""
    import zipfile

    import numpy as np
    from PIL import Image

    if row.get('index') != i or row.get('split') not in SPLIT_NAMES:
        raise ValueError('Manifest requires contiguous index and explicit train/val/test split')
    if type(row.get('allowed', True)) is not bool:
        raise ValueError('allowed must be a JSON boolean')
    t = byplane[row['plane']]
    if row['archive'] != t['source_plane']:
        raise ValueError('Archive/plane mismatch')
    archive = member(row['archive'])
    image = member(row['name'])
    label = member(row['label'])
    g = row['geometry']
    w, cw, ch, left, top = _validated_geometry(g, protocol)
    h = g['input_height']
    with zipfile.ZipFile(archives / archive) as z:
        im = Image.open(io.BytesIO(z.read(image))).convert('L')
        lines = z.read(label).decode('utf-8').splitlines()
    if im.size != (g['original_width'], g['original_height']):
        raise ValueError('Original dimensions mismatch')
    digest = hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()
    if row.get('sha256') and row['sha256'] != digest:
        raise ValueError('Source image hash mismatch')
    if digest in seen and seen[digest] != row['split']:
        raise ValueError('Exact image duplicate across splits')
    seen[digest] = row['split']
    resample = getattr(Image.Resampling, protocol.preprocessing.image_resample.upper())
    a = np.zeros((h, w), np.uint8)
    a[top : top + ch, left : left + cw] = np.asarray(im.resize((cw, ch), resample))
    masks = _rasterize(lines, spec['classes'], spec['class_map'][archive], (w, h), cw, ch, left, top)
    np.savez(spool / f'{i}.npz', x=a, packed=np.packbits(masks, axis=-1))
    return w, masks.sum((1, 2))


def _write_cache(data, records, classes, height, prepared, geometry, spool):
    """Assemble spooled items into fixed-width memory-mapped arrays plus per-image metadata."""
    import numpy as np

    count = len(records)
    maxw = max(w for w, _ in prepared)
    x = np.lib.format.open_memmap(data / 'x.npy', mode='w+', dtype=np.uint8, shape=(count, height, maxw))
    packed = np.lib.format.open_memmap(
        data / 'masks_packed.npy', mode='w+', dtype=np.uint8, shape=(count, len(classes), height, maxw // 8)
    )
    x[:] = 0
    packed[:] = 0
    areas = []
    widths = []
    for i, (width, area) in enumerate(prepared):
        with np.load(spool / f'{i}.npz') as item:
            x[i, :, :width] = item['x']
            packed[i, :, :, : width // 8] = item['packed']
        areas.append(area)
        widths.append(width)
    x.flush()
    packed.flush()
    del x, packed
    arrays = dict(
        areas=areas,
        widths=widths,
        plane=[r['plane'] for r in records],
        split=[SPLIT_NAMES.index(r['split']) for r in records],
    )
    for name, value in arrays.items():
        np.save(data / (name + '.npy'), np.asarray(value))
    with (data / 'geometry.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(geometry[0]))
        writer.writeheader()
        writer.writerows(geometry)
    return np.asarray(areas)


def _write_cohort(root, records):
    allowed = {n: [i for i, r in enumerate(records) if r['split'] == n and r.get('allowed', True)] for n in SPLIT_NAMES}
    dump(
        root / 'cohort.json',
        dict(
            allowed_indices=allowed,
            quarantine_count=sum(not r.get('allowed', True) for r in records),
            quarantine_by_original_split={
                n: sum(r['split'] == n and not r.get('allowed', True) for r in records) for n in SPLIT_NAMES
            },
            patient_independence_verified=False,
            policy='Explicit manifest allowed flags; no score-based selection or automatic grouping',
        ),
    )
    return allowed


def _write_targets(root, targets, records, allowed, areas):
    """Record classification and segmentation-observed indices for each target and split."""
    for t in targets:
        t['splits'] = {}
        for split, n in zip(['train', 'valid', 'test'], SPLIT_NAMES):
            ids = [i for i in allowed[n] if records[i]['plane'] == t['source_plane_id']]
            seg = [i for i in ids if areas[i, t['source_cached_class_id']] > 0]
            t['splits'][split] = dict(
                classification_images=len(ids),
                classification_indices=ids,
                segmentation_observed_images=len(seg),
                segmentation_indices=seg,
            )
    dump(root / 'confirmed_targets.json', dict(targets=targets))


def _write_study_plan(root, plan, targets):
    plan = dict(plan)
    splits = ['train', 'valid', 'test']
    plan['classification_counts'] = [sum(t['splits'][s]['classification_images'] for t in targets) for s in splits]
    plan['segmentation_counts'] = [sum(t['splits'][s]['segmentation_observed_images'] for t in targets) for s in splits]
    plan['data_preparation'] = PREPARATION + '; not a claim to reproduce private historical cohort'
    dump(root / 'study_plan.json', plan)
    return plan


def _reference(path, root):
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return str(path)  # Different Windows drives; private run metadata only.


def _prepare(manifest, definition, root, data, archives, protocol, spool):
    root = Path(root).resolve()
    data = Path(data).resolve()
    archives = Path(archives).resolve()
    if root == data or data in root.parents:
        raise ValueError('Data destination cannot contain the run destination')
    if root.exists() or data.exists():
        raise ValueError('New run and data destinations required; no overwrite/resume')
    records = read(manifest)
    spec = read(definition)
    template = read(protocol)
    settings = protocol_from_dict(template, Path(protocol).name)
    targets = spec['targets']
    classes = spec['classes']
    if not 2 <= len(targets) <= 7 or [t['id'] for t in targets] != list(range(len(targets))):
        raise ValueError('Two through seven contiguous ordered targets required')
    byplane = {t['source_plane_id']: t for t in targets}
    if len(byplane) != len(targets) or not records:
        raise ValueError('Distinct source planes and nonempty explicit manifest required')
    # Validate all inputs before creating output; spool one decoded image at a time.
    seen = {}
    prepared = [_spool_record(i, row, byplane, spec, settings, archives, seen, spool) for i, row in enumerate(records)]
    geometry = [dict(index=i, **row['geometry']) for i, row in enumerate(records)]
    root.mkdir(parents=True)
    data.mkdir(parents=True)
    areas = _write_cache(data, records, classes, settings.input_height, prepared, geometry, spool)
    del prepared
    dump(data / 'manifest.json', records)
    dump(
        data / 'audit.json',
        dict(
            classes=classes,
            class_map=spec['class_map'],
            split_policy='User-supplied fixed manifest; patient independence unverified',
            preparation=PREPARATION,
        ),
    )
    allowed = _write_cohort(root, records)
    _write_targets(root, targets, records, allowed, areas)
    plan = _write_study_plan(root, template, targets)
    initialization = Path(os.environ.get('PCS_INITIALIZATION_DIR', str(data.parent / 'initialization')))
    dump(
        root / 'data_references.json',
        dict(
            immutable_data_source=_reference(data, root),
            source_archives=_reference(archives, root),
            initialization_source=_reference(initialization.expanduser().resolve(), root),
            manifest_sha256=sha(data / 'manifest.json'),
            split_sha256=sha(data / 'split.npy'),
            cohort_sha256=sha(root / 'cohort.json'),
        ),
    )
    dump(
        root / 'preparation.json',
        dict(
            manifest_sha256=sha(manifest),
            definition_sha256=sha(definition),
            protocol_sha256=sha(protocol),
            images=len(records),
            no_split_inference=True,
            patient_independence_verified=False,
        ),
    )
    return dict(
        images=len(records),
        run=str(root),
        data=str(data),
        classification_counts=plan['classification_counts'],
        segmentation_counts=plan['segmentation_counts'],
    )


def initialize(root, encoder, metadata):
    import torch

    root = Path(root)
    if any((root / n).exists() for n in INITIALIZATION_FILES):
        raise ValueError('Initialization exists; refusing overwrite')
    value = read(metadata)
    if not all(k in value for k in ['gray_mean', 'gray_std']):
        raise ValueError('Explicit normalization metadata required')
    mean = torch.as_tensor(value['gray_mean'])
    std = torch.as_tensor(value['gray_std'])
    if (
        mean.numel() != 1
        or std.numel() != 1
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise ValueError('One finite grayscale mean and positive std required')
    state = torch.load(encoder, map_location='cpu', weights_only=True)
    # Validate architecture/state before publishing initialization files.
    from ..models.segmentation import build_unet

    torch.manual_seed(SEED)
    network = build_unet(num_targets())
    network.encoder.load_state_dict(state)
    root.mkdir(parents=True, exist_ok=True)
    dump(root / 'encoder_weights_metadata.json', value)
    torch.save(state, root / 'encoder_mobilenet_v2_imagenet_gray.pt')
    torch.save(network.state_dict(), root / INITIALIZATION_FILES[2])
    return dict(seed=SEED, download=False, encoder_sha256=sha(encoder), root=str(root))
