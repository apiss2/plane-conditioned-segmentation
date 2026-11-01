import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


def record_behavior(key, value):
    observer = sys.modules.get('conftest')
    if observer is not None and hasattr(observer, 'record_behavior'):
        observer.record_behavior(key, value)


from plane_conditioned_segmentation.data.preparation import member, prepare
from plane_conditioned_segmentation.paths import REPO_ROOT, experiment_root


class PortableDataTests(unittest.TestCase):
    def fixtures(self, base):
        archives = base / 'archives'
        archives.mkdir()
        definition = json.loads((REPO_ROOT / 'configs/data_definition.json').read_text())
        rows = []
        for t in definition['targets']:
            local = definition['class_map'][t['source_plane']].index(definition['classes'][t['source_cached_class_id']])
            with zipfile.ZipFile(archives / t['source_plane'], 'w') as z:
                for split in ['train', 'val', 'test']:
                    i = len(rows)
                    rng = np.random.default_rng(i)
                    buf = io.BytesIO()
                    Image.fromarray(rng.integers(0, 255, (12, 16), dtype=np.uint8)).save(buf, format='PNG')
                    image = f'synthetic/images/item_{i}.png'
                    label = f'synthetic/labels/item_{i}.txt'
                    z.writestr(zipfile.ZipInfo(image, date_time=(2020, 1, 1, 0, 0, 0)), buf.getvalue())
                    z.writestr(
                        zipfile.ZipInfo(label, date_time=(2020, 1, 1, 0, 0, 0)), f'{local} .2 .2 .8 .2 .8 .8 .2 .8\n'
                    )
                    rows.append(
                        dict(
                            index=i,
                            plane=t['source_plane_id'],
                            archive=t['source_plane'],
                            name=image,
                            label=label,
                            split=split,
                            geometry=dict(
                                original_width=16,
                                original_height=12,
                                input_width=544 if i % 2 else 736,
                                input_height=384,
                                content_width=512,
                                content_height=384,
                                left=16 if i % 2 else 112,
                                top=0,
                            ),
                        )
                    )
        manifest = base / 'manifest.json'
        manifest.write_text(json.dumps(rows))
        return manifest, archives

    def test_external_regeneration_and_loading_without_original_cache(self):
        with tempfile.TemporaryDirectory(prefix='pcs-portable-test-') as td:
            base = Path(td)
            manifest, archives = self.fixtures(base)
            env = os.environ.copy()
            env.update(
                PCS_RUNS_DIR=str(base / 'runs'),
                PCS_EXPERIMENT='synthetic',
                PCS_DATA_DIR=str(base / 'cache'),
                PCS_ARCHIVES_DIR=str(archives),
                PCS_INITIALIZATION_DIR=str(base / 'initialization'),
            )
            forbidden = REPO_ROOT / 'runs'
            code = """import os,sys
blocked=[os.path.normcase(os.path.join(os.path.dirname(sys.argv[1]),'data')),os.path.normcase(os.path.join(os.path.dirname(sys.argv[1]),'archive')),os.path.normcase(os.path.abspath(sys.argv[1])),os.path.normcase(os.path.join(os.path.dirname(os.path.dirname(sys.argv[1])),'fetal_sevenplane_smp_20261031'))]
def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if any(path==b or path.startswith(b+os.sep) for b in blocked):raise PermissionError('Original cache/source forbidden')
sys.addaudithook(guard)
from plane_conditioned_segmentation.cli import main
raise SystemExit(main(sys.argv[2:]))
"""

            def cli(*args):
                r = subprocess.run(
                    [sys.executable, '-I', '-B', '-c', code, str(forbidden), *args],
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                )
                self.assertEqual(r.returncode, 0, r.stderr)
                return json.loads(r.stdout)

            output = cli('prepare-data', '--manifest', str(manifest))
            self.assertEqual(output['images'], 21)
            data = cli('check-data')
            self.assertEqual(data['segmentation_counts'], [7, 7, 7])
            self.assertEqual(data['classification_counts'], [7, 7, 7])
            self.assertEqual(data['data'], str(base / 'cache'))
            self.assertGreater(data['target_pixels'], 0)
            # A second independent regeneration must reproduce numeric arrays and metadata.
            second = base / 'cache2'
            prepare(
                manifest,
                REPO_ROOT / 'configs/data_definition.json',
                base / 'runs/second',
                second,
                archives,
                REPO_ROOT / 'configs/sevenplane_protocol.json',
            )
            for name in [
                'x.npy',
                'masks_packed.npy',
                'areas.npy',
                'widths.npy',
                'plane.npy',
                'split.npy',
                'geometry.csv',
                'manifest.json',
            ]:
                self.assertEqual(
                    hashlib.sha256((base / 'cache' / name).read_bytes()).digest(),
                    hashlib.sha256((second / name).read_bytes()).digest(),
                    name,
                )
            record_behavior(
                'prepared_cache',
                {
                    n: hashlib.sha256((base / 'cache' / n).read_bytes()).hexdigest()
                    for n in [
                        'x.npy',
                        'masks_packed.npy',
                        'areas.npy',
                        'widths.npy',
                        'plane.npy',
                        'split.npy',
                        'geometry.csv',
                        'manifest.json',
                    ]
                },
            )
            with self.assertRaises(ValueError):
                prepare(
                    manifest,
                    REPO_ROOT / 'configs/data_definition.json',
                    base / 'runs/synthetic',
                    base / 'cache',
                    archives,
                    REPO_ROOT / 'configs/sevenplane_protocol.json',
                )
            # Random encoder fixture only: validate initialization + actual model forward with no cached weights.
            import segmentation_models_pytorch as smp
            import torch

            encoder = base / 'encoder.pt'
            torch.save(
                smp.encoders.get_encoder('mobilenet_v2', in_channels=1, depth=5, weights=None).state_dict(), encoder
            )
            metadata = base / 'normalization.json'
            metadata.write_text(json.dumps(dict(gray_mean=0.45, gray_std=0.25)))
            cli('initialize-models', '--encoder', str(encoder), '--metadata', str(metadata))
            forward = (
                code[: code.index('from plane_conditioned_segmentation.cli import main')]
                + """import torch
from plane_conditioned_segmentation.models.segmentation import Model
torch.set_num_threads(2)
with torch.inference_mode():
    for mode in ['plain','expert','film','condconv']:
        m=Model(mode,0 if mode=='expert' else None).eval();z=m(torch.zeros(1,1,64,64),torch.eye(7)[:1]);assert torch.isfinite(z).all()
print('external initialization and forwards passed')
"""
            )
            r = subprocess.run(
                [sys.executable, '-I', '-B', '-c', forward, str(forbidden)], env=env, capture_output=True, text=True
            )
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_storage_environment_override(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(os.environ, {'PCS_RUNS_DIR': td, 'PCS_EXPERIMENT': 'other'}),
        ):
            self.assertEqual(experiment_root(), Path(td) / 'other')

    def test_cli_honors_experiment_environment(self):
        from plane_conditioned_segmentation.cli import main

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(os.environ, {'PCS_RUNS_DIR': td, 'PCS_EXPERIMENT': 'external_named_run'}),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['plan']), 0)
            self.assertTrue(
                all(s['experiment'] == 'external_named_run' for s in json.loads(output.getvalue())['stages'])
            )
            self.assertFalse((Path(td) / 'external_named_run').exists())

    def test_default_output_and_input_storage_are_separate(self):
        from plane_conditioned_segmentation.paths import prepared_data_root

        clean = {k: v for k, v in os.environ.items() if not k.startswith('PCS_')}
        with patch.dict(os.environ, clean, clear=True):
            run = experiment_root('fixture')
            self.assertEqual(run, REPO_ROOT / 'runs/fixture')
            self.assertEqual(prepared_data_root(run), REPO_ROOT / 'data/fixture/cache')

    def test_member_escape_refused(self):
        for name in ['../secret', '/absolute', 'bad\\name', 'drive:secret']:
            with self.assertRaises(ValueError):
                member(name)

    def test_cross_split_exact_duplicate_refused_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            manifest, archives = self.fixtures(base)
            rows = json.loads(manifest.read_text())
            with zipfile.ZipFile(archives / rows[0]['archive'], 'a') as z:
                raw = z.read(rows[0]['name'])
                z.writestr('synthetic/images/duplicate.png', raw)
            rows[1]['name'] = 'synthetic/images/duplicate.png'
            manifest.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                prepare(
                    manifest,
                    REPO_ROOT / 'configs/data_definition.json',
                    base / 'run',
                    base / 'cache',
                    archives,
                    REPO_ROOT / 'configs/sevenplane_protocol.json',
                )
            self.assertFalse((base / 'run').exists())
            self.assertFalse((base / 'cache').exists())


if __name__ == '__main__':
    unittest.main()
