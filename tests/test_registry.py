import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from plane_conditioned_segmentation.config import PROTOCOL_TEMPLATE
from plane_conditioned_segmentation.pipeline.registry import describe, registration
from plane_conditioned_segmentation.pipeline.stages import STAGES


class RegistryTests(unittest.TestCase):
    def fixture(self, root):
        def write(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj))

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        checkpoints = {}
        for names in STAGES.values():
            for name in names:
                folder = root / 'full' / name
                folder.mkdir(parents=True)
                (folder / 'best.pt').write_bytes(name.encode())
                checkpoints[str((folder / 'best.pt').relative_to(root))] = sha(folder / 'best.pt')
                write(folder / 'result.json', dict(seed=17, macro_accuracy=0.8, learning_sufficiency_pass=True))
                write(folder / 'history.json', [])
                write(folder / 'configuration.json', dict(seed=17))
        write(root / 'FINAL_MODEL_LOCK.json', dict(checkpoints=checkpoints))
        write(
            root / 'study_plan.json',
            dict(
                json.loads(PROTOCOL_TEMPLATE.read_text(encoding='utf-8')),
                segmentation_counts=[7, 7, 7],
                classification_counts=[7, 7, 7],
            ),
        )
        write(root / 'cohort.json', {})
        write(root / 'confirmed_targets.json', dict(targets=[dict(id=i) for i in range(7)]))
        write(root / 'data/manifest.json', [])
        (root / 'data/split.npy').write_bytes(b'fixture data signature')
        write(
            root / 'data_references.json',
            dict(
                immutable_data_source='data',
                source_archives='archives',
                manifest_sha256=sha(root / 'data/manifest.json'),
                split_sha256=sha(root / 'data/split.npy'),
                cohort_sha256=sha(root / 'cohort.json'),
            ),
        )
        write(
            root / 'final_evaluation/summary.json',
            dict(full_training_models=11, seed=17, test_segmentation_images=7, scores={'fixture': {}}),
        )
        (root / 'final_evaluation/REPORT.md').write_text('fixture')

    def test_register_plan_write_reuse_and_changed_history_protection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            self.assertTrue(registration(root)['read_only'])
            self.assertFalse((root / 'run_manifest.json').exists())
            self.assertEqual(registration(root, True)['action'], 'registered_existing_completed_run')
            before = (root / 'run_manifest.json').read_bytes()
            self.assertEqual(registration(root, True)['action'], 'reuse_registration')
            self.assertTrue(describe(root)['complete'])
            (root / 'full/plain/history.json').write_text('[{"changed":true}]')
            with self.assertRaises(ValueError):
                registration(root, True)
            self.assertEqual((root / 'run_manifest.json').read_bytes(), before)

    def test_incomplete_run_cannot_register(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                registration(root, True)
            self.assertFalse((root / 'run_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
