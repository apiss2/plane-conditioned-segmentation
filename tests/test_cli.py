import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from plane_conditioned_segmentation.paths import experiment_root
from plane_conditioned_segmentation.pipeline.stages import STAGES, existing_model, prerequisites, stage_plan
from plane_conditioned_segmentation.training.sufficiency import self_check


class StageTests(unittest.TestCase):
    def test_order_and_count(self):
        self.assertEqual(
            list(STAGES), ['stage1-classifier', 'stage2-experts', 'stage3-baseline', 'stage4-film', 'stage5-condconv']
        )
        self.assertEqual(sum(len(v) for v in STAGES.values()), 11)

    def test_paths_reject_escape(self):
        with self.assertRaises(ValueError):
            experiment_root('../outside')

    def test_partial_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'full/plain').mkdir(parents=True)
            self.assertEqual(existing_model(root, 'plain')['action'], 'blocked_inspect_partial')

    def test_missing_prerequisite(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                prerequisites(Path(td), 'stage2-experts')

    def test_checkpoint_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / 'full/classifier'
            folder.mkdir(parents=True)
            weight = folder / 'best.pt'
            weight.write_bytes(b'changed')
            (folder / 'result.json').write_text(json.dumps(dict(seed=17)))
            (root / 'FINAL_MODEL_LOCK.json').write_text(
                json.dumps(dict(checkpoints={str(weight.relative_to(root)): '0' * 64}))
            )
            with self.assertRaises(ValueError):
                existing_model(root, 'classifier')

    def test_target_selection(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(stage_plan(Path(td), 'stage2-experts', 3)['models'][0]['name'], 'expert_3')
            with self.assertRaises(ValueError):
                stage_plan(Path(td), 'stage2-experts', 7)

    def test_recorded_sufficiency_unchanged(self):
        self.assertTrue(self_check())


if __name__ == '__main__':
    unittest.main()
