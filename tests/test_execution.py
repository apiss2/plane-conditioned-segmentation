import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plane_conditioned_segmentation import cli
from plane_conditioned_segmentation.config import PROTOCOL_TEMPLATE
from plane_conditioned_segmentation.pipeline import stages


class ExecutionTests(unittest.TestCase):
    def result(self, root, name, **extra):
        if not (root / 'study_plan.json').exists():
            (root / 'study_plan.json').write_text(PROTOCOL_TEMPLATE.read_text(encoding='utf-8'))
        folder = root / 'full' / name
        folder.mkdir(parents=True)
        (folder / 'best.pt').write_bytes(b'fixture')
        (folder / 'result.json').write_text(
            json.dumps(dict(seed=17, learning_sufficiency_pass=True, macro_accuracy=0.8, **extra))
        )

    def test_dispatch_only_selected_stage(self):
        # Replace expensive training only; exercise actual stage execution/selection.
        for stage in stages.STAGES:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for prior, names in stages.STAGES.items():
                    if prior == stage:
                        break
                    for name in names:
                        self.result(root, name)
                calls = []

                def classifier():
                    calls.append(('classifier',))
                    self.result(root, 'classifier')

                def seg(name, mode, target):
                    calls.append((name, mode, target))
                    self.result(root, name)

                with (
                    patch.object(stages, 'REPO_ROOT', root),
                    patch('plane_conditioned_segmentation.training.classifier.train_classifier', classifier),
                    patch('plane_conditioned_segmentation.training.segmentation.train_seg', seg),
                ):
                    output = stages.execute_stage(root, stage, 3 if stage == 'stage2-experts' else None)
                expected = (
                    [('classifier',)]
                    if stage == 'stage1-classifier'
                    else [('expert_3', 'expert', 3)]
                    if stage == 'stage2-experts'
                    else [(stages.STAGES[stage][0], stages.STAGES[stage][0], None)]
                )
                self.assertEqual(calls, expected)
                self.assertTrue(all(x['action'] == 'reuse' for x in output['models']))
                self.assertFalse((root / '.local/execution.lock').exists())

    def test_all_experts_dispatch_sequentially(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.result(root, 'classifier')
            calls = []

            def seg(name, mode, target):
                calls.append(target)
                self.result(root, name)

            with (
                patch.object(stages, 'REPO_ROOT', root),
                patch('plane_conditioned_segmentation.training.segmentation.train_seg', seg),
            ):
                stages.execute_stage(root, 'stage2-experts')
            self.assertEqual(calls, list(range(7)))

    def test_existing_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.result(root, 'classifier')
            weight = root / 'full/classifier/best.pt'
            before = weight.read_bytes()
            with (
                patch.object(stages, 'REPO_ROOT', root),
                patch('plane_conditioned_segmentation.training.classifier.train_classifier') as train,
            ):
                stages.execute_stage(root, 'stage1-classifier')
                train.assert_not_called()
            self.assertEqual(weight.read_bytes(), before)

    def test_partial_quality_and_prerequisite_block_before_training(self):
        for kind in ['partial', 'quality', 'prerequisite']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                if kind == 'partial':
                    (root / 'full/classifier').mkdir(parents=True)
                if kind == 'quality':
                    self.result(root, 'classifier')
                    p = root / 'full/classifier/result.json'
                    v = json.loads(p.read_text())
                    v['macro_accuracy'] = 0.5
                    p.write_text(json.dumps(v))
                with (
                    patch.object(stages, 'REPO_ROOT', root),
                    patch('plane_conditioned_segmentation.training.classifier.train_classifier') as train,
                    patch('plane_conditioned_segmentation.training.segmentation.train_seg') as seg,
                ):
                    with self.assertRaises(ValueError):
                        stages.execute_stage(root, 'stage2-experts' if kind == 'prerequisite' else 'stage1-classifier')
                    train.assert_not_called()
                    seg.assert_not_called()

    def test_classifier_nonfinite_and_missing_score_block(self):
        for score in [None, float('nan'), float('inf'), 0.749]:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                self.result(root, 'classifier')
                p = root / 'full/classifier/result.json'
                v = json.loads(p.read_text())
                v['macro_accuracy'] = score
                p.write_text(json.dumps(v))
                self.assertEqual(stages.existing_model(root, 'classifier')['action'], 'blocked_quality_review')

    def test_selected_quality_note_is_data_driven_for_any_model(self):
        for name in ['plain', 'expert_6']:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                self.result(
                    root,
                    name,
                    learning_sufficiency_basis='terminal_training_history',
                    train=dict(macro_dice=0.8, macro_global_dice=0.8, per_target={'0': dict(dice=0.8)}),
                )
                self.assertIsNone(stages.existing_model(root, name)['quality_note'])
                path = root / 'full' / name / 'result.json'
                v = json.loads(path.read_text())
                v['train']['macro_dice'] = 0.5
                path.write_text(json.dumps(v))
                self.assertIn('Selected best remains below', stages.existing_model(root, name)['quality_note'])

    def test_lock_is_exclusive_and_not_removed_on_collision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.result(root, 'classifier')
            folder = root / '.local'
            folder.mkdir()
            lock = folder / 'execution.lock'
            lock.write_text('other process')
            with patch.object(stages, 'REPO_ROOT', root):
                with self.assertRaises(ValueError):
                    stages.execute_stage(root, 'stage1-classifier')
            self.assertEqual(lock.read_text(), 'other process')

    def test_force_unsupported_and_plan_execute_conflict(self):
        for args in [['stage1-classifier', '--force'], ['stage1-classifier', '--plan', '--execute']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
