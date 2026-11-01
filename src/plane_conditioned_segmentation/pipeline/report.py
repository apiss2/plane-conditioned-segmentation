"""Generate private run reports from saved artifacts, without embedded study outcomes."""

from ..common import NUM_TARGETS
from ..common import read_json as read
from ..data.dataset import ROOT

MODEL_NAMES = ['classifier', *[f'expert_{i}' for i in range(NUM_TARGETS)], 'plain', 'film', 'condconv']


def _score_lines(summary):
    lines = ['| Model/condition | Target macro Dice | Target image Dice | Global macro Dice |', '|---|---:|---:|---:|']
    for name, r in summary['scores'].items():
        lines.append(f'| {name} | {r["macro_dice"]:.6f} | {r["dice"]:.6f} | {r["macro_global_dice"]:.6f} |')
    return lines


def _quality_lines():
    lines = []
    for name in MODEL_NAMES:
        r = read(ROOT / 'full' / name / 'result.json')
        lines.append(
            f'- {name}: best epoch {r.get("best_epoch")}; '
            f'gate basis {r.get("learning_sufficiency_basis", "classifier macro recall")}; '
            f'gate pass {r.get("learning_sufficiency_pass", "see classifier score")}.'
        )
    return lines


def main():
    out = ROOT / 'final_evaluation'
    s = read(out / 'summary.json')
    plan = read(ROOT / 'study_plan.json')
    cohort = read(ROOT / 'cohort.json')
    lines = [
        '# Private run results',
        '',
        'This report and all execution results stay in the ignored run directory.',
        '',
        '## Data and protocol',
        '',
        f'Segmentation counts: {plan["segmentation_counts"]}; classification counts: {plan["classification_counts"]}',
        f'Cohort policy: {cohort.get("policy", "See retained cohort manifest and audit history")}. '
        'Patient independence is not established by these scores.',
        'Encoder, normalization, acquisition license and patient/group provenance require independent verification. '
        'Test is not used for model or threshold selection.',
        '',
        *_score_lines(s),
        '',
        f'Classifier accuracy {s["classifier_accuracy"]:.6f}; macro recall {s["classifier_macro_accuracy"]:.6f}.',
        '',
        '## Quality assessment',
        '',
        *_quality_lines(),
        '',
        'Terminal-history sufficiency and selected-best quality are distinct. A terminal pass does not establish that '
        'the selected checkpoint meets the original train/global gate. Retain any selected-checkpoint failure and '
        'protocol-change caveats. No clinical generalization or patient independence is established.',
        'Detailed per-image metrics, common-original-grid scores, original reports and timing conditions remain '
        'private within this run.',
    ]
    path = out / 'REPORT.md'
    assert not path.exists()
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(path, flush=True)


if __name__ == '__main__':
    main()
