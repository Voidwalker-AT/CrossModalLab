"""Select paper-support assets into a fresh directory; never changes the release."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dual_model_evidence import OLD, load, MODELS
from scripts.build_overleaf_evidence import tex

SELECTED_FIGURES = ['02_three_evidence_states', '01_system_architecture', '03_recorded_failure_examples',
                    '04_checkpoint_comparison', '05_class_recall', '07_format_vs_correctness', '10_historical_smol_probes']
SELECTED_KEYS = ['W1', 'W2', 'W4', 'W5', 'W7', 'W8', 'W14', 'W15', 'W18', 'W19', 'LoRA', 'QwenCard', 'SmolCard']


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(text.encode('utf-8'))


def table(out, name, headers, rows, caption, source, *, wide=False):
    with (out / 'tables' / (name + '.csv')).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    md = '# ' + caption + '\n\nSource: ' + source + '\n\n| ' + ' | '.join(headers) + ' |\n|'
    md += '---|' * len(headers) + '\n'
    md += ''.join('| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in rows)
    put(out / 'tables' / (name + '.md'), md)
    env = 'table*' if wide else 'table'
    spec = 'l' + ('X' * (len(headers) - 1))
    content = ('% Source: ' + source + '\n' + '\\begin{' + env + '}[t]\n\\centering\\small\n'
               '\\caption{' + tex(caption) + '}\n\\label{tab:' + name + '}\n'
               '\\begin{tabularx}{\\linewidth}{' + spec + '}\n\\toprule\n')
    content += ' & '.join(tex(h) for h in headers) + r' \\ \midrule' + '\n'
    content += '\n'.join(' & '.join(tex(v) for v in row) + r' \\' for row in rows)
    content += '\n\\bottomrule\n\\end{tabularx}\n\\end{' + env + '}\n'
    put(out / 'tables' / (name + '.tex'), content)


def build(out, audit_path):
    if out.exists() or not out.resolve().is_relative_to(ROOT):
        raise ValueError('fresh selection directory within workspace required')
    data = load(audit_path)
    for folder in ('figures', 'tables', 'equations', 'references', 'audit'):
        (out / folder).mkdir(parents=True)
    shutil.copyfile(audit_path, out / 'audit/dual_model_audit.json')
    figure_manifest = load(OLD / 'figures/figure_manifest.json')
    chosen, includes = {}, []
    for index, name in enumerate(SELECTED_FIGURES, 1):
        info = figure_manifest[name]
        wide = info['width_inches'] > 4
        for suffix in ('png', 'svg'):
            shutil.copyfile(OLD / 'figures' / (name + '.' + suffix), out / 'figures' / (name + '.' + suffix))
        shutil.copyfile(OLD / 'figures' / (name + '_source.json'), out / 'figures' / (name + '_source.json'))
        chosen[name] = dict(info, paper_order=index, ieee_environment='figure*' if wide else 'figure',
                           intended_width_inches=7.16 if wide else 3.5)
        env = 'figure*' if wide else 'figure'
        includes.append('\\begin{' + env + '}[t]\n\\centering\n'
            '\\includegraphics[width=\\linewidth]{figures/' + name + '.png}\n'
            '\\caption{' + tex(info['caption'] + ' ' + info['limitations']) + '}\n'
            '\\label{fig:' + name + '}\n\\end{' + env + '}\n')
    put(out / 'figures/selection_manifest.json', json.dumps(chosen, indent=2) + '\n')
    put(out / 'figures/figure_includes.tex', '\n'.join(includes))
    # Preserve only original stimuli actually used in the selected failure figure.
    for kind in ('entailed', 'contradicted', 'underdetermined'):
        source = OLD / 'figures/source_images' / ('xc_ho_0011__' + kind + '.png')
        target = out / 'figures/source_images' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(OLD / 'equations/equations_verified.tex', out / 'equations/equations_verified.tex')
    shutil.copyfile(OLD / 'equations/equations_explained.md', out / 'equations/equations_explained.md')

    q = data['qwen']['metrics']
    condition_rows = [(c, 'Qwen', step, prompt, '18 / 6') for c, step, prompt in
                      [('frozen', 0, 'standard'), ('prompt', 0, 'augmented'), ('ce5', 5, 'standard'),
                       ('es5', 5, 'standard'), ('ce50', 50, 'standard'), ('es5prompt', 5, 'augmented')]]
    condition_rows.append(('Smol 0D2', 'SmolVLM2', 0, 'separate legacy battery', '36 / 30 groups'))
    table(out, 't1_evaluation_conditions', ['Condition', 'Model', 'Updates', 'Prompt', 'Cases / groups'], condition_rows,
          'Distinct historical evaluation conditions; rows from different batteries are not a leaderboard.',
          'audit/dual_model_audit.json#qwen; #smol; audit/model_inventory.md', wide=True)
    table(out, 't2_dataset_independence', ['Evidence set', 'Records', 'Grouping', 'Interpretation'], [
        ('Qwen generation', 18, '6 families', 'Repeated development; six conditions'),
        ('Smol 0D2', 36, '30 matched groups', 'Includes six pairs and entity reuse'),
        ('Smol Phase 1F', 108, '27 train / 9 validation families', '81 / 27 rows; development-selected probes'),
        ('Factorial variants', 9216, '6 families; 72 images', 'Design diagnostics, no new VLM evaluation'),
        ('Qwen new scores', 3, '1 family', '9 forwards; candidate scoring only')],
        'Sample counts do not establish statistical independence.', 'audit/dual_model_audit.json; original tables/b_dataset_units.csv', wide=True)
    table(out, 't3_qwen_behavior', ['Condition', 'Correct', 'BA', 'Macro F1', 'U recall'],
          [(c, f"{v['correct']}/18", f"{v['balanced_accuracy']:.6f}", f"{v['macro_f1']:.6f}", f"{v['tp']['Undetermined']}/6") for c, v in q.items()],
          'Historical Qwen generations on six development families; descriptive, not a superiority test.',
          'audit/dual_model_audit.json#qwen.metrics; original audit/raw_predictions.json', wide=True)
    table(out, 't4_format_compliance', ['Condition', 'Correct', 'Invalid', 'Exact'],
          [(c, f"{v['correct']}/18", f"{v['semantic_invalid']}/18", f"{v['exact_format']}/18") for c, v in q.items()],
          'Prefix-parser semantic correctness, invalid labels and exact-label formatting are separate endpoints.',
          'audit/dual_model_audit.json#qwen.metrics')
    table(out, 't5_training_configuration', ['Setting', 'Recorded value'], [
        ('Base', 'Qwen3-VL-2B, NF4 double quantization'), ('LoRA', 'rank 2; alpha 4; q/k/v/o projections'),
        ('Trainable', '802816 parameters; 224 tensors'), ('Optimizer', 'AdamW; learning rate 0.0001; weight decay 0.01'),
        ('AdamW settings', 'betas 0.9 / 0.999; epsilon 1e-8'), ('Batch / accumulation', '1 / 1'),
        ('Compute', 'BF16; adapter dtype separately recorded'), ('Budget', 'CE 50 updates; CE+ES paused at 5'),
        ('ES settings', 'margin 1.0; coefficient 0.25')],
        'Executed historical adaptation configuration; update budgets differ and no method superiority is established.',
        'audit/dual_model_audit.json#qwen.training_identity; original tables/e_training_config.csv')
    table(out, 't6_smol_measurements', ['Measurement', 'Observed', 'Evidence limit'], [
        ('0D2 behavior', '18/36 correct; BA 0.425926', '30 matched groups; different battery from Qwen'),
        ('Class recall Yes / No / U', '10/12; 8/18; 0/6', 'Not general model accuracy'),
        ('Passive captures', '10 captures; 200 residual rows', '4 layers; 2048 coordinates; no fresh generation'),
        ('Phase 1F probes', '81 train; 27 validation rows', '27 / 9 family split; 13 specs including PCA'),
        ('L18 last-token BA', '0.777778; stored CI [0.592593, 0.962963]', 'B=2000 family resamples; selection; correlational')],
        'SmolVLM2 contributes behavioral and representation evidence, separately from Qwen fine-tuning.',
        'audit/dual_model_audit.json#smol; #neural', wide=True)
    table(out, 's1_qwen_candidate_scores', ['Variant', 'Gold', 'Yes LL', 'No LL', 'U LL', 'Argmax'],
          [(r['sample_id'].split('::')[1], r['ground_truth_label'], *[f'{x:.9f}' for x in r['summed_loglikelihoods']], r['argmax_answer'])
           for r in data['qwen_replication']['rows']],
          'Optional supplement: frozen Qwen candidate likelihoods for xc_ho_0011, one development family. Yes versus U gap is 0.000009536743; not calibrated confidence or generation accuracy.',
          'audit/dual_model_audit.json#qwen_replication; artifacts/publication/20261025/qwen_replication_20261025_464531de/', wide=True)
    pilot = data['matched_pilot']['rows']
    table(out, 's2_matched_pilot', ['Case', 'Gold', 'Smol preference', 'Qwen preference'],
          [(r['sample_id'].split('::')[1], r['ground_truth_label'], r['argmax_answer'], pilot['qwen'][i]['argmax_answer'])
           for i, r in enumerate(pilot['smol'])],
          'Optional supplement: matched two-case candidate-scoring pilot, p3dev2_2026102403_000. One family; no model ranking. Model-native templates differ; task inputs and summed scoring match.',
          'audit/dual_model_audit.json#matched_pilot', wide=True)
    put(out / 'audit/smol_raw_behavior.json', json.dumps(data['smol']['rows'], indent=2) + '\n')
    put(out / 'audit/figure_table_selection.md', '# Final asset selection\n\nSeven main-paper figures; six compact main tables. Two optional supplementary tables. Do not shrink double-column assets into a single column.\n\n'
        + '\n'.join(f"- Figure {i+1}: `figures/{n}.png` (SVG also supplied); {chosen[n]['ieee_environment']}, width {chosen[n]['intended_width_inches']} inches; sources and captions in `figures/selection_manifest.json`." for i, n in enumerate(SELECTED_FIGURES))
        + '\n\nExact table filenames: `tables/t1_evaluation_conditions.tex`, `t2_dataset_independence.tex`, `t3_qwen_behavior.tex`, `t4_format_compliance.tex`, `t5_training_configuration.tex`, `t6_smol_measurements.tex`; optional `s1_qwen_candidate_scores.tex` and `s2_matched_pilot.tex`. CSV and Markdown counterparts preserve numeric values.\n\nOmit activation-norm plot (no distinct mechanistic result), GPU plot (engineering detail), five-step loss plot (not a learning curve), redundant confusion panels and shortcut-ceiling plot (appendix discussion suffices). Keep every historical original intact. The probe figure keeps stored intervals; endpoints are not freshly re-estimated or corrected for selecting the best layer.\n')
    bib = (OLD / 'references/references_verified.bib').read_text(encoding='utf-8')
    entries = {m.group(1): m.group(0) for m in re.finditer(r'(?ms)^@\w+\{([^,]+),.*?(?=^@|\Z)', bib)}
    selected = '\n'.join(entries[k].strip() for k in SELECTED_KEYS) + '\n\n'
    selected += '''@inproceedings{ProbeControls,
  author = {John Hewitt and Percy Liang},
  title = {Designing and Interpreting Probes with Control Tasks},
  booktitle = {Proceedings of EMNLP-IJCNLP},
  year = {2019},
  pages = {2733--2743},
  doi = {10.18653/v1/D19-1275},
  url = {https://aclanthology.org/D19-1275/},
  note = {Official metadata and abstract checked 2026-10-25; not a VLM experiment}
}
'''
    put(out / 'references/references_selected.bib', selected)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.audit)
    print(json.dumps(dict(status='SELECTED', figures=7, main_tables=6, supplementary_tables=2, references=14)))
