"""Versioned Phase 3 reanalysis; stored records only, no model or holdout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paper.tables.build_development_tables import summarize, bootstrap
from src.training.answer_parser import strict_parse_answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'paper/tables/phase3_20261024',help='New checkout-local directory; existing evidence is never overwritten')
    out=parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('output must remain inside this checkout')
    if out.exists():raise ValueError('versioned tables already exist; preserve earlier evidence')
    source=ROOT/'artifacts/paired/phase_2c_c/paired_evaluations_v1.jsonl'
    rows=[json.loads(line)for line in source.read_text(encoding='utf-8').splitlines()]
    tags=sorted({r['model_tag']for r in rows});groups={tag:[r for r in rows if r['model_tag']==tag]for tag in tags}
    identities={tag:{r['sample_id']:r for r in group}for tag,group in groups.items()}
    if len(tags)!=2 or set(identities[tags[0]])!=set(identities[tags[1]]):raise ValueError('unpaired records')
    for sid in identities[tags[0]]:
        a,b=[identities[tag][sid]for tag in tags]
        if any(a[k]!=b[k]for k in ('image_sha256','prompt_sha256','ground_truth_label','prompt')):raise ValueError('cross-model input identity mismatch')
    models={}
    for tag,group in groups.items():
        if any(strict_parse_answer(r['raw_completion'])['parsed_label']!=r['parsed_answer']for r in group):raise ValueError('parser convention disagrees')
        result=summarize(group)
        f1=[]
        for label in ('Yes','No','Undetermined'):
            tp=sum(r['ground_truth_label']==r['parsed_answer']==label for r in group)
            denominator=sum(r['ground_truth_label']==label for r in group)+sum(r['parsed_answer']==label for r in group)
            f1.append(2*tp/denominator if denominator else 0)
        result['macro_f1']=sum(f1)/3
        result['all_three_family_variants_correct']=sum(all(r['parsed_answer']==r['ground_truth_label']for r in group if r['family_id']==family)for family in result['family_accuracy'])
        models[tag]=result
    deltas={f:models['qwen3vl2']['family_accuracy'][f]-models['smolvlm2']['family_accuracy'][f]for f in models['qwen3vl2']['family_accuracy']}
    pilot_dir=ROOT/'artifacts/phase3/20261024_v2/pilot'
    execution=json.loads((pilot_dir/'execution.json').read_text())
    scored=[json.loads(line)for line in (pilot_dir/'candidate_scores.jsonl').read_text().splitlines()]
    for row in scored:
        means=row['length_normalized_diagnostic']
        row['length_normalized_argmax']=row['candidate_strings'][max(range(3),key=lambda i:means[i])].strip()
        row['rank_flip']=row['length_normalized_argmax']!=row['argmax_answer']
    result={'historical_models':models,'paired_qwen_minus_smol':{'delta':sum(deltas.values())/len(deltas),'descriptive_ci':bootstrap(deltas)},
            'historical_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'historical_scope':'18 development cases / six families; prior results, not a new replication; parser rechecked uniformly',
            'new_pilot':{'status':execution['status'],'forward_attempts':execution['forward_attempts'],'complete_cross_model':False,
                         'failure':execution.get('error_message'),'records':scored,'optimizer_steps':0,
                         'interpretation':'One family/two cases, only Qwen scored. Not a three-class benchmark, generation test, causal test, calibration or training result.'}}
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (out/'historical_cross_model.csv').open('w',newline='',encoding='utf-8')as stream:
        writer=csv.writer(stream);writer.writerow(['model','cases','families','balanced_accuracy','macro_f1','recall_yes','recall_no','recall_undetermined','exact_format_count','unsupported_fork_rate','over_abstention_rate'])
        for tag,r in models.items():writer.writerow([tag,r['n'],r['families'],r['balanced_accuracy'],r['macro_f1'],*[r['per_class_recall'][c]for c in ('Yes','No','Undetermined')],r['exact_label_output_count'],r['false_definitive_on_underdetermined'],r['over_abstention_on_sufficient']])
    print(json.dumps({'models':{tag:{k:r[k]for k in ('balanced_accuracy','macro_f1','per_class_recall','exact_label_output_count')}for tag,r in models.items()},'paired_difference':result['paired_qwen_minus_smol'],'new_pilot_rank_flips':[r['rank_flip']for r in scored]},indent=2))


if __name__=='__main__':main()
