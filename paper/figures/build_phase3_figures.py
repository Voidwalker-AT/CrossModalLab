"""Phase 3 plots from verified stored evidence; no neural traces or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'paper/figures/phase3_20261024',help='New checkout-local directory; existing evidence is never overwritten')
    out=parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('output must remain inside this checkout')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.research_audit import verify_partial_pilot
    verify_partial_pilot(ROOT/'artifacts/phase3/20261024_v2/pilot')
    if out.exists():raise ValueError('preserve existing versioned figures')
    out.mkdir(parents=True)
    audit_path=ROOT/'artifacts/phase3/20261024_v2/audit.json'
    table_path=ROOT/'paper/tables/phase3_20261024/analysis.json'
    audit=json.loads(audit_path.read_text());data=json.loads(table_path.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    catalog={}
    def save(fig,name,caption,sources):
        fig.text(.08,-.08,caption,fontsize=10,va='top',linespacing=1.5)
        for ext in ('svg','png'):
            p=out/f'{name}.{ext}';fig.savefig(p,dpi=170,bbox_inches='tight')
            catalog[p.name]={'classification':'OBSERVED' if not name.startswith('01') else 'SYMBOLICALLY DERIVED / DATASET DIAGNOSTIC',
                             'caption':caption,'source_sha256':{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest()for s in sources},
                             'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    counts=[audit['text_role_ambiguity']['deterministic_groups'],audit['image_role_ambiguity']['deterministic_groups'],audit['factorial_text_ambiguity']['deterministic_groups'],audit['factorial_image_ambiguity']['deterministic_groups']]
    ax.bar(range(4),counts,color=['#a36d21','#a36d21','#267864','#267864'])
    ax.set(xticks=range(4),xticklabels=['v2 text roles','v2 image roles','New text roles','New image roles'],yticks=[0,1,2],ylim=(0,2),ylabel='Role groups with only one gold label',title='A correct symbolic label does not rule out a shortcut')
    for i,(value,total)in enumerate(zip(counts,[2,2,4,4])):ax.text(i,value+.08,f'{value} of {total} groups',ha='center')
    save(fig,'01_role_shortcuts','Historical scope: 18 validation cases; new factorial: 96 development cases / 12 families.\nOracle-parsed features, not model behavior. Counterbalancing removes deterministic strata.\nBoth single-modality lookup balanced accuracies remain 2/3: partial logical information is not full entailment.',[audit_path])
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,row in zip(axes,data['new_pilot']['records']):
        x=list(range(3));width=.36
        ax.bar([v-width/2 for v in x],row['summed_loglikelihoods'],width,label='Sum of token log-probabilities',color='#267864')
        ax.bar([v+width/2 for v in x],row['length_normalized_diagnostic'],width,label='Mean per token (diagnostic)',color='#a36d21')
        ax.set(xticks=x,xticklabels=['Yes','No','Undetermined'],ylim=(-23,0),ylabel='Sum: nats; mean: nats per token',title=f"Correct label: {row['ground_truth_label']}\nSum chooses {row['argmax_answer']}; mean chooses {row['length_normalized_argmax']}")
    axes[0].legend(loc='lower left',fontsize=9)
    save(fig,'02_candidate_length_sensitivity','Frozen Qwen, one family / two same-image development cases; six forwards total. Higher score is less negative.\nNo generated completion or confidence estimate. Changing the scoring rule flips both choices; it does not establish a better method.\nThe cross-model pilot stopped before Smol loaded; no new Smol result is plotted.',[table_path,ROOT/'artifacts/phase3/20261024_v2/pilot/candidate_scores.jsonl'])
    fig,ax=plt.subplots(figsize=(10,4.7));labels=['Yes','No','Undetermined'];colors=['#267864','#72578d']
    for offset,(tag,color)in enumerate(zip(('smolvlm2','qwen3vl2'),colors)):
        values=[data['historical_models'][tag]['per_class_recall'][c]for c in labels]
        ax.bar([i+(offset-.5)*.36 for i in range(3)],values,.36,label='Historical '+('SmolVLM2' if tag=='smolvlm2' else 'Qwen3-VL'),color=color)
    ax.set(xticks=range(3),xticklabels=labels,ylim=(0,1.15),ylabel='Recall: correct cases / six cases',title='Both models missed the historical insufficient-evidence cases');ax.legend(fontsize=10)
    save(fig,'03_historical_cross_model','Stored Phase 2C-C development evidence, 18 paired cases / six families per model. Higher recall is better.\nQwen BA 55.6%; Smol BA 44.4%. Paired descriptive difference CI: -11.1 to 27.8 percentage points.\nReanalysis of old outputs, not new replication; exact-format compliance differs and model superiority is not established.',[table_path,ROOT/'artifacts/paired/phase_2c_c/paired_evaluations_v1.jsonl'])
    (out/'manifest.json').write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8')
    print('PHASE3_FIGURES',len(catalog),str(out))


if __name__=='__main__':main()
