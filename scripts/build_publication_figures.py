"""New versioned figures/tables from recorded CPU audits and the isolated pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('fresh checkout-local figure directory required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ROOT/'artifacts/publication/20261025/analysis.json'
    data=json.loads(source.read_text());out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    manifest={}
    def save(fig,name,caption):
        fig.text(.08,-.03,caption,fontsize=10,va='top',linespacing=1.5)
        for extension in ('png','svg'):
            path=out/f'{name}.{extension}'
            fig.savefig(path,dpi=180,bbox_inches='tight',metadata={'Date':None} if extension=='svg' else None)
            manifest[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':source.relative_to(ROOT).as_posix(),
                                 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'caption':caption}
        plt.close(fig)
    names=['majority','text_only_oracle_roles','image_only_oracle_roles','role_assignment','prompt_template','entity_identity','position_only','combined_nonsemantic']
    labels=['Majority','Text roles*','Image roles*','Role assignment','Template/order','Entity identity','Position only','Combined nuisance']
    fig,ax=plt.subplots(figsize=(10,5.4))
    values=[data['benchmark']['baselines'][name]['in_sample_balanced_accuracy_ceiling'] for name in names]
    ax.barh(labels[::-1],values[::-1],color=['#287965' if v>.5 else '#acbfb4' for v in values[::-1]])
    ax.axvline(1/3,color='#775b34',linestyle='--',label='Three-class BA chance reference')
    ax.axvline(1,color='#287965',linestyle=':',label='Joint symbolic oracle')
    ax.set(xlim=(0,1.04),xlabel='Balanced-accuracy ceiling on audited feature strata',title='Additional information requires both premises')
    ax.legend(loc='lower right',fontsize=9)
    fig.tight_layout()
    save(fig,'01_controlled_feature_ceilings','*Oracle-parsed roles, not VLM inference. Exact design fractions: single modality 2/3; nuisance 1/3; joint oracle 1.\n9,216 variants / six dependent development families; three fit and three evaluate lookups.\nOnly specified feature classes audited; neither independent confirmation nor exhaustive shortcut removal.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=False)
    for ax,tag in zip(axes,('smol','qwen')):
        model=data['models'][tag]
        margins=[r['summed_loglikelihoods'][1]-r['summed_loglikelihoods'][2] for r in model['cases']]
        ax.bar(['Gold: No','Gold: Undetermined'],margins,color=['#647e6c','#b06d43'])
        ax.axhline(0,color='#526255',linewidth=1)
        for i,value in enumerate(margins):ax.text(i,value+.04,f'{value:.3f}',ha='center')
        ax.set(title='SmolVLM2' if tag=='smol' else 'Qwen3-VL',ylabel='Summed LL(No) − LL(Undetermined)',ylim=(-.5,max(margins)*1.23))
    fig.tight_layout()
    save(fig,'02_matched_pair_margins','Twelve additional forwards; two fixed cases in one exposed development family.\nBoth margins remain positive in both models: the proposed opposed-margin causal screen fails.\nWithin-model score contrasts only; unequal axes and model scales are not cross-model calibrated confidence.')
    table=out/'candidate_scores.csv'
    with table.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model','sample_id','gold','summed_ll_answer','ll_yes','ll_no','ll_undetermined','ll_per_token_answer','revision'])
        for tag in ('smol','qwen'):
            for row in data['models'][tag]['cases']:
                writer.writerow([tag,row['sample_id'],row['ground_truth_label'],row['argmax_answer'],*row['summed_loglikelihoods'],row['length_normalized_answer'],row['revision']])
    manifest[table.name]={'sha256':hashlib.sha256(table.read_bytes()).hexdigest(),'source':source.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'GENERATED_FROM_STORED_EVIDENCE','figures':2,'tables':1,'output':str(out)}))


if __name__=='__main__':main()
