"""Regenerate this package's figures from its frozen JSON/CSV; no inference.

Requires existing matplotlib, numpy and Pillow. Run from any working directory.
"""
import csv,json,hashlib,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'figures'
data=json.loads((BASE/'audit/recomputed.json').read_text(encoding='utf-8'))
cases=json.loads((BASE/'audit/development_cases.json').read_text(encoding='utf-8'))
neural=json.loads((BASE/'audit/neural_evidence_checked.json').read_text(encoding='utf-8'))
palette=['#0072B2','#E69F00','#009E73','#CC79A7','#555555']
conditions=['frozen','prompt','ce5','es5','ce50'];names=['Frozen','Prompt','CE 5','CE+ES 5','CE 50'];labels=['Yes','No','Undetermined']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','savefig.dpi':320})
manifest={}
def save(fig,name,caption,interpretation,limits,sources,classification='HISTORICAL OBSERVATIONS'):
    fig.tight_layout(pad=1.2)
    files={}
    for ext in ['png','svg']:
        path=OUT/f'{name}.{ext}';fig.savefig(path,dpi=320,metadata={'Date':None}if ext=='svg'else None);files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    plotted=[{'bars':[{'x':float(p.get_x()),'height':float(p.get_height()),'width':float(p.get_width())}for p in ax.patches if isinstance(p,matplotlib.patches.Rectangle)],'scatter':[c.get_offsets().tolist()for c in ax.collections if isinstance(c,matplotlib.collections.PathCollection)],'lines':[{'x':list(map(float,l.get_xdata())),'y':list(map(float,l.get_ydata()))}for l in ax.lines],'matrices':[np.asarray(im.get_array()).tolist()for im in ax.images if np.asarray(im.get_array()).ndim==2]}for ax in fig.axes]
    plt.close(fig)
    manifest[name]={'files':files,'width_inches':float(fig.get_figwidth()),'height_inches':float(fig.get_figheight()),'raster_dpi':320,'caption':caption,'interpretation':interpretation,'limitations':limits,'classification':classification,'packaged_source_data':sources,'source_data_sha256':{s:hashlib.sha256((BASE/s).read_bytes()).hexdigest()for s in sources},'original_sources':'See manifest/source_hashes.json and audit/raw_predictions.json; copied research sources in audit/source_records/.','generation_script':'figures/generate_figures.py','plotted_artists':plotted}

fig,ax=plt.subplots(figsize=(7.16,2.5));ax.axis('off')
boxes=[(.02,.57,'Controlled image\n+ text premise'),(.36,.57,'Pinned Qwen VLM\nNF4 + optional LoRA'),(.70,.57,'Recorded outputs\n+ parser / metrics'),(.02,.08,'Dual symbolic oracle\nreachability + orders'),(.36,.08,'Separate historical\nSmol captures / probes'),(.70,.08,'Read-only dashboard\n+ evidence package')]
for x,y,t in boxes:ax.text(x,y,t,transform=ax.transAxes,va='bottom',bbox=dict(boxstyle='round,pad=.7',facecolor='#F1F5F7',edgecolor='#536977'),fontsize=8)
for a,b in [((.27,.68),(.34,.68)),((.61,.68),(.68,.68)),((.83,.55),(.83,.36))]:ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#444444'))
ax.annotate('',xy=(.82,.51),xytext=(.14,.35),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#666666',linestyle='dashed',connectionstyle='arc3,rad=-.1'))
ax.annotate('',xy=(.68,.19),xytext=(.61,.19),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#666666',linestyle='dashed'))
ax.set_title('Research workflow — conceptual system diagram',pad=13)
save(fig,'01_system_architecture','Conceptual CrossModalLab workflow. Stored predictions and symbolic ground truth feed a read-only dashboard; historical Smol representation analysis is a separate evidence stream.','Separates behavior, logical ground truth and representation analysis.','Conceptual engineering diagram, not measured neural pathways or activations.',['audit/development_cases.json','audit/recomputed.json'],'CONCEPTUAL SYSTEM DIAGRAM')

fig,axes=plt.subplots(1,3,figsize=(7.16,2.55))
for ax,title,edges,orders in zip(axes,['Yes: entailed','No: contradicted','Undetermined'],[[(0,1),(1,2)],[(1,0),(2,1)],[(0,1),(2,1)]],['A < B < C','C < B < A','A < C < B or C < A < B']):
    ax.set(xlim=(-.3,2.3),ylim=(-.8,.9));ax.axis('off');ax.set_title(title)
    for i,ch in enumerate('ABC'):ax.text(i,0,ch,ha='center',va='center',bbox=dict(boxstyle='circle',fc='white',ec='#555555'))
    for j,(a,b)in enumerate(edges):ax.annotate('',xy=(b-.14*np.sign(b-a),.02),xytext=(a+.14*np.sign(b-a),.02),arrowprops=dict(arrowstyle='->',color=palette[j],lw=2))
    ax.text(1,-.5,orders,ha='center',fontsize=7);ax.text(1,.6,'Query: is A left of C?',ha='center',fontsize=7)
fig.text(.5,.02,'Blue = visual premise; orange = textual premise. Node positions are a diagram layout.',ha='center',fontsize=7)
save(fig,'02_three_evidence_states','Three strict-left-of premise sets for the same query R(A,C). Chains entail the query or its negation; a fork admits both query orderings.','Undetermined is insufficient evidence, not false.','Symbolic illustration derived from verified logical definitions; not an empirical measurement or neural graph.',['audit/development_cases.json'],'SYMBOLICALLY DERIVED SCHEMATIC')

fig,axes=plt.subplots(1,3,figsize=(7.16,3.0))
for ax,suffix in zip(axes,['entailed','contradicted','underdetermined']):
    c=next(c for c in cases if c['sample_id']=='xc_ho_0011::'+suffix);im=Image.open(OUT/'source_images'/f'xc_ho_0011__{suffix}.png');ax.imshow(im);ax.axis('off');ax.set_title('Gold: '+c['ground_truth_label'])
    pred=next(r for r in json.loads((BASE/'audit/raw_predictions.json').read_text())if r['condition']=='es5'and r['sample_id']==c['sample_id'])
    premise=c['prompt'].split('\n\n')[0];text='\n'.join(textwrap.wrap(premise,29))+f"\nCE+ES 5 raw answer: {pred['raw_completion']}"
    ax.text(.5,-.06,text,transform=ax.transAxes,ha='center',va='top',fontsize=7)
fig.subplots_adjust(bottom=.36);fig.suptitle('Same query: golden disc left of crimson block?',fontsize=9,y=.98)
# Reserve the actual text margin without relying on bbox-tight raster resizing.
save(fig,'03_recorded_failure_examples','Original, hash-verified stimuli for xc_ho_0011 entailed, contradicted and underdetermined development variants. Historical Qwen CE+ES5 generated Yes, No and No, respectively. Text premises are printed under each image.','The fork answer No is unsupported despite being correctly formatted.','One exposed development family, not independent generalization. Images are original controlled benchmark stimuli; literal 2D shapes are disc and wedge.',['audit/raw_predictions.json','audit/development_cases.json'])

metrics=data['conditions'];fig,ax=plt.subplots(figsize=(3.5,2.75));vals=[metrics[k]['balanced_accuracy']for k in conditions]
ax.bar(names,vals,color=palette);ax.set(ylim=(0,1),ylabel='Balanced accuracy',title='Qwen development behavior');ax.tick_params(axis='x',rotation=30)
for i,k in enumerate(conditions):ax.text(i,vals[i]+.025,f"{metrics[k]['correct']}/18",ha='center',fontsize=7)
save(fig,'04_checkpoint_comparison','Recomputed Qwen balanced accuracy on the same 18 development cases/six families. Prompt is frozen Qwen with augmented abstention instructions; CE50 and CE+ES5 have different update budgets.','Descriptive checkpoint differences; CE50 has zero Undetermined recall.','Repeated development inspection; no superiority claim. Paired CE+ES5 minus CE5 interval includes zero; no uncertainty bars are invented here.',['audit/raw_predictions.json','audit/recomputed.json'])

fig,ax=plt.subplots(figsize=(7.16,2.8));x=np.arange(len(conditions));w=.24
for j,label in enumerate(labels):ax.bar(x+(j-1)*w,[metrics[k]['recall'][label]for k in conditions],w,label=label,color=palette[j])
ax.set(xticks=x,xticklabels=names,ylim=(0,1.15),ylabel='Recall (six cases per class)',title='Qwen class-specific recall');ax.legend(loc='upper left',ncol=3)
save(fig,'05_class_recall','Class-specific recall recomputed from raw historical generations; every class has six development cases.','Overall accuracy can hide failure to recognize insufficient evidence.','All classes share six dependent families; no independent sample or causal interpretation.',['audit/raw_predictions.json','audit/recomputed.json'])

fig,axes=plt.subplots(1,5,figsize=(7.16,2.25),sharey=True)
for ax,k,name in zip(axes,conditions,names):
    cm=np.array([[metrics[k]['confusion_matrix'][a][b]for b in [*labels,'Invalid']]for a in labels]);ax.imshow(cm,cmap='Blues',vmin=0,vmax=6);ax.set_title(name,fontsize=8);ax.set(xticks=range(4),xticklabels=['Y','N','U','Inv'],yticks=range(3),yticklabels=['Yes','No','U']);ax.set_xlabel('Predicted',fontsize=7)
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',fontsize=7,color='white'if v>=4 else '#222222')
axes[0].set_ylabel('Gold');save(fig,'06_confusion_matrices','Qwen confusion counts: rows are gold classes, columns are parsed predictions including Invalid. Each row sums to six; color scale is fixed at 0–6.','Invalid outputs remain errors; CE50 parses all outputs but misses every fork.','Color represents counts, not confidence. These are prefix-parser outcomes, not exact-format acceptance.',['audit/raw_predictions.json','audit/recomputed.json'])

fig,ax=plt.subplots(figsize=(3.5,2.8));x=np.arange(5)
ax.bar(x-.18,[metrics[k]['correct']/18 for k in conditions],.36,label='Semantic accuracy',color=palette[0]);ax.bar(x+.18,[metrics[k]['exact_format']/18 for k in conditions],.36,label='Exact-label compliance',color=palette[1]);ax.set(xticks=x,xticklabels=names,ylim=(0,1.05),ylabel='Fraction of 18 cases');ax.tick_params(axis='x',rotation=30);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),fontsize=6.7)
save(fig,'07_format_vs_correctness','Semantic accuracy and exact-label compliance are recomputed separately. CE50 is 12/18 correct under the historical parser, 0/18 semantic-invalid and 0/18 exact-label compliant.','Malformed outputs and incorrect answers are distinct failures.','Exact-label check trims outer whitespace only for classification; raw strings remain unchanged in the evidence.',['audit/raw_predictions.json','audit/recomputed.json'])

training=data['training'];steps=[r['step']for r in training];fig,ax=plt.subplots(figsize=(3.5,2.65))
for j,(field,name)in enumerate([('ce_loss','CE'),('es_loss','ES'),('total_loss','Total')]):ax.scatter(steps,[r[field]for r in training],color=palette[j],label=name,marker=['o','s','^'][j],s=25)
ax.set(xticks=steps,xlabel='Recorded optimizer update',ylabel='Loss (natural-log units)',title='Five observations, different examples',ylim=(-.5,22));ax.legend(ncol=3,loc='upper center',fontsize=7)
save(fig,'08_five_step_losses','Exact observed CE, ES and CE+0.25 ES at five updates. No lines, smoothing or interpolated points are added.','Each example supplies a different loss; this is a feasibility record, not a fixed-input learning curve.','Five selected updates cannot establish optimization convergence, method benefit or generalization.',['audit/recomputed.json','tables/f_five_step_telemetry.csv'])

fig,ax=plt.subplots(figsize=(3.5,2.8))
for j,(field,label)in enumerate([('torch_max_allocated_mib','Peak allocated'),('torch_max_reserved_mib','Peak reserved')]):ax.scatter(steps,[r['memory'][field]for r in training],label=label,color=palette[j],marker=['o','s'][j],s=24)
ax.set(xticks=steps,xlabel='Recorded optimizer update',ylabel='CUDA allocator memory (MiB)',ylim=(0,4096));ax.legend(loc='lower right');ax.set_title('Historical Qwen CE+ES telemetry')
save(fig,'09_gpu_memory','Per-update allocator maxima during the historical five-update pilot. Dedicated device capacity is 4096 MiB; allocator usage is distinct from sampled NVIDIA physical usage.','Shows bounded observed allocator demand, not present admission or guaranteed safe future capacity.','Sampled physical memory peaks can miss transient allocations. This mission made zero new model forwards and passed no replication RAM admission.',['audit/recomputed.json','tables/f_five_step_telemetry.csv'])

probe=data['probe'];fig,ax=plt.subplots(figsize=(3.5,3.0));x=[p['layer']for p in probe];y=[p['balanced_accuracy']for p in probe]
ax.errorbar(x,y,yerr=[[p['balanced_accuracy']-p['ci']['ci_low']for p in probe],[p['ci']['ci_high']-p['balanced_accuracy']for p in probe]],fmt='o',capsize=3,color=palette[0]);ax.axhline(1/3,color='#666666',ls='--',label='Three-class BA reference');ax.set(xticks=x,ylim=(0,1),xlabel='Recorded decoder layer (last token)',ylabel='Probe balanced accuracy');ax.set_title('HISTORICAL SMOLVLM2\nCORRELATIONAL — NOT CAUSAL',fontsize=8.5);ax.legend(loc='lower right',fontsize=6.5)
save(fig,'10_historical_smol_probes','Historical SmolVLM2 last-token probe BA: layers 3/12/18/23 = 0.333333/0.518519/0.777778/0.740741. External probes used 81 training cases/27 families and 27 validation cases/nine families. Bars retain stored descriptive B=2000 family intervals.','Label accessibility to an external probe, not evidence the VLM causally used that information.','Layer/selector/L2 selected on development data. BA independently checked from confusion counts; stored CI endpoints were not independently resampled in this mission. Separate from Qwen.',['audit/recomputed.json','audit/neural_evidence_checked.json'])

capture=neural['captures'][0];points=[p for p in capture['points']if p['selector']=='last'];fig,ax=plt.subplots(figsize=(3.5,2.65));ax.scatter([p['layer']for p in points],[p['l2_norm']for p in points],color=palette[2]);ax.set(xticks=[3,12,18,23],xlabel='Recorded decoder layer',ylabel='Residual-row L2 (activation units)',title='Historical SmolVLM2 — last token')
save(fig,'11_historical_activation_norms','Stored residual-row L2 norms for '+capture['sample_id']+'; four sampled layers, last-token selector, 2048 coordinates per row. Norms were rechecked against saved vectors.','Magnitude of a vector in a recorded representation, not confidence, evidence flow or neuron importance.','One historical capture; separate Smol model, no Qwen attribution, no intervening or unmeasured layers.',['audit/neural_evidence_checked.json'])

shortcut=data['shortcut_audit'];keys=['majority','text_only_oracle_roles','image_only_oracle_roles','role_assignment','prompt_template','entity_identity','position_only','combined_nonsemantic'];titles=['Majority','Text roles*','Image roles*','Role assignment','Prompt/order','Entity identity','Position only','Combined nuisance']
fig,ax=plt.subplots(figsize=(3.5,3.15));v=[shortcut['baselines'][k]['in_sample_balanced_accuracy_ceiling']for k in keys];ax.barh(titles[::-1],v[::-1],color=palette[0]);ax.axvline(1/3,color='#555555',ls='--');ax.set(xlim=(0,1),xlabel='Audited feature-stratum BA ceiling',title='CPU dataset information diagnostic')
save(fig,'12_shortcut_diagnostic','CPU feature-stratum balanced-accuracy ceilings on 9216 variants/six dependent families: audited nuisance features 1/3, single-modality oracle-parsed roles 2/3; joint symbolic oracle 1. Asterisks identify oracle-parsed role features.','Logical non-entailment by one premise does not imply statistical label unpredictability.','Only listed features audited. These are design diagnostics, not VLM scores, exhaustive shortcut clearance or independent samples.',['audit/recomputed.json'])

(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (OUT/'figure_source_summary.csv').open('w',newline='',encoding='utf-8')as f:
    writer=csv.writer(f);writer.writerow(['figure','classification','data','caption','interpretation','limitations'])
    for k,v in manifest.items():writer.writerow([k,v['classification'],';'.join(v['packaged_source_data']),v['caption'],v['interpretation'],v['limitations']])
print(json.dumps({'figures':len(manifest),'formats':['PNG 320 DPI','SVG vector'],'status':'GENERATED_FROM_FROZEN_PACKAGE_DATA'}))
