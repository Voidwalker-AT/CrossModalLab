import {useState} from 'react';
import type {Case,Condition} from './main';
import {RawOutput, statusName} from './explanations';

type Point={layer:number;selector:string;token_index:number|null;l2_norm:number;shape:number[];dtype:string};
type Capture={sample_id:string;role:string;historical_correct:boolean;historical_parsed_answer:string;ground_truth:string;condition:string;image_sha256:string;matched_group_id:string|null;source:string;sha256:string;points:Point[]};
export type NeuralEvidence={available:boolean;model:{id:string;revision:string;checkpoint:string};captures:Capture[];probe:{points:{layer:number;selector:string;balanced_accuracy:number;ci:{ci_low:number;ci_high:number}}[];source:string;validation_families:number;family_ids:string[];limitation:string};causal_audit:{source:string;records:number;interventions:number;missing_planned:number;targeted_abs_delta:number;unrelated_token_abs_delta:number;limitation:string};alignment:string;missing:string[];inventory:{source:string;validation_status:string;classification:string}[];validation:{captures:number;activation_rows:number;feature_files:number;probe_points:number}};

function Trajectory({series,unit,maximum}:{series:{name:string;color:string;points:{layer:number;value:number;low?:number;high?:number}[]}[];unit:string;maximum?:number}){
 const ceiling=maximum??Math.max(1,...series.flatMap(s=>s.points.map(p=>p.value)))*1.12;
 const x=(layer:number)=>65+(layer-3)/20*505,y=(value:number)=>240-value/ceiling*185;
 return <svg viewBox="0 0 640 310" className="neural-plot" role="img" aria-label={`${unit} at recorded layers 3, 12, 18 and 23`}>
  {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="65" x2="580" y1={y(t*ceiling)} y2={y(t*ceiling)} stroke="#dce5d8"/><text x="54" y={y(t*ceiling)+4} textAnchor="end">{(t*ceiling).toFixed(ceiling>2?0:2)}</text></g>)}
  {[3,12,18,23].map(l=><text key={l} x={x(l)} y="264" textAnchor="middle">{l}</text>)}
  <text x="65" y="30">{unit}</text><text x="320" y="288" textAnchor="middle">Recorded decoder layer index</text>
  {series.map(s=><g key={s.name}><polyline fill="none" stroke={s.color} strokeWidth="3" points={s.points.map(p=>`${x(p.layer)},${y(p.value)}`).join(' ')}/>{s.points.map(p=><g key={p.layer}>
   {p.low!==undefined&&p.high!==undefined&&<path d={`M${x(p.layer)},${y(p.low)}V${y(p.high)}M${x(p.layer)-5},${y(p.low)}h10M${x(p.layer)-5},${y(p.high)}h10`} stroke={s.color}/>}
   <circle tabIndex={0} cx={x(p.layer)} cy={y(p.value)} r="5" fill={s.color}><title>{s.name}: layer {p.layer}, {unit} {p.value}{p.low!==undefined?`; descriptive 95% family CI [${p.low}, ${p.high}]`:''}</title></circle>
  </g>)}</g>)}
 </svg>
}

export function NeuralDiagnostics({data}:{data:NeuralEvidence}){
 const [sample,setSample]=useState(''),[selector,setSelector]=useState('last'),[probeSelector,setProbeSelector]=useState('last');
 if(!data.available)return <section className="panel"><h2>Neural Diagnostics</h2><span className="tag amber">NOT MEASURED</span><p>No curated neural bundle is available in this checkout.</p></section>;
 const item=data.captures.find(c=>c.sample_id===sample)??data.captures[0];
 const paired=item.matched_group_id?data.captures.find(c=>c.sample_id!==item.sample_id&&c.matched_group_id===item.matched_group_id&&c.image_sha256===item.image_sha256&&c.condition===item.condition):undefined;
 const current=item.points.filter(p=>p.selector===selector).sort((a,b)=>a.layer-b.layer);
 const series=[{name:item.sample_id,color:'#267864',points:current.map(p=>({layer:p.layer,value:p.l2_norm}))},...(paired?[{name:paired.sample_id,color:'#a36d21',points:paired.points.filter(p=>p.selector===selector).sort((a,b)=>a.layer-b.layer).map(p=>({layer:p.layer,value:p.l2_norm}))}]:[])];
 const probes=data.probe.points.filter(p=>p.selector===probeSelector).sort((a,b)=>a.layer-b.layer);
 return <section className="neural-diagnostics" aria-label="Recorded neural diagnostics">
  <div className="panel"><span className="eyebrow">HISTORICAL NEURAL EVIDENCE · SEPARATE MODEL / EXPERIMENT</span><div className="panel-heading"><h2>Neural Diagnostics · SmolVLM2</h2><span className="tag amber">CORRELATED — NOT CAUSAL</span></div>
   <p>These are existing measured SmolVLM2 captures and probe results. They are <b>not the selected Qwen checkpoint's activations</b> and do not explain a particular Qwen error.</p>
   <details><summary>Verify historical model identity and capture inventory</summary><code className="hash-value">{data.model.id} @ {data.model.revision}</code><p className="caption">{data.model.checkpoint} · {data.validation.captures} captures / {data.validation.activation_rows} sparse rows · {data.validation.feature_files} feature files validated on CPU · no new model run.</p></details>
  </div>
  <div className="neural-panels"><div className="panel"><h3>How large are the stored internal representations?</h3><p>An activation norm is the length of a vector of internal numbers. Higher means a larger magnitude, not a better answer. The horizontal axis shows four sampled Transformer layers; the vertical axis uses activation units, not probabilities.</p>
   <label className="neural-control">Historical capture<select aria-label="Neural capture" value={item.sample_id} onChange={e=>setSample(e.target.value)}>{data.captures.map(c=><option key={c.sample_id}>{c.sample_id}</option>)}</select></label>
   <label className="neural-control">Token selector<select aria-label="Activation token selector" value={selector} onChange={e=>setSelector(e.target.value)}>{['last','first_image','last_image','mean_image','last_text_non_image'].map(s=><option key={s}>{s}</option>)}</select></label>
   <p>The selector chooses which stored row to read: last = final token; first/last image = boundary image tokens; mean image = average across image tokens; last text non image = final non-image text token. Each point is one recorded layer, not a neuron.</p><Trajectory series={series} unit="Residual-row L2 norm (activation units)"/>
   <div className="neural-legend"><span>● Green: selected historical example</span>{paired&&<span>● Gold: matched logical variant, same image</span>}</div>
   <p className="caption">Phase 0D2 recorded behavior: {item.historical_parsed_answer}; gold {item.ground_truth}. {item.role}. The answer was recorded earlier, not generated during this passive capture.</p>
   <p><b>What we observed:</b> For this selector, the norm is {current[0].l2_norm.toFixed(2)} at layer {current[0].layer} and {current[current.length-1].l2_norm.toFixed(2)} at layer {current[current.length-1].layer}. Only these four layers were sampled. {paired?'The two lines compare related inputs; text can differ, so this does not isolate a cause of correctness.':'A change in magnitude does not identify a causal failure layer.'}</p><details><summary>Verify the four measured vectors</summary><table className="data-table"><thead><tr><th>Layer</th><th>Token index</th><th>L2 norm</th><th>Shape / dtype</th></tr></thead><tbody>{current.map(p=><tr key={p.layer}><td>{p.layer}</td><td>{p.token_index??'Image-token aggregate'}</td><td>{p.l2_norm.toFixed(6)}</td><td>[{p.shape.join(', ')}] / {p.dtype}</td></tr>)}</tbody></table></details>
   <p className="caption">{paired?data.alignment:'No validated clean/failure partner for this capture. A comparison is not invented.'}</p><p className="caption">Norm growth is not evidence of improved reasoning, information flow or a causal failure layer.</p>
   <details><summary>Capture source and SHA-256</summary><code className="hash-value">{item.source}</code><code className="hash-value">{item.sha256}</code>{paired&&<code className="hash-value">Paired source: {paired.source} · {paired.sha256}</code>}</details>
  </div><div className="panel"><h3>Can an external classifier read the evidence state?</h3><p>A probe is a separately trained classifier that reads stored representations. Balanced accuracy averages recall for Yes, No and Undetermined; higher is better for this classifier. Purple points show four separate probes; vertical bars show uncertainty from resampling families. They do not track the model’s own answer.</p>
   <label className="neural-control">Probe token selector<select aria-label="Probe token selector" value={probeSelector} onChange={e=>setProbeSelector(e.target.value)}>{['last','last_text_non_image','mean_image'].map(s=><option key={s}>{s}</option>)}</select></label>
   <Trajectory unit="Probe balanced accuracy (fraction)" maximum={1} series={[{name:'Stored validation probe',color:'#72578d',points:probes.map(p=>({layer:p.layer,value:p.balanced_accuracy,low:p.ci.ci_low,high:p.ci.ci_high}))}]}/>
   <p>81 training cases / 27 families; 27 validation cases / {data.probe.validation_families} families. Error bars: stored 2,000-resample descriptive family-bootstrap intervals.</p><p className="caption">{data.probe.limitation} Each point is a separately fitted probe, not a neural answer trajectory.</p>
   <p><b>What we observed:</b> The selected probe scores {(100*probes[0].balanced_accuracy).toFixed(1)}% at layer {probes[0].layer} and {(100*probes[probes.length-1].balanced_accuracy).toFixed(1)}% at layer {probes[probes.length-1].layer}. Decodability shows accessible information; it does not prove the model uses it.</p><details><summary>Verify probe values and intervals</summary><table className="data-table"><thead><tr><th>Layer</th><th>BA</th><th>95% descriptive CI</th></tr></thead><tbody>{probes.map(p=><tr key={p.layer}><td>{p.layer}</td><td>{p.balanced_accuracy.toFixed(6)}</td><td>[{p.ci.ci_low.toFixed(6)}, {p.ci.ci_high.toFixed(6)}]</td></tr>)}</tbody></table></details>
   <details><summary>Probe source and validation families</summary><code className="hash-value">{data.probe.source}</code><p className="caption">{data.probe.family_ids.join(', ')}</p></details>
  </div></div>
  <div className="panel"><h3>Causal evidence audit</h3><span className="tag rose">UNVERIFIED · no relation-specific causal mechanism</span><p>{data.causal_audit.records} surviving rows / {data.causal_audit.interventions} interventions; {data.causal_audit.missing_planned} planned records missing.</p><p>Targeted mean |Δ margin|: {data.causal_audit.targeted_abs_delta.toFixed(6)}. Unrelated-token control: {data.causal_audit.unrelated_token_abs_delta.toFixed(6)}.</p><p className="caption">{data.causal_audit.limitation}</p><code className="hash-value">{data.causal_audit.source}</code><h3>NOT MEASURED / not validated for this dashboard</h3><ul className="facts">{data.missing.map(s=><li key={s}>{s}</li>)}</ul>
   <details><summary>Neural evidence inventory</summary>{data.inventory.map(s=><div key={s.source}><code className="hash-value">{s.source}</code><p className="caption">{s.classification} · {s.validation_status}</p></div>)}</details>
  </div>
 </section>
}

export function CheckpointComparison({item,conditions}:{item:Case;conditions:Condition[]}){
 const selected=['frozen','ce5','ce50','es5'].map(id=>conditions.find(c=>c.id===id)!);
 const correct=selected.filter(c=>item.predictions[c.id].correct).length;
 return <section className="panel checkpoint-comparison"><div className="panel-heading"><h3>Did fine-tuning change this answer?</h3><a className="button" href={`/api/comparison/${encodeURIComponent(item.sample_id)}.svg`} download={`${item.sample_id.replaceAll('::','__')}_comparison.svg`}>↓ Export comparison SVG</a></div>
  <p className="comparison-takeaway"><b>{correct===0?'All four checkpoints answered this case incorrectly.':correct===4?'All four checkpoints gave the correct extracted answer.':`${correct} of the four checkpoints gave the correct extracted answer.`}</b> Correctness and output format are separate: a readable label can still be unsupported, and a correct extracted label can have a malformed suffix.</p>
  <div className="table-scroll"><table className="data-table checkpoint-table"><thead><tr><th>Checkpoint and training</th><th>Extracted answer</th><th>Correctness</th><th>Output format</th></tr></thead><tbody>{selected.map(c=>{const p=item.predictions[c.id];return <tr key={c.id}><td><b>{c.name}</b><small>{c.id==='frozen'?'Before our fine-tuning':c.id==='es5'?'Exploratory evidence-sufficiency training':'Standard supervised training'}</small></td><td><strong>{p.parsed_answer}</strong></td><td>{p.correct?'Correct':'Incorrect'}</td><td>{p.format_compliant?'Exact format':'Malformed generation'}</td></tr>})}</tbody></table></div>
  <p>CE means standard cross-entropy training. CE+ES adds an experimental evidence-sufficiency objective. Step 5 or 50 counts optimizer updates, not epochs or model versions. All four use the same image and standard task prompt. CE+ES has no step-50 result.</p>
  <details><summary>Explore exact answers from all four checkpoints</summary><div className="checkpoint-cards">{selected.map(c=><article key={c.id}><h4>{c.name}</h4><RawOutput prediction={item.predictions[c.id]} source={c.source_artifact}/></article>)}</div></details>
  <div className="behavior-boundary"><b>Internal neural measurements have not yet been recorded for this example.</b><p>Architecture Explorer offers historical SmolVLM2 measurements as a separate exploration. They do not explain this Qwen answer. Perception, composition and calibration remain competing, unverified explanations.</p></div>
 </section>
}

export function EvidenceGallery({cases,condition,onSelect}:{cases:Case[];condition:string;onSelect:(id:string)=>void}){
 return <section className="panel evidence-gallery"><h2>Scientific evidence gallery</h2><p className="caption">All 18 original development stimuli. Select a case to inspect its input, logical evidence and recorded answers. Images are hash-verified; no tutorial data are mixed in.</p><div className="gallery-grid">{cases.map(c=><article key={c.sample_id}><button onClick={()=>onSelect(c.sample_id)}><img loading="lazy" src={`/api/image/${encodeURIComponent(c.sample_id)}`} alt={`Original ${c.sample_id}`}/><b>{statusName(c.symbolic_status)}</b></button><p>Expected {c.ground_truth_label} · recorded output {c.predictions[condition].parsed_answer}</p><a href={`/api/comparison/${encodeURIComponent(c.sample_id)}.svg`} download>Export comparison SVG</a></article>)}</div></section>
}
