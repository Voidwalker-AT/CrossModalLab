import {useState} from 'react';
import type {Case,Condition} from './main';

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
   <code className="hash-value">{data.model.id} @ {data.model.revision}</code><p className="caption">{data.model.checkpoint} · {data.validation.captures} captures / {data.validation.activation_rows} sparse rows · {data.validation.feature_files} feature files validated on CPU · no new model run.</p>
  </div>
  <div className="neural-panels"><div className="panel"><h3>Measured residual activation norms</h3>
   <label className="neural-control">Historical capture<select aria-label="Neural capture" value={item.sample_id} onChange={e=>setSample(e.target.value)}>{data.captures.map(c=><option key={c.sample_id}>{c.sample_id}</option>)}</select></label>
   <label className="neural-control">Token selector<select aria-label="Activation token selector" value={selector} onChange={e=>setSelector(e.target.value)}>{['last','first_image','last_image','mean_image','last_text_non_image'].map(s=><option key={s}>{s}</option>)}</select></label>
   <Trajectory series={series} unit="Residual-row L2 norm (activation units)"/>
   <div className="neural-legend"><span>● Green: {item.sample_id}</span>{paired&&<span>● Gold: {paired.sample_id}</span>}</div>
   <p className="caption">Phase 0D2 recorded behavior: {item.historical_parsed_answer}; gold {item.ground_truth}. {item.role}. The answer was recorded earlier, not generated during this passive capture.</p>
   <table className="data-table"><thead><tr><th>Layer</th><th>Token index</th><th>L2 norm</th><th>Shape / dtype</th></tr></thead><tbody>{current.map(p=><tr key={p.layer}><td>{p.layer}</td><td>{p.token_index??'Image-token aggregate'}</td><td>{p.l2_norm.toFixed(6)}</td><td>[{p.shape.join(', ')}] / {p.dtype}</td></tr>)}</tbody></table>
   <p className="caption">{paired?data.alignment:'No validated clean/failure partner for this capture. A comparison is not invented.'}</p><p className="caption">Norm growth is not evidence of improved reasoning, information flow or a causal failure layer.</p>
   <details><summary>Capture source and SHA-256</summary><code className="hash-value">{item.source}</code><code className="hash-value">{item.sha256}</code>{paired&&<code className="hash-value">Paired source: {paired.source} · {paired.sha256}</code>}</details>
  </div><div className="panel"><h3>Representation probe trajectory</h3>
   <label className="neural-control">Probe token selector<select aria-label="Probe token selector" value={probeSelector} onChange={e=>setProbeSelector(e.target.value)}>{['last','last_text_non_image','mean_image'].map(s=><option key={s}>{s}</option>)}</select></label>
   <Trajectory unit="Probe balanced accuracy (fraction)" maximum={1} series={[{name:'Stored validation probe',color:'#72578d',points:probes.map(p=>({layer:p.layer,value:p.balanced_accuracy,low:p.ci.ci_low,high:p.ci.ci_high}))}]}/>
   <p>81 training cases / 27 families; 27 validation cases / {data.probe.validation_families} families. Error bars: stored 2,000-resample descriptive family-bootstrap intervals.</p><p className="caption">{data.probe.limitation} Each point is a separately fitted probe, not a neural answer trajectory.</p>
   <table className="data-table"><thead><tr><th>Layer</th><th>BA</th><th>95% descriptive CI</th></tr></thead><tbody>{probes.map(p=><tr key={p.layer}><td>{p.layer}</td><td>{p.balanced_accuracy.toFixed(6)}</td><td>[{p.ci.ci_low.toFixed(6)}, {p.ci.ci_high.toFixed(6)}]</td></tr>)}</tbody></table>
   <details><summary>Probe source and validation families</summary><code className="hash-value">{data.probe.source}</code><p className="caption">{data.probe.family_ids.join(', ')}</p></details>
  </div></div>
  <div className="panel"><h3>Causal evidence audit</h3><span className="tag rose">UNVERIFIED · no relation-specific causal mechanism</span><p>{data.causal_audit.records} surviving rows / {data.causal_audit.interventions} interventions; {data.causal_audit.missing_planned} planned records missing.</p><p>Targeted mean |Δ margin|: {data.causal_audit.targeted_abs_delta.toFixed(6)}. Unrelated-token control: {data.causal_audit.unrelated_token_abs_delta.toFixed(6)}.</p><p className="caption">{data.causal_audit.limitation}</p><code className="hash-value">{data.causal_audit.source}</code><h3>NOT MEASURED / not validated for this dashboard</h3><ul className="facts">{data.missing.map(s=><li key={s}>{s}</li>)}</ul>
   <details><summary>Neural evidence inventory</summary>{data.inventory.map(s=><div key={s.source}><code className="hash-value">{s.source}</code><p className="caption">{s.classification} · {s.validation_status}</p></div>)}</details>
  </div>
 </section>
}

export function CheckpointComparison({item,conditions}:{item:Case;conditions:Condition[]}){
 const selected=['frozen','ce5','ce50','es5'].map(id=>conditions.find(c=>c.id===id)!);
 return <section className="panel checkpoint-comparison"><div className="panel-heading"><h3>Matched model input/output comparison</h3><a className="button" href={`/api/comparison/${encodeURIComponent(item.sample_id)}.svg`} download={`${item.sample_id.replaceAll('::','__')}_comparison.svg`}>↓ Export comparison SVG</a></div>
  <p className="caption">Same original image and standard task prompt. All four recorded checkpoints; CE+ES stopped at step five. Source details and full model revision are embedded in the SVG.</p>
  <div className="checkpoint-cards">{selected.map(c=>{const p=item.predictions[c.id];return <article key={c.id}><h4>{c.name}</h4><span className={`tag ${p.correct?'green':'rose'}`}>{p.behavior_category}</span><p><b>{p.parsed_answer}</b> · {p.format_category}</p><pre>{JSON.stringify(p.raw_completion)}</pre><details><summary>Recorded source</summary><code className="hash-value">{c.source_artifact}</code></details></article>})}</div>
  <div className="explanation-levels"><div><h4>What should have happened?</h4><span className="tag green">SYMBOLICALLY DERIVED</span><p>{item.solver.explanation}</p></div><div><h4>What the model produced</h4><span className="tag">OBSERVED</span><p>Exact recorded answers are shown above, with semantic and format judgments kept separate.</p></div><div><h4>What we observed internally</h4><span className="tag amber">NOT MEASURED for this Qwen case</span><p>Historical Smol measurements are available separately in Architecture Explorer.</p></div><div><h4>What remains unknown</h4><span className="tag rose">UNVERIFIED</span><p>Perception, composition, representation, readout and calibration are competing explanations. These outputs alone do not distinguish them.</p></div></div>
 </section>
}

export function EvidenceGallery({cases,condition,onSelect}:{cases:Case[];condition:string;onSelect:(id:string)=>void}){
 return <section className="panel evidence-gallery"><h2>Scientific evidence gallery</h2><p className="caption">All 18 original development stimuli. Select a case to inspect its input, logical evidence and recorded answers. Images are hash-verified; no tutorial data are mixed in.</p><div className="gallery-grid">{cases.map(c=><article key={c.sample_id}><button onClick={()=>onSelect(c.sample_id)}><img loading="lazy" src={`/api/image/${encodeURIComponent(c.sample_id)}`} alt={`Original ${c.sample_id}`}/><b>{c.sample_id}</b></button><p>{c.ground_truth_label} · recorded output {c.predictions[condition].parsed_answer}</p><a href={`/api/comparison/${encodeURIComponent(c.sample_id)}.svg`} download>Export comparison SVG</a></article>)}</div></section>
}
