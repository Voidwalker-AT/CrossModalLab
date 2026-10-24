import {useState} from 'react';
import type {Case, Prediction} from './main';

export const objectName=(s:string)=>s.replaceAll('_',' ');
export const statusName=(s:string)=>({entailed:'Supported by the evidence',contradicted:'Contradicted by the evidence',underdetermined:'Not enough information'}[s]??s);
export function logicExplanation(item:Case){
 const facts=item.premises.map(p=>`${objectName(p.subject)} is left of ${objectName(p.object)}`);
 const q=`${objectName(item.query.subject)} is left of ${objectName(item.query.object)}`;
 return item.symbolic_status==='underdetermined'
  ?`We know ${facts.join(' and ')}. These facts do not tell us which of the two queried objects comes first. Both orders fit the evidence, so the answer is Undetermined, not No.`
  :item.symbolic_status==='entailed'
  ?`We know ${facts.join(' and ')}. Follow the two evidence arrows: together they establish that ${q}. The answer is Yes.`
  :`We know ${facts.join(' and ')}. The evidence path goes in the reverse direction to the question. Strict left-of cannot hold both ways, so the answer is No.`;
}
export function behaviorExplanation(item:Case,p:Prediction){
 if(p.correct)return `The extracted answer matches the conclusion justified by the supplied evidence.${p.format_compliant?'':' Its output format still failed the required answer-only rule.'}`;
 if(!['Yes','No','Undetermined'].includes(p.parsed_answer))return 'The recorded output could not be interpreted as one of the three allowed answers. It counts as incorrect.';
 if(item.ground_truth_label==='Undetermined')return `The model answered ${p.parsed_answer}, but neither direction is established. A definite answer turns missing information into an unsupported claim.`;
 if(p.parsed_answer==='Undetermined')return 'The model withheld a definite answer even though the two premises settle the question. This is an incorrect abstention.';
 return `The model answered ${p.parsed_answer}, while the supplied premises establish ${item.ground_truth_label}. Its conclusion disagrees with the logical evidence.`;
}
export function escapedPreview(raw:string){return JSON.stringify(raw).replaceAll('\ufffd','\\ufffd').slice(0,64)+(JSON.stringify(raw).replaceAll('\ufffd','\\ufffd').length>64?'…':'')}
export function RawOutput({prediction,source}:{prediction:Prediction;source:string}){
 const [copied,setCopied]=useState('');
 const replacements=[...prediction.raw_completion].filter(c=>c==='\ufffd').length;
 return <div className="output-evidence"><p><b>Output format:</b> {prediction.format_category}</p>
  {!prediction.format_compliant&&<p>{replacements?`The recorded string contains ${replacements} Unicode replacement characters (U+FFFD).`:'The model generated extra text or an unparseable answer.'} {['Yes','No','Undetermined'].includes(prediction.parsed_answer)?`The historical parser accepted ${prediction.parsed_answer}, but the complete output fails the answer-only rule.`:'No valid label was extracted.'} The cause of the malformed text is unverified.</p>}
  <details><summary>Verify exact raw output and source</summary><p>Short escaped preview</p><code className="raw-preview">{escapedPreview(prediction.raw_completion)}</code><pre className="raw-answer">{prediction.raw_completion}</pre><button className="button" onClick={async()=>{try{await navigator.clipboard.writeText(prediction.raw_completion);setCopied('Exact raw output copied.')}catch{setCopied('Clipboard unavailable. Select the raw text above to copy.')}}}>Copy raw output</button><span role="status">{copied}</span><code className="hash-value">{source}</code><p>Only the preview is shortened. The raw text and clipboard content preserve the complete recorded string.</p></details>
 </div>
}
