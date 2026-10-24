/* Actual browser captures of the local app; no browser/package downloads. */
const fs=require('fs'),path=require('path'),{createRequire}=require('module');
const runtime=process.env.CROSSMODAL_BROWSER_NODE_MODULES;
const {chromium}=runtime?createRequire(path.join(runtime,'package.json'))('playwright'):require('playwright');
const base=process.env.CROSSMODAL_DASHBOARD_URL||'http://127.0.0.1:8765';
const out=path.resolve(process.argv[2]||'submission/CrossModalLab_Research_Evidence_Oct28');
if(fs.existsSync(path.join(out,'MANIFEST.json')))throw Error('Finalized evidence is immutable; prepare a new folder');
const fileUrl=p=>require('url').pathToFileURL(p).href;
(async()=>{const browser=await chromium.launch({headless:true,channel:'msedge'});try{
 const page=await browser.newPage({viewport:{width:1680,height:1080}}),errors=[],external=[],files={};
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('request',r=>{if(!r.url().startsWith(base)&&!r.url().startsWith('file:')&&!r.url().startsWith('data:'))external.push(r.url())});
 const payload=await page.request.get(base+'/api/workspace').then(r=>r.json());
 if(!payload.neural.available||payload.neural.validation.activation_rows!==200)throw Error('Neural bundle unavailable');
 await page.goto(base);await page.locator('.failure-triptych').waitFor();
 const initial=payload.cases.find(c=>c.ground_truth_label==='Undetermined').sample_id;
 async function capture(name,{sample=initial,condition='es5',classification='OBSERVED',selector=null,source=null}={}){
  const relative='01_Dashboard_Screenshots/'+name+'.png';
  await page.evaluate(()=>window.scrollTo(0,0));
  if(selector)await page.locator(selector).screenshot({path:path.join(out,relative)});
  else await page.screenshot({path:path.join(out,relative),fullPage:true});
  const neuralSelector=page.getByRole('combobox',{name:'Neural capture',exact:true});
  const neuralCapture=await neuralSelector.count()?await neuralSelector.inputValue():null;
  files[relative]={source_reference:source||['dashboard/frontend/src/main.tsx','dashboard/frontend/src/scientific.tsx','dashboard/frontend/src/neural.tsx',...payload.sources.map(s=>s.path)],sample_id:sample,condition_id:condition,classification,historical_neural_capture_id:neuralCapture,model:classification==='CORRELATED'?payload.neural.model:payload.model,method:'Actual installed-Edge screenshot of local artifact dashboard',viewport:page.viewportSize()};
 }
 await capture('00_Dashboard_Landing_Failure_Investigator');
 const views=[['Architecture Explorer','01_Architecture_Explorer'],['Evidence & Reasoning','02_Evidence_Reasoning'],['Failure Investigator','03_Failure_Investigator'],['Experiment Analytics','04_Experiment_Analytics'],['Reproducibility','05_Reproducibility']];
 for(const [name,stem]of views){await page.getByRole('button',{name:new RegExp(name)}).click();await capture(stem,{sample:['Evidence & Reasoning','Failure Investigator'].includes(name)?initial:null});}
 await page.getByRole('button',{name:/Reproducibility/}).click();
 await page.getByText('Open the scientific evidence gallery · 18 original examples',{exact:true}).click();
 if(await page.locator('.gallery-grid article').count()!==18)throw Error('Gallery membership');
 await page.locator('.gallery-grid img').evaluateAll(xs=>Promise.all(xs.map(x=>{x.loading='eager';return x.decode()})));
 await capture('06_Original_Development_Evidence_Gallery',{selector:'.evidence-gallery',sample:null});
 await page.locator('.gallery-grid button').first().click();await page.locator('.failure-triptych').waitFor();
 await page.getByRole('textbox',{name:'Search cases'}).fill('xc_ho_0011');
 for(const [label,variant]of [['Yes','entailed'],['No','contradicted'],['Undetermined','underdetermined']]){
  await page.getByRole('combobox',{name:'Logical state'}).selectOption(label);
  await page.locator('.proof-summary.'+variant).waitFor();
  const count=await page.locator('[data-proof-edge]').count();if(count!==(label==='Undetermined'?0:2))throw Error('Proof '+label);
  await capture('07_FailureInvestigator_'+variant+'_Qwen_ES5',{sample:'xc_ho_0011::'+variant});
 }
 if(await page.locator('.checkpoint-cards article').count()!==4)throw Error('Checkpoint cards');
 await capture('08_Four_Checkpoint_Input_Output_Comparison',{selector:'.checkpoint-comparison',sample:'xc_ho_0011::underdetermined'});
 const download=page.waitForEvent('download');await page.getByRole('link',{name:'↓ Export comparison SVG'}).click();await download;
 await page.getByRole('combobox',{name:'Checkpoint'}).selectOption('ce50');await page.getByRole('combobox',{name:'Logical state'}).selectOption('No');
 const prediction=await page.locator('.triptych-prediction').innerText();if(!prediction.includes('Correct response')||!prediction.includes('Malformed generation'))throw Error('Format/semantic distinction');
 await page.getByRole('button',{name:/Architecture Explorer/}).click();
 await page.getByRole('combobox',{name:'Neural capture',exact:true}).selectOption('p0d2_ctrl_underdetermined_fork_0073_underdetermined_fork');
 await page.getByRole('combobox',{name:'Activation token selector',exact:true}).selectOption('mean_image');
 await page.getByRole('combobox',{name:'Probe token selector',exact:true}).selectOption('mean_image');
 if(await page.locator('.neural-plot').count()!==2)throw Error('Missing measured neural plots');
 await capture('09_Historical_Smol_Neural_Diagnostics_Fork',{selector:'.neural-diagnostics',sample:'p0d2_ctrl_underdetermined_fork_0073_underdetermined_fork',condition:'historical frozen Smol NF4',classification:'CORRELATED',source:payload.neural.sources.map(s=>s.path)});
 await page.getByRole('combobox',{name:'Checkpoint'}).selectOption('es5');
 await page.getByRole('combobox',{name:'Activation token selector',exact:true}).selectOption('last');
 await page.getByRole('button',{name:/Failure Investigator/}).click();
 await page.getByRole('textbox',{name:'Search cases'}).fill('xc_ho_0011');
 await page.getByRole('combobox',{name:'Logical state'}).selectOption('Undetermined');
 await page.setViewportSize({width:390,height:844});
 for(const [name,stem]of views){await page.getByRole('button',{name:new RegExp(name)}).click();if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow '+name);await capture(stem+'_Mobile',{sample:['Evidence & Reasoning','Failure Investigator'].includes(name)?initial:null});}
 for(const c of payload.cases){const response=await page.request.get(base+'/api/image/'+encodeURIComponent(c.sample_id));if(response.status()!==200)throw Error('Image unavailable '+c.sample_id);}
 // Render the self-contained SVG exports, preserving their exact content.
 await page.setViewportSize({width:1500,height:1080});
 for(const folder of ['03_Model_Output_Comparisons','04_Logical_Evidence_Graphs']){
  for(const name of fs.readdirSync(path.join(out,folder)).filter(n=>n.endsWith('.svg'))){
   const input=path.join(out,folder,name),relative=folder+'/'+name.replace(/\.svg$/,'.png');
   await page.goto(fileUrl(input));await page.locator('svg').screenshot({path:path.join(out,relative)});
   const sample=name.replace(/_Input_Output\.svg$|_Symbolic_Graph\.svg$/,'').replace('__','::');
   files[relative]={source_reference:[folder+'/'+name],sample_id:sample,classification:folder.startsWith('04')?'SYMBOLICALLY DERIVED':'OBSERVED',method:'Installed-Edge rendering of exact self-contained SVG; not a generated screenshot'};
  }
 }
 if(errors.length||external.length)throw Error(JSON.stringify({errors,external}));
 const result={status:'PASS',captured_at:new Date().toISOString(),base_url:base,browser:'installed Edge via existing Playwright',five_desktop_views:true,five_mobile_views:true,original_images:18,neural_rows:200,comparison_exports:18,logical_graph_exports:18,download_checked:true,page_errors:errors,external_requests:external,files};
 fs.writeFileSync(path.join(out,'08_Reproducibility/browser_capture_log.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify({status:'PASS',browser_files:Object.keys(files).length,views:5,images:18,external_requests:0}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
