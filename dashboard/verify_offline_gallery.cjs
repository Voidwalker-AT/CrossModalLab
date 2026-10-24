/* Verify an extracted release using an existing Playwright runtime and installed Edge. */
const fs=require('fs'),path=require('path'),{createRequire}=require('module'),{pathToFileURL}=require('url');
const runtime=process.env.CROSSMODAL_BROWSER_NODE_MODULES;
const {chromium}=runtime?createRequire(path.join(runtime,'package.json'))('playwright'):require('playwright');
const root=path.resolve(process.argv[2]);
const reportDir=path.resolve('.cache/evidence_verification');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),network=[],errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route(/^https?:/,r=>{network.push(r.request().url());return r.abort()});
  await page.goto(pathToFileURL(path.join(root,'Gallery.html')).href);
  const hrefs=await page.locator('a').evaluateAll(xs=>xs.map(x=>x.getAttribute('href')));
  for(const href of hrefs){
   if(href.startsWith('#')){if(!await page.locator(href).count())throw Error('Missing anchor '+href);continue}
   const target=path.resolve(root,decodeURIComponent(href.split('#')[0]));
   if(!target.startsWith(root+path.sep)||!fs.existsSync(target))throw Error('Missing or unsafe link '+href);
  }
  // Open the archive collections too; every exported PNG must have an explanation.
  await page.locator('.archive>details').evaluateAll(xs=>xs.forEach(x=>x.open=true));
  const images=await page.locator('figure').evaluateAll(xs=>Promise.all(xs.map(async f=>{
   const image=f.querySelector('img');image.loading='eager';await image.decode();
   return {src:image.getAttribute('src'),ok:image.naturalWidth>0,title:f.querySelector('h3')?.textContent,
           explanation:f.querySelector('figcaption p')?.textContent,source:!!f.querySelector('figcaption a[href="MANIFEST.json"]')};
  })));
  if(images.some(x=>!x.ok||!x.title||x.explanation.length<50||!x.source))throw Error('Unexplained or broken image');
  const unique=new Set(images.map(x=>x.src));
  const manifest=JSON.parse(fs.readFileSync(path.join(root,'MANIFEST.json'),'utf8'));
  if(unique.size!==Object.keys(manifest.files).filter(n=>n.endsWith('.png')).length)throw Error('Gallery omitted exported PNG');
  await page.locator('.archive>details').evaluateAll(xs=>xs.forEach(x=>x.open=false));
  await page.screenshot({path:path.join(reportDir,'human_gallery_desktop.png')});
  await page.locator('#answers').screenshot({path:path.join(reportDir,'human_gallery_answers.png')});
  await page.setViewportSize({width:390,height:844});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
  await page.locator('#input').screenshot({path:path.join(reportDir,'human_gallery_mobile.png')});
  await page.getByRole('link',{name:'What correct reasoning requires',exact:true}).focus();await page.keyboard.press('Enter');
  if(!page.url().endsWith('#logic'))throw Error('Keyboard story navigation');
  await page.goto(pathToFileURL(path.join(root,'09_Demonstration/Preliminary_Paper.html')).href);
  const paper=await page.locator('figure img').evaluateAll(xs=>Promise.all(xs.map(async x=>{await x.decode();return x.naturalWidth>0})));
  if(paper.length!==5||paper.some(x=>!x)||network.length||errors.length)throw Error(JSON.stringify({paper,network,errors}));
  const result={status:'PASS',unique_images:unique.size,featured_and_archive_images:images.length,all_images_explained:true,local_links:hrefs.length,paper_images:5,network_requests:network.length,page_errors:errors,mobile_overflow:false,keyboard_navigation:true};
  fs.writeFileSync(path.join(reportDir,'human_offline_browser.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
