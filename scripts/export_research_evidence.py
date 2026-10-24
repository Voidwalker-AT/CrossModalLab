"""Prepare/finalize a self-contained visual gallery from stored development evidence.

Preparation uses already installed matplotlib; viewing needs no runtime. Browser
screenshots are captured separately from the running app, then included by finalize.
Never opens a model, full dataset manifest, optimizer, or final-holdout image.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.backend.artifacts import ArtifactStore
from dashboard.backend.exports import comparison_svg, logical_svg
from dashboard.backend.gallery import render_gallery, describe_visual

FOLDERS = ["01_Dashboard_Screenshots", "02_Visual_Inputs", "03_Model_Output_Comparisons",
           "04_Logical_Evidence_Graphs", "05_Neural_Activation_Analysis", "06_Experiment_Analytics",
           "07_Research_Figures", "08_Reproducibility", "09_Demonstration"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    if out.exists():
        raise ValueError("Preserving existing evidence directory; choose a new --output")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":12,"font.family":"DejaVu Sans","svg.fonttype":"none"})
    out.mkdir(parents=True)
    for folder in FOLDERS:(out/folder).mkdir()
    store=ArtifactStore(); data=store.workspace(); neural=data["neural"]
    catalog={}
    def record(path,sources,classification="OBSERVED",sample_id=None,**extra):
        catalog[path.relative_to(out).as_posix()]={"source_reference":sources,"sample_id":sample_id,
                                                 "classification":classification,**extra}
    def write(path,raw,sources,classification="OBSERVED",sample_id=None,**extra):
        path.write_bytes(raw if isinstance(raw,bytes) else raw.encode("utf-8"))
        record(path,sources,classification,sample_id,**extra)
    def figure(fig,name,folder,sources,classification="OBSERVED",sample_id=None,**extra):
        import textwrap
        relative=folder+"/"+name+".png"
        title,description,limits=describe_visual(relative,{"sample_id":sample_id,**extra},data)
        fig.text(.08,-.30,"\n".join(textwrap.wrap(description,105))+"\n"+"\n".join(textwrap.wrap(limits,105)),fontsize=10,linespacing=1.5,va="top")
        if classification=="CORRELATED":
            import textwrap
            fig.text(.08,-.20,neural["model"]["id"]+" @ "+neural["model"]["revision"]+"\n"+
                     "\n".join(textwrap.fill("Source: "+str(source),112) for source in sources),fontsize=8)
        for ext in ("svg","png"):
            path=out/folder/(name+"."+ext)
            fig.savefig(path,bbox_inches="tight",dpi=180,metadata={"Date":None} if ext=="svg" else None)
            if ext=="svg":path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines())+"\n",encoding="utf-8")
            record(path,sources,classification,sample_id,**extra)
        plt.close(fig)
    for case in data["cases"]:
        stem=case["sample_id"].replace("::","__")
        source="dashboard/assets/images/"+stem+".png"
        image=out/FOLDERS[1]/(stem+".png")
        write(image,store.image(case["sample_id"]).read_bytes(),[source],sample_id=case["sample_id"])
        assert sha(image)==case["image_sha256"]
        svg=comparison_svg(store,case["sample_id"])
        sources=[c["source_artifact"] for c in data["conditions"] if c["id"] in {"frozen","ce5","ce50","es5"}]
        write(out/FOLDERS[2]/(stem+"_Input_Output.svg"),svg,sources,sample_id=case["sample_id"])
        graph=logical_svg(store,case['sample_id'])
        write(out/FOLDERS[3]/(stem+"_Symbolic_Graph.svg"),graph,["dashboard/assets/development_cases.json","src/symbolic/solver.py"],
              "SYMBOLICALLY DERIVED",case["sample_id"],interpretation="Logical oracle; not an internal neural trace")
    write(out/FOLDERS[7]/"Exact_Recorded_Answers.json",json.dumps({"model":data["model"],"conditions":data["conditions"],"cases":data["cases"]},indent=2),["dashboard/assets/development_cases.json",*[c["source_artifact"]for c in data["conditions"]]])
    capture_csv=[]
    colors=["#267864","#a36d21","#72578d","#456e9c","#9e505f"]
    for capture in neural["captures"]:
        fig,ax=plt.subplots(figsize=(9,5))
        for color,selector in zip(colors,["last","first_image","last_image","mean_image","last_text_non_image"]):
            points=sorted([p for p in capture["points"] if p["selector"]==selector],key=lambda p:p["layer"])
            ax.plot([p["layer"]for p in points],[p["l2_norm"]for p in points],"o-",label=selector,color=color)
        ax.set(xlabel="Recorded decoder layer index",ylabel="Residual-row L2 norm (activation units)",xticks=[3,12,18,23],title="Historical frozen SmolVLM2 | measured activation norms")
        ax.legend(fontsize=9,ncol=2);fig.text(.09,-.04,capture["sample_id"]+"\n"+capture["condition"]+" | CORRELATED - NOT CAUSAL; norm growth is not reasoning evidence.",fontsize=9)
        figure(fig,"Activation_Norms_"+capture["sample_id"],FOLDERS[4],[capture["source"]],"CORRELATED",capture["sample_id"],
               model=neural["model"],units="activation-vector L2",token_selectors=["last","first_image","last_image","mean_image","last_text_non_image"])
        for point in capture["points"]:capture_csv.append({"sample_id":capture["sample_id"],"source":capture["source"],**point})
    csvpath=out/FOLDERS[4]/"All_200_Activation_Rows.csv"
    with csvpath.open("w",newline="",encoding="utf-8") as stream:
        fields=["sample_id","source","layer","selector","token_index","l2_norm","recomputed_l2","shape","dtype"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(capture_csv)
    record(csvpath,["dashboard/assets/neural_evidence_v1.json"],"CORRELATED")
    groups={c["matched_group_id"] for c in neural["captures"] if c["matched_group_id"]}
    for group in sorted(groups):
        pair=[c for c in neural["captures"] if c["matched_group_id"]==group]
        if len(pair)!=2 or len({c["image_sha256"] for c in pair})!=1 or len({c["condition"] for c in pair})!=1:continue
        fig,ax=plt.subplots(figsize=(9,5))
        for capture,color in zip(pair,colors):
            points=sorted([p for p in capture["points"] if p["selector"]=="last"],key=lambda p:p["layer"])
            ax.plot([p["layer"] for p in points],[p["l2_norm"] for p in points],"o-",label=capture["role"],color=color)
        ax.set(title="SmolVLM2 matched logical variants | last-token residual norm",xlabel="Recorded decoder layer index",ylabel="L2 norm (activation units)",xticks=[3,12,18,23]);ax.legend(fontsize=9)
        fig.text(.08,-.07,"\n".join(c["sample_id"] for c in pair)+"\nSame image/group/selector; text may differ. Correlated comparison, not an isolated failure intervention.",fontsize=9)
        figure(fig,"Matched_Variants_"+group,FOLDERS[4],[c["source"]for c in pair],"CORRELATED",sample_id=[c["sample_id"]for c in pair],alignment=neural["alignment"])
    for selector in ["last","last_text_non_image","mean_image"]:
        points=sorted([p for p in neural["probe"]["points"] if p["selector"]==selector],key=lambda p:p["layer"])
        fig,ax=plt.subplots(figsize=(8,5));values=[p["balanced_accuracy"]for p in points]
        ax.errorbar([p["layer"]for p in points],values,yerr=[[p["balanced_accuracy"]-p["ci"]["ci_low"]for p in points],[p["ci"]["ci_high"]-p["balanced_accuracy"]for p in points]],fmt="o-",capsize=5,color="#72578d")
        ax.axhline(1/3,color="#7e887d",linestyle="--",label="Three-class balanced chance (1/3)")
        ax.set(ylim=(0,1),xticks=[3,12,18,23],xlabel="Recorded decoder layer index",ylabel="Probe balanced accuracy",title="SmolVLM2 external representation probe | "+selector);ax.legend(fontsize=9)
        fig.text(.08,-.05,"27 validation cases / 9 families; 2,000-resample descriptive family CIs.\nDevelopment-selected layer/selector/L2. CORRELATED - NOT CAUSAL; no generalization claim.",fontsize=9)
        figure(fig,"Probe_Trajectory_"+selector,FOLDERS[4],[neural["probe"]["source"]],"CORRELATED",family_ids=neural["probe"]["family_ids"],units="balanced-accuracy fraction")
    write(out/FOLDERS[4]/"Neural_Evidence_Inventory.json",json.dumps(neural,indent=2),["dashboard/assets/neural_evidence_v1.json"],"CORRELATED")
    for condition in data["conditions"]:
        m=condition["metrics"];matrix=[[m["confusion_matrix"][gold][pred] for pred in ("Yes","No","Undetermined","Invalid")]for gold in ("Yes","No","Undetermined")]
        fig,ax=plt.subplots(figsize=(7,4.7));ax.imshow(matrix,cmap="Greens",vmin=0,vmax=6)
        for i,row in enumerate(matrix):
            for j,value in enumerate(row):ax.text(j,i,str(value),ha="center",va="center",fontsize=15,color="white" if value>=4 else "#143e30")
        ax.set(xticks=range(4),xticklabels=["Yes","No","Undetermined","Invalid"],yticks=range(3),yticklabels=["Yes","No","Undetermined"],xlabel="Parsed answer",ylabel="Gold label",title=condition["name"]+" | development confusion")
        fig.text(.10,-.01,"18 cases / 6 families; six cases per class. Historical prefix parsing; formatting reported separately.",fontsize=8)
        figure(fig,"Confusion_Matrix_"+condition["id"],FOLDERS[5],[condition["source_artifact"]],condition_id=condition["id"],model=data["model"],units="case count")
    for metric,name,unit in [("ce_loss","CE_Loss","loss"),("es_loss","Evidence_Margin_Loss","loss"),("torch_max_allocated_mib","Allocator_Peak_Memory","MiB")]:
        fig,ax=plt.subplots(figsize=(8,4.3));values=[r[metric] if metric in r else r["memory"][metric]for r in data["training"]]
        ax.scatter([r["step"]for r in data["training"]],values,color="#267864",s=55);ax.set(xticks=range(1,6),xlabel="Recorded optimizer step",ylabel=unit,title=name.replace("_"," ")+" | CE+ES five-step pilot")
        fig.text(.09,-.03,"Different input at every step; not a fixed-input learning curve. Memory is allocator peak, not physical VRAM.",fontsize=9)
        figure(fig,name,FOLDERS[5],["artifacts/training/phase_2i_b/runs/"+data["run_id"]+"/train_metrics_append.jsonl"],model=data["model"],units=unit)
    for path in sorted((ROOT/"paper/figures").glob("0[1-5]_*.???")):
        if path.suffix not in {".png",".svg"}:continue
        write(out/FOLDERS[6]/path.name,path.read_bytes(),[path.relative_to(ROOT).as_posix()],"SYMBOLICALLY DERIVED" if path.name.startswith('01_') else "UNVERIFIED" if path.name.startswith('02_') else "OBSERVED")
    for source in ["paper/evidence_registry.md","paper/claims_registry.md","paper/references/verified_sources.md","paper/submission_plan.md","dashboard/WSCTL_EXIT_CODE_FIX.md","dashboard/EVIDENCE_EXPORT.md","dashboard/NEURAL_EVIDENCE.md"]:
        path=ROOT/source
        write(out/FOLDERS[7]/path.name,path.read_bytes(),[source],interpretation="Documentation; not a new empirical result")
    for source in ["paper/tables/submission_metrics.csv","paper/tables/development_summary.json","paper/tables/family_predictions.csv"]:
        path=ROOT/source;write(out/FOLDERS[5]/path.name,path.read_bytes(),[source])
    write(out/FOLDERS[7]/"Recorded_Training_Telemetry.json",json.dumps(data["training"],indent=2),[p["path"] for p in data["sources"] if p["path"].endswith('train_metrics_append.jsonl')])
    for source in ["submission/2026-10-28/five_minute_demo.md","submission/2026-10-28/demo_examples.json","dashboard/human_centered_demo.md"]:
        path=ROOT/source;write(out/FOLDERS[8]/path.name,path.read_bytes(),[source])
    html=(ROOT/"paper/manuscript/draft.html").read_text(encoding="utf-8")
    for a,b in [("../figures/","../07_Research_Figures/"),("../tables/","../06_Experiment_Analytics/"),("../references/verified_sources.md","../08_Reproducibility/verified_sources.md"),("../evidence_registry.md","../08_Reproducibility/evidence_registry.md"),("../claims_registry.md","../08_Reproducibility/claims_registry.md"),("../submission_plan.md","../08_Reproducibility/submission_plan.md"),("../../submission/2026-10-28/demo_examples.json","demo_examples.json")]:html=html.replace(a,b)
    html=html.replace('../07_Research_Figures/inventory.md','../README.md')
    write(out/FOLDERS[8]/"Preliminary_Paper.html",html,["paper/manuscript/draft.html"])
    readme='''# CrossModalLab research evidence — October 28

Open [the visual gallery](Gallery.html) in any browser. This folder and ZIP need no Python, Node, GPU, WSCTL or ULVO to view. SVGs are scalable; PNGs are convenient for slides. Original benchmark PNG bytes are preserved.

01: actual application screenshots. 02: all 18 original validation images. 03: exact four-checkpoint input/output comparisons. 04: symbolic graphs. 05: historical SmolVLM2 activation/probe plots and inventory. 06: recorded development analytics and CSVs. 07: paper figures. 08: reproducibility and evidence registries. 09: demonstration guide and preliminary paper.

The 18 Qwen validation cases and historical Smol neural experiments are separate. Smol captures must never be attributed to a selected Qwen error. Norms and probes are CORRELATED — NOT CAUSAL. The surviving causal controls do not establish a relation-specific mechanism. Missing Qwen activations, attention tensors, intermediate answer trajectories and neuron-level evidence are NOT MEASURED. No CE+ES step-50 result exists.

Comparisons preserve raw malformed suffixes using JSON escapes. Historical semantic parsing and exact formatting are different measures. Development reuse, six families and one five-step seed prohibit superiority or generalization claims. No new GPU run, optimizer step, training, model download or holdout access was used to prepare this export.

MANIFEST.json records each file's source reference, case ID where relevant, evidence classification, byte size and SHA-256. Source vectors/weights/checkpoints and credentials are excluded. Source tensor hashes attest to the local CPU validation; the tensor files themselves are not included. Static plots are exports, not live model inference.
'''
    write(out/"README.md",readme,["scripts/export_research_evidence.py"],interpretation="Sharing instructions and interpretation limits")
    (out/"08_Reproducibility/export_catalog.json").write_text(json.dumps(catalog,indent=2),encoding="utf-8")
    print(json.dumps({"prepared":str(out),"registered_files":len(catalog),"new_gpu_forwards":0}))


def finalize(out):
    archive=out.parent/(out.name+'.zip')
    target=ROOT/'artifacts/reports'/('research_evidence_manifest_'+out.name.removeprefix('CrossModalLab_Research_Evidence_Oct28_')+'.json')
    if (out/'MANIFEST.json').exists() or archive.exists() or target.exists():
        raise ValueError('Finalized release preserved; choose a fresh versioned output')
    catalog=json.loads((out/"08_Reproducibility/export_catalog.json").read_text(encoding="utf-8"))
    capture_log=out/"08_Reproducibility/browser_capture_log.json"
    browser=json.loads(capture_log.read_text(encoding="utf-8"))
    if browser["status"]!="PASS":raise ValueError("browser evidence did not pass")
    catalog.update(browser["files"])
    (out/"Gallery.html").write_text(render_gallery(catalog,ArtifactStore().workspace()),encoding="utf-8")
    for name in ["Gallery.html","08_Reproducibility/export_catalog.json","08_Reproducibility/browser_capture_log.json"]:
        catalog[name]={"source_reference":["scripts/export_research_evidence.py"],"sample_id":None,"classification":"OBSERVED","interpretation":"Export documentation / actual capture log"}
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    if actual!=set(catalog):raise ValueError('Uncatalogued payload: '+str(actual.symmetric_difference(catalog)))
    patterns=[rb'GOCSPX-[A-Za-z0-9_-]{20,}',rb'AIza[0-9A-Za-z_-]{35}',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}',rb'AKIA[A-Z0-9]{16}',rb'"client_secret"\s*:\s*"[^"\s]{8,}"']
    files={}
    for name,info in sorted(catalog.items()):
        path=out/name;raw=path.read_bytes()
        if path.suffix.lower() in {'.pt','.bin','.safetensors','.npz','.npy','.env'}:raise ValueError('Forbidden payload '+name)
        if any(re.search(p,raw)for p in patterns):raise ValueError('Credential pattern in '+name)
        source_hashes={}
        for reference in info["source_reference"]:
            candidate=(ROOT/reference).resolve()
            if candidate.is_relative_to(ROOT) and candidate.is_file():source_hashes[reference]=sha(candidate)
            elif (out/reference).is_file():source_hashes[reference]=sha(out/reference)
        files[name]={**info,"source_sha256":source_hashes,"bytes":len(raw),"sha256":sha(path)}
    manifest={"schema":"crossmodallab_visual_evidence_v1","source_commit":__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"is_synthetic_demo":False,"new_gpu_forwards":0,"new_optimizer_steps":0,"final_holdout_accessed":False,"files":files}
    manifest_path=out/'MANIFEST.json'
    if manifest_path.exists():raise ValueError('Existing finalized manifest preserved; choose a new output')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=out.parent/(out.name+'.zip')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob('*')):
            if path.is_file():zipped.write(path,arcname=out.name+'/'+path.relative_to(out).as_posix())
    scratch=ROOT/'.cache/evidence_verification'/uuid4().hex
    scratch.mkdir(parents=True)
    with ZipFile(archive) as zipped:
        if zipped.testzip() is not None:raise ValueError('ZIP CRC failed')
        for name in zipped.namelist():
            if not (scratch/name).resolve().is_relative_to(scratch.resolve()):raise ValueError('Unsafe archive path')
        zipped.extractall(scratch)
    extracted=scratch/out.name
    assert {p.relative_to(extracted).as_posix() for p in extracted.rglob('*')if p.is_file()}==set(files)|{'MANIFEST.json'}
    for name,identity in files.items():
        assert (extracted/name).stat().st_size==identity['bytes'] and sha(extracted/name)==identity['sha256'],name
    assert sha(extracted/'MANIFEST.json')==sha(manifest_path)
    report={"status":"PASS","folder":str(out),"archive":str(archive),"archive_bytes":archive.stat().st_size,"archive_sha256":sha(archive),"files_verified":len(files)+1,"extracted_to":str(extracted),"source_commit":manifest['source_commit']}
    target=ROOT/'artifacts/reports'/('research_evidence_manifest_'+out.name.removeprefix('CrossModalLab_Research_Evidence_Oct28_')+'.json')
    if target.exists():raise ValueError('Preserving existing small export manifest')
    target.write_bytes(manifest_path.read_bytes())
    (ROOT/'.cache/evidence_verification/result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'submission/CrossModalLab_Research_Evidence_Oct28_HumanCentered_20261024')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):parser.error('output must stay within the checkout')
    (finalize if args.finalize else prepare)(out)
