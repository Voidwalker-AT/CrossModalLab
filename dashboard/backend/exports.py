"""Self-contained comparison SVG: exact stored inputs/outputs, no model execution."""
import base64
from html import escape
import json
import textwrap


def comparison_svg(store, sample_id):
    data = store.workspace()
    case = next(c for c in data["cases"] if c["sample_id"] == sample_id)
    conditions = {c["id"]: c for c in data["conditions"]}
    keys = ("frozen", "ce5", "ce50", "es5")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1080" viewBox="0 0 1500 1080">',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0 L0 6 L8 3Z" fill="#267864"/></marker></defs>',
             '<rect width="1500" height="1080" fill="#f5f7f2"/>']
    def text(x,y,value,size=15,color="#213d39"):
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{escape(str(value))}</text>')
    def block(x,y,value,width=95,size=14,leading=19):
        for paragraph in str(value).splitlines():
            for line in textwrap.wrap(paragraph,width=width) or [""]:
                text(x,y,line,size)
                y += leading
        return y
    text(28,35,"CrossModalLab | observed model input/output comparison",24)
    text(28,61,case["sample_id"] + " | development validation | no final holdout",14)
    text(28,83,data["model"]["id"] + " @ " + data["model"]["revision"],12)
    parts.append('<rect x="20" y="98" width="1460" height="300" rx="10" fill="white"/>')
    png = base64.b64encode(store.image(sample_id).read_bytes()).decode()
    parts.append(f'<image x="38" y="118" width="315" height="220" href="data:image/png;base64,{png}"/>')
    text(38,358,"Original benchmark image; bytes preserved",12)
    block(38,377,"SHA-256: "+case["image_sha256"],width=48,size=9,leading=12)
    text(390,126,"Exact source prompt (all four conditions below use the standard prompt)",15)
    block(390,151,case["prompt"],width=121,size=13,leading=18)
    text(390,372,"Prompt SHA-256: "+case["prompt_sha256"],11)
    parts.append('<rect x="20" y="416" width="1460" height="256" rx="10" fill="white"/>')
    text(38,444,"SYMBOLICALLY DERIVED | expected answer: "+case["ground_truth_label"],17)
    text(730,441,case["sample_id"],12)
    text(730,461,"Query only: "+case["query"]["subject"]+" < "+case["query"]["object"]+" ? (not a premise)",12)
    positions = {entity: (100+230*i,530) for i,entity in enumerate(case["entities"].values())}
    for index,premise in enumerate(case["premises"]):
        a,b=positions[premise["subject"]],positions[premise["object"]]
        direction=1 if b[0]>a[0] else -1
        color="#267864" if premise["modality"]=="visual" else "#a36d21"
        dash='' if premise["modality"]=="visual" else ' stroke-dasharray="7 4"'
        parts.append(f'<path d="M{a[0]+direction*25} {a[1]} Q{(a[0]+b[0])/2} {470 if index else 598} {b[0]-direction*27} {b[1]}" fill="none" stroke="{color}" stroke-width="3"{dash} marker-end="url(#arrow)"/>')
    for letter,entity in case["entities"].items():
        x,y=positions[entity]
        parts.append(f'<circle cx="{x}" cy="{y}" r="24" fill="#eff5ee" stroke="#267864"/>')
        text(x-5,y+5,letter,17)
        text(x-70,y+43,entity.replace("_"," "),12)
    text(38,626,"Green solid: visual premise | Gold dashed: textual premise",12)
    block(730,480,case["solver"]["explanation"],width=78,size=13,leading=19)
    orders = case["reasoning"]["compatible_orderings"]
    block(730,580,"Compatible ordering(s): "+" ; ".join(" < ".join(o["entities"]) for o in orders),width=78,size=12,leading=18)
    text(38,654,"This graph is the symbolic oracle, not a neural thought process. Internal cause remains UNVERIFIED.",12)
    for i,key in enumerate(keys):
        x=20+i*369
        prediction=case["predictions"][key]
        parts.append(f'<rect x="{x}" y="692" width="353" height="321" rx="10" fill="white"/>')
        text(x+14,720,conditions[key]["name"],17)
        text(x+14,746,"Parsed: "+str(prediction["parsed_answer"]),16)
        text(x+14,771,prediction["behavior_category"],13)
        text(x+14,794,"Format: "+prediction["format_category"],12)
        block(x+14,821,"Raw JSON: "+json.dumps(prediction["raw_completion"],ensure_ascii=True),width=43,size=11,leading=16)
        block(x+14,898,"Source: "+conditions[key]["source_artifact"],width=48,size=9,leading=13)
        text(x+14,991,"OBSERVED | checkpoint step "+str(conditions[key]["step"]),11)
    text(28,1041,"CE+ES is paused at five steps. Unequal training budgets; no superiority or causal-mechanism claim.",14)
    text(28,1064,"Raw suffixes are preserved as JSON escapes. Exact format and semantic extraction are evaluated separately.",12)
    metadata={"sample_id":sample_id,"model":data["model"],"image_sha256":case["image_sha256"],
              "prompt":case["prompt"],"prompt_sha256":case["prompt_sha256"],
              "predictions":{k:case["predictions"][k] for k in keys},"classification":"OBSERVED / SYMBOLICALLY DERIVED"}
    parts.append('<metadata>'+escape(json.dumps(metadata,ensure_ascii=True))+'</metadata></svg>')
    return "\n".join(parts).encode("utf-8")
