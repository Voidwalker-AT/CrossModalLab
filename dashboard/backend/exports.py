"""Self-contained explained SVGs; exact strings and identities remain in metadata."""
import base64
from html import escape
import json
import textwrap
from dashboard.backend.explanations import STATUS, name, explain_case, comparison_takeaway


class Canvas:
    def __init__(self, height):
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="{height}" viewBox="0 0 1500 {height}">',
                    f'<rect width="1500" height="{height}" fill="#f5f7f2"/>',
                    '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0 L0 6 L8 3Z" fill="context-stroke"/></marker></defs>']
    def text(self,x,y,value,size=21,color='#213d39'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{escape(str(value))}</text>')
    def block(self,x,y,value,width=95,size=21,leading=30):
        for paragraph in str(value).splitlines():
            for line in textwrap.wrap(paragraph,width=width) or ['']:
                self.text(x,y,line,size);y+=leading
        return y
    def finish(self,metadata):
        self.parts.append('<metadata>'+escape(json.dumps(metadata,ensure_ascii=True))+'</metadata></svg>')
        return '\n'.join(self.parts).encode('utf-8')


def draw_graph(canvas,case,top):
    t=canvas.text
    t(40,top+32,'What should the answer be?  '+case['ground_truth_label']+' — '+STATUS[case['symbolic_status']],26)
    positions={entity:(110+230*i,top+142) for i,entity in enumerate(case['entities'].values())}
    for index,p in enumerate(case['premises']):
        a,b=positions[p['subject']],positions[p['object']];direction=1 if b[0]>a[0] else -1
        color='#267864' if p['modality']=='visual' else '#94621a'
        dash='' if p['modality']=='visual' else ' stroke-dasharray="8 5"'
        curve=top+(75 if index else 208)
        if index in case['reasoning']['proof_edge_indices']:
            canvas.parts.append(f'<path data-proof-edge="{index}" d="M{a[0]+direction*25} {a[1]} Q{(a[0]+b[0])/2} {curve} {b[0]-direction*27} {b[1]}" fill="none" stroke="#9bbb8c" stroke-width="15" opacity=".4"/>')
        canvas.parts.append(f'<path d="M{a[0]+direction*25} {a[1]} Q{(a[0]+b[0])/2} {curve} {b[0]-direction*27} {b[1]}" fill="none" stroke="{color}" stroke-width="4"{dash} marker-end="url(#arrow)"/>')
    for letter,entity in case['entities'].items():
        x,y=positions[entity];canvas.parts.append(f'<circle cx="{x}" cy="{y}" r="25" fill="white" stroke="#267864"/>')
        t(x-7,y+7,letter,23);t(x-82,y+60,name(entity),18)
    t(40,top+242,'Green solid: image fact. Gold dashed: text fact.',18)
    t(40,top+270,'Every arrow means “is left of.” Diagram positions are not scene coordinates.',17)
    canvas.block(720,top+85,'Query only: Is '+name(case['query']['subject'])+' left of '+name(case['query']['object'])+'? (not a premise)',width=62,size=20,leading=27)
    canvas.block(720,top+163,explain_case(case),width=62,size=20,leading=27)
    orders=case['reasoning']['compatible_orderings']
    canvas.block(40,top+310,'Compatible logical order'+('s' if len(orders)>1 else '')+': '+'; or '.join(' < '.join(name(n) for n in o['entities']) for o in orders),width=130,size=18,leading=25)
    t(40,top+370,'These are symbolic consequences of supplied facts, not a measured neural reasoning trace.',18)


def logical_svg(store,sample_id):
    case=next(c for c in store.workspace()['cases'] if c['sample_id']==sample_id)
    c=Canvas(520);c.text(40,40,'CrossModalLab | Read the evidence before choosing an answer',30)
    draw_graph(c,case,65);c.text(40,505,'Source: development_cases.json + dual symbolic solver | '+sample_id,16)
    return c.finish({'case':case,'classification':'SYMBOLICALLY DERIVED','neural_trace':False})


def comparison_svg(store,sample_id):
    data=store.workspace();case=next(c for c in data['cases'] if c['sample_id']==sample_id)
    conditions={c['id']:c for c in data['conditions']};keys=('frozen','ce5','ce50','es5')
    c=Canvas(1320);t=c.text
    t(40,45,'Can the picture and text settle this question?',32)
    t(40,78,'CrossModalLab | '+STATUS[case['symbolic_status']]+' | actual development evidence',21)
    png=base64.b64encode(store.image(sample_id).read_bytes()).decode()
    c.parts.append(f'<image x="40" y="105" width="360" height="240" href="data:image/png;base64,{png}"/>')
    t(440,135,'What question did we ask?',26)
    c.block(440,177,case['prompt'].split('\n\n')[0],width=80,size=23,leading=32)
    c.block(440,260,'Is '+name(case['query']['subject'])+' left of '+name(case['query']['object'])+'?',width=80,size=24,leading=34)
    t(40,373,'Original model input. The third object is supplied by text; its position is not invented.',19)
    draw_graph(c,case,400)
    t(40,818,'Did fine-tuning change the answer?',28)
    t(40,854,comparison_takeaway(case),23)
    for x,value in [(40,'Checkpoint / training'),(650,'Extracted answer'),(925,'Correctness'),(1180,'Format')]:t(x,901,value,20)
    for i,key in enumerate(keys):
        p=case['predictions'][key];y=945+53*i
        description='before fine-tuning' if key=='frozen' else 'exploratory sufficiency' if key=='es5' else 'standard supervised'
        t(40,y,conditions[key]['name']+' — '+description,20)
        t(650,y,p['parsed_answer'],24);t(925,y,'Correct' if p['correct'] else 'Incorrect',21)
        t(1180,y,'Exact label' if p['format_compliant'] else 'Malformed',21)
    t(40,1170,'Steps count optimizer updates, not epochs. CE+ES stopped at five; no step-50 result exists.',20)
    t(40,1204,'Malformed suffixes are preserved in SVG metadata and the accompanying Exact_Recorded_Answers.json.',19)
    t(40,1238,'Internal neural measurements have not yet been recorded for this Qwen example. Cause: UNVERIFIED.',19)
    t(40,1272,sample_id+' | '+data['model']['id'],17)
    t(40,1300,'Revision: '+data['model']['revision']+' | Small reused development set; no superiority claim.',16)
    return c.finish({'sample_id':sample_id,'model':data['model'],'image_sha256':case['image_sha256'],'prompt':case['prompt'],'prompt_sha256':case['prompt_sha256'],'predictions':{k:case['predictions'][k]for k in keys},'sources':{k:conditions[k]['source_artifact']for k in keys},'classification':'OBSERVED / SYMBOLICALLY DERIVED'})
