"""Offline research story. All images refer to existing catalogued evidence."""
from html import escape
from dashboard.backend.explanations import STATUS, explain_case, comparison_takeaway, name


def describe_visual(relative, info, data):
    """Return a title, explanation and limits for every exported image."""
    folder, filename = relative.split('/', 1)
    sample = info.get('sample_id')
    case = next((c for c in data['cases'] if c['sample_id'] == sample), None)
    if case and folder.startswith('02_'):
        return ('What the model sees: '+STATUS[case['symbolic_status']].lower(),
                'Original image for the question: Is '+name(case['query']['subject'])+' left of '+name(case['query']['object'])+'? The text adds: '+case['prompt'].split('\n\n')[0],
                'This is an actual controlled benchmark stimulus, not a photograph or an invented tutorial result. The image alone need not settle the question.')
    if case and folder.startswith(('03_', '04_')):
        return (('Follow the evidence: ' if folder.startswith('04_') else 'Compare the recorded answers: ')+STATUS[case['symbolic_status']].lower(),
                explain_case(case), 'The graph is symbolic ground truth, not the model’s thought process. '+comparison_takeaway(case))
    if folder.startswith('05_'):
        if filename.startswith('Probe_'):
            selector=filename.removeprefix('Probe_Trajectory_').removesuffix('.png')
            points=sorted((p for p in data['neural']['probe']['points'] if p['selector']==selector),key=lambda p:p['layer'])
            return ('Can an external classifier read the evidence state?',
                    f"Purple points show separately fitted probes at layers 3, 12, 18 and 23. For {selector.replace('_',' ')}, balanced accuracy changes from {points[0]['balanced_accuracy']:.1%} to {points[-1]['balanced_accuracy']:.1%}. Higher means better label recovery by this external classifier; bars are descriptive family-bootstrap intervals (2,000 resamples), not confidence in a model answer.",
                    'Historical SmolVLM2: 27 validation cases, nine families. Layer, selector and regularization were development-selected. Accessible information does not prove causal use or generalization.')
        if filename.startswith('Matched_'):
            return ('Same image, two related logical tasks',
                    'Colored lines compare last-token vector magnitudes at four recorded SmolVLM2 layers. Higher means a longer internal vector, not more evidence or better reasoning. The same image, group, condition and selector were validated; the text can differ.',
                    'Only two historical captures per pair. This comparison does not isolate correctness as a cause and cannot explain a Qwen prediction.')
        capture=next((c for c in data['neural']['captures'] if c['sample_id']==sample),None)
        text='Colored lines identify token selections: last token; first and last image token; mean of image tokens; last non-image text token. Four sampled layer indices are on the horizontal axis; vector length (L2 norm) is vertical. Higher means greater magnitude, not improved reasoning.'
        if capture:
            pts=sorted((p for p in capture['points'] if p['selector']=='last'),key=lambda p:p['layer'])
            text+=f" The last-token norm is {pts[0]['l2_norm']:.2f} at layer 3 and {pts[-1]['l2_norm']:.2f} at layer 23."
        return ('How internal representation size changes across layers',text,
                'One historical frozen SmolVLM2 capture, five selectors and four sampled layers. No values are measured between those layers. Magnitude does not identify a causal neuron, circuit or failure layer; this is a different model from Qwen.')
    if folder.startswith('06_'):
        if filename.startswith('Confusion_'):
            c=next(c for c in data['conditions'] if c['id']==info['condition_id']);m=c['metrics']
            return ('Which answers did '+c['name']+' confuse?',
                    f"Rows are correct labels; columns are extracted answers. Each row has six cases. Darker green means more cases (0–6); the diagonal is correct. This checkpoint correctly recognizes {round(m['per_class_recall']['Undetermined']*6)}/6 insufficient-evidence cases. Its balanced accuracy is {m['balanced_accuracy']:.1%}; exact-format compliance is {m['exact_compliance']:.1%}.",
                    '18 reused development cases in six families. Historical prefix parsing can accept malformed text; semantic correctness and exact formatting remain separate. No independent generalization claim.')
        definitions={'CE_Loss':('How costly was the correct answer at each update?','Cross-entropy penalizes low likelihood of the correct answer; lower is a smaller penalty on that input.'),
                     'Evidence_Margin_Loss':('Did the correct answer meet the evidence margin?','Evidence-sufficiency loss penalizes too small a score gap between the correct answer and alternatives; lower means a smaller margin penalty.'),
                     'Allocator_Peak_Memory':('How much GPU memory did the pilot use?','Vertical values are peak allocated MiB; lower means less allocator memory, not necessarily faster or better reasoning.')}
        title,definition=definitions.get(filename.removesuffix('.png'),('Recorded pilot measurements','Each point is an actual stored measurement.'))
        return (title,definition+' Five separate dots are the five recorded optimizer updates, each using a different input. Colors identify the named metric; no smooth learning curve is inferred.',
                'One five-step CE+ES seed. Losses across different inputs do not establish learning progress. Allocator memory differs from total physical VRAM. No CE+ES step-50 result exists.')
    if folder.startswith('07_'):
        titles={'01_':'Three meanings of a spatial question','02_':'How the research connects inputs, evidence and evaluation','03_':'Accuracy and answer format tell different stories','04_':'Which logical classes remain difficult?','05_':'A definite answer without enough evidence'}
        title=next((v for k,v in titles.items() if filename.startswith(k)),'Research evidence overview')
        return (title,'This preserved preliminary-paper figure summarizes the same recorded development evidence or its formal research pipeline. Read the labels and caption inside the figure; the pipeline is a conceptual design, not a measured neural circuit.',
                '18 Qwen cases, six families, one short exploratory pilot. Higher accuracy/recall/format compliance is better, but unequal update budgets and malformed outputs prevent a simple method-ranking claim. Formal diagrams are not neural traces.')
    titles={'00_':'Start with the question and recorded answer','01_':'Explore the model architecture and separate historical measurements','02_':'Follow image and text evidence','03_':'Investigate a recorded failure','04_':'Read class-sensitive experimental results','05_':'Verify where the evidence came from','06_':'Browse original development images','07_':'Compare logical controls','08_':'Compare checkpoints on the same question','09_':'Explore a historical SmolVLM2 example','10_':'Walk through the research in five minutes'}
    title=next((v for k,v in titles.items() if filename.startswith(k)),'Explore the research workbench')
    return (title+(' · mobile layout' if 'Mobile' in filename else ''),
            'Actual screenshot of the read-only CrossModalLab application. Start with the plain-language question and conclusion; expand technical panels only when you want to verify a detail.',
            'Screenshots replay stored evidence without a GPU. Architecture diagrams are conceptual; Smol measurements are separate from Qwen behavior. No screenshot establishes a causal mechanism.')


def render_gallery(catalog, data):
    pngs=[(r,i) for r,i in sorted(catalog.items()) if r.endswith('.png')]
    def card(relative,info):
        title,description,limits=describe_visual(relative,info,data)
        return f'<figure><h3>{escape(title)}</h3><a href="{escape(relative)}"><img loading="lazy" src="{escape(relative)}" alt="{escape(title)} — click to enlarge"></a><figcaption><p>{escape(description)}</p><p class="limit">{escape(limits)}</p><p class="category">Evidence category: {escape(info.get("classification","OBSERVED"))}</p><a href="MANIFEST.json">Verify sources and file hashes</a><details><summary>Technical filename and source records</summary><code>{escape(relative)}</code><p>{escape(str(info.get("source_reference",[])))}</p></details></figcaption></figure>'
    sections=[('problem','The problem','A confident answer can be unsupported. Cross-modal reasoning means combining an image fact and a text fact. We ask whether the evidence supports Yes, supports No, or leaves the question open.'),
              ('input','What the model sees','Begin with one actual recorded case: xc_ho_0011::underdetermined. The image shows a golden disc left of a teal wedge. The text says the crimson block is left of the teal wedge. Is the disc left of the block?'),
              ('logic','What correct reasoning requires','Both disc–block–wedge and block–disc–wedge fit the facts. The symbolic answer is Undetermined. Missing information is not evidence for No.'),
              ('answers','What the model actually answered','All four standard-prompt checkpoints answered No on this featured fork. Frozen Qwen and CE+ES have clean labels; CE checkpoints have malformed suffixes. A format difference is not a correction of this reasoning error.'),
              ('neural','What we observed inside a different Transformer','Internal neural measurements have not yet been recorded for this Qwen example. The historical SmolVLM2 plots below are a separate experiment. They measure internal vector magnitudes and externally readable labels, not causal explanations of Qwen.'),
              ('results','What the experiments show','18 development cases, six families and one five-step pilot seed. CE+ES step five gets 9/18 correct, versus 7/18 for CE step five. The descriptive family-bootstrap interval includes zero. Two correct abstentions coexist with four incorrect abstentions. No CE+ES step-50 result exists.'),
              ('verify','Limits and reproducibility','Recorded behavior is OBSERVED; logical consequences are SYMBOLICALLY DERIVED; neural associations are CORRELATED. A relation-specific causal mechanism remains UNVERIFIED. These controlled stimuli are actual benchmark inputs; no synthetic tutorial results enter the metrics.')]
    featured={'input':'02_Visual_Inputs/xc_ho_0011__underdetermined.png','logic':'04_Logical_Evidence_Graphs/xc_ho_0011__underdetermined_Symbolic_Graph.png','answers':'03_Model_Output_Comparisons/xc_ho_0011__underdetermined_Input_Output.png','neural':'05_Neural_Activation_Analysis/Probe_Trajectory_last.png','results':'06_Experiment_Analytics/Confusion_Matrix_es5.png'}
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CrossModalLab — a guided evidence story</title><style>body{margin:0;background:#f5f7f2;color:#203e34;font:18px/1.7 Arial,sans-serif}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:42px;line-height:1.2}h2{font-size:30px;line-height:1.3}h3{font-size:22px}nav{display:flex;flex-wrap:wrap;gap:12px}nav a,a{color:#246b54}section{padding:30px 0;border-bottom:1px solid #dce5d8}figure{margin:24px 0;padding:24px;background:#fff;border:1px solid #dce5d8;border-radius:12px}img{display:block;width:100%;max-height:650px;object-fit:contain;background:#fafbf7}p{max-width:80ch}.limit{border-left:3px solid #a67c2f;padding-left:16px}.category{font-size:14px;font-weight:bold}code{overflow-wrap:anywhere}summary{cursor:pointer;padding:12px 0}a:focus-visible,summary:focus-visible{outline:3px solid #72578d;outline-offset:4px}.archive figure{break-inside:avoid}@media(max-width:600px){main{padding:24px 16px}figure{padding:16px}h1{font-size:32px}h2{font-size:25px}body{font-size:16px}}@media print{nav{display:none}body{background:white;font-size:12pt}main{max-width:none}figure{break-inside:avoid}img{max-height:18cm}details{display:block}}</style><main><p>CROSSMODALLAB · OCTOBER 28 RESEARCH DEMONSTRATION</p><h1>When should a vision-language model say “we don’t know”?</h1><p>Follow one question from its original input to the recorded answer, then explore the experiments and their limits. This gallery works entirely offline. Click any image to open it at full size.</p><nav aria-label="Research story">']
    html += [f'<a href="#{key}">{escape(title)}</a>' for key,title,_ in sections]
    html.append('</nav>')
    for key,title,intro in sections:
        html.append(f'<section id="{key}"><h2>{escape(title)}</h2><p>{escape(intro)}</p>')
        if key in featured:
            relative=featured[key]
            if relative not in catalog:raise ValueError('Missing featured evidence '+relative)
            html.append(card(relative,catalog[relative]))
        if key=='verify':html.append('<p><a href="08_Reproducibility/Exact_Recorded_Answers.json">Exact Qwen answers, prompts and sources</a> · <a href="05_Neural_Activation_Analysis/Neural_Evidence_Inventory.json">Historical neural evidence</a> · <a href="MANIFEST.json">Full provenance manifest</a> · <a href="09_Demonstration/human_centered_demo.md">Five-minute presentation guide</a> · <a href="09_Demonstration/Preliminary_Paper.html">Preliminary manuscript</a></p><p>Partner-machine verification, a human timed rehearsal and final submission requirements remain pending. No new training, model downloads or final holdout evaluation were used.</p>')
        html.append('</section>')
    html.append('<section class="archive"><h2>Explore all exported visuals</h2><p>The featured story above is the suggested reading path. These expandable collections retain every original export, with explanations and source references.</p>')
    names=['Dashboard in use','Original model inputs','Recorded answers by checkpoint','Logical evidence and compatible orders','Historical SmolVLM2 measurements','Development results and recorded costs','Preliminary paper figures']
    for index,title in enumerate(names,1):
        html.append(f'<details><summary>{escape(title)}</summary>')
        html.extend(card(r,i) for r,i in pngs if r.startswith(f'{index:02d}_'))
        html.append('</details>')
    html.append('</section><p>Scientific files are preserved with exact hashes. No model weights or optimizer state are included.</p></main></html>')
    return ''.join(html)
