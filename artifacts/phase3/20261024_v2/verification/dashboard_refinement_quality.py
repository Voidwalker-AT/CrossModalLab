from pathlib import Path
import json
root=Path.cwd(); data=json.loads((root/'artifacts/training/phase_2i_b_pilot_20261023/preflight_and_plan.json').read_text())
import hashlib
for name,h in data['history_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
print('PILOT_HISTORY_40_FILES_UNCHANGED')
def lum(s):
 a=[int(s[i:i+2],16)/255 for i in (0,2,4)];a=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in a];return sum(v*w for v,w in zip(a,(.2126,.7152,.0722)))
for fg,bg in [('5b6b5d','ffffff'),('805a20','fbf2df'),('83503d','f8eae5'),('694c85','ffffff')]:
 a,b=sorted((lum(fg),lum(bg)));ratio=(b+.05)/(a+.05);assert ratio>=4.5;print('TEXT_CONTRAST',fg,bg,round(ratio,2))
