"""Bundle the 80k and 90k sweeps into one page: the galleries are identical across
checkpoints, so they are hoisted out and shipped once."""
import json
from pathlib import Path
import sys, os
_H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_H))
from gaze_paths import OUT
D=Path(OUT)
SETS=[('fullrot_v2_data.json','80k'),('fullrot_v2_85k_data.json','85k'),('fullrot_v2_90k_data.json','90k')]
bundle=None; sets=[]
for f,_ in SETS:
    p=D/f
    if not p.exists(): print("skip (missing)",f); continue
    j=json.load(open(p))
    if bundle is None:
        bundle={"yaws":j["yaws"],"pitches":j["pitches"],"cell":j["cell"],"galleries":j["galleries"],"sets":[]}
    sets.append({"ckpt":j["ckpt"],"cn_scale":j["cn_scale"],"steps":j["steps"],"seed":j["seed"],"sources":j["sources"]})
bundle["sets"]=sets
out=D/'gazectrl_demo_v2.html'
out.write_text((D/'_demo_v2_template.html').read_text().replace('/*__DATA__*/',json.dumps(bundle,separators=(',',':'))))
print(f"SAVED {out}  {out.stat().st_size/1e6:.2f} MB  | checkpoints: {[s['ckpt'] for s in sets]}")
