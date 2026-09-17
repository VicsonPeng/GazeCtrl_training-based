#!/usr/bin/env python
"""Phase4 build: read all phase3 manifests, embed selected frames as base64,
emit a self-contained HTML review tool (Gaze Label Studio). Reviewer picks per
frame: use 3D / use 6D / set own dy / discard. Exports JSON of decisions."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, json, glob, base64, argparse
import cv2

P3 = RAW + "/phase3"
OUT = OUT + "/gaze_label_studio.html"
EMBED_W = 420    # downscale embedded review frames to this width
EMBED_Q = 82     # jpeg quality


BATCH_DIR = f'{P3}/batches'
OUT_DIR = OUT


def data_uri(path):
    im = cv2.imread(path)
    if im is not None:
        h, w = im.shape[:2]
        if w > EMBED_W:
            im = cv2.resize(im, (EMBED_W, int(round(h * EMBED_W / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, EMBED_Q])
        if ok:
            return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    with open(path, 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


def all_frame_keys():
    """Every selected frame across all manifests, stable order (tag, frame)."""
    keys = []
    for m in sorted(glob.glob(f'{P3}/manifest/*.json')):
        if 'DEMO' in m:
            continue
        man = json.load(open(m))
        for fr in man['frames']:
            keys.append(f"{man['tag']}|{fr['frame']}")
    return keys


def build_items(keys):
    """Load full items (with base64 image) for the given 'tag|frame' keys."""
    want = {}
    for k in keys:
        tag, fr = k.rsplit('|', 1); want.setdefault(tag, set()).add(int(fr))
    items = []
    for tag in sorted(want):
        mp = f'{P3}/manifest/{tag}.json'
        if not os.path.exists(mp):
            continue
        man = json.load(open(mp))
        for fr in man['frames']:
            if fr['frame'] not in want[tag]:
                continue
            ip = f'{P3}/{fr["img"]}'
            if not os.path.exists(ip):
                continue
            items.append(dict(tag=man['tag'], src=man['src_id'], mode=man['mode'],
                              frame=fr['frame'], img=data_uri(ip), head=fr.get('head'),
                              id_flag=fr.get('id_flag'), id_sim=fr.get('id_sim'), sharp=fr.get('sharp'),
                              used=fr['used'], g3=fr['g3'], d6=fr['d6'], chosen=fr['chosen']))
    for i, it in enumerate(items):
        it['i'] = i
    return items


def write_html(items, out, batch_label=''):
    title = f'Gaze Label · {batch_label}' if batch_label else 'Gaze Label Studio'
    html = (TEMPLATE.replace('/*__DATA__*/', json.dumps(items))
            .replace('/*__BATCH__*/', batch_label)
            .replace('<title>Gaze Label Studio</title>', f'<title>{title}</title>'))
    with open(out, 'w') as f:
        f.write(html)
    print(f'{len(items)} frames -> {out}')


def dedup_pool_keys():
    """Ordered 'tag|frame' keys from the per-id deduped id_final/ sets."""
    keys = []
    for f in sorted(glob.glob(f'{P3}/id_final/*.json')):
        for fr in json.load(open(f))['frames']:
            keys.append(f"{fr['tag']}|{fr['frame']}")
    return keys


def labeled_keys():
    """'tag|frame' already decided in a saved batch export."""
    ks = set()
    for f in glob.glob(f'{BATCH_DIR}/exports/*.json'):
        for d in json.load(open(f)):
            ks.add(f"{d['tag']}|{d['frame']}")
    return ks


def load_ledger():
    p = f'{BATCH_DIR}/ledger.json'
    return json.load(open(p)) if os.path.exists(p) else {}


def save_ledger(led):
    os.makedirs(BATCH_DIR, exist_ok=True)
    json.dump(led, open(f'{BATCH_DIR}/ledger.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='*', help='ad-hoc: build these tags to --out')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--make_next', action='store_true', help='cut the next unassigned batch')
    ap.add_argument('--batch', type=int, help='rebuild an existing batch id')
    ap.add_argument('--batch_size', type=int, default=200)
    ap.add_argument('--dedup', action='store_true', help='draw from per-id deduped id_final/, skipping already-labeled frames')
    ap.add_argument('--align_id', action='store_true', help='never split an id across batches (batch size becomes approximate)')
    ap.add_argument('--keys_file', help='build a new batch from an explicit JSON list of frame keys (for 补标)')
    args = ap.parse_args()

    if args.keys_file:
        led = load_ledger()
        keys = json.load(open(args.keys_file))
        bid = (max((int(b) for b in led), default=0) + 1)
        led[str(bid)] = keys; save_ledger(led)
        out = f'{OUT_DIR}/gaze_label_batch_{bid:02d}.html'
        write_html(build_items(keys), out, f'批次 {bid}')
        print(f'BATCH {bid}: {len(keys)} frames -> {out}')
        return

    if args.make_next or args.batch is not None:
        led = load_ledger()
        if args.make_next:
            assigned = {k for ks in led.values() for k in ks}
            src = dedup_pool_keys() if args.dedup else all_frame_keys()
            skip = assigned | (labeled_keys() if args.dedup else set())
            pool = [k for k in src if k not in skip]
            if not pool:
                print('no unassigned frames left'); return
            bid = (max((int(b) for b in led), default=0) + 1)
            if args.align_id:
                # fill whole ids (never split an id across batches)
                from collections import OrderedDict
                byid = OrderedDict()
                for k in pool:
                    byid.setdefault(k.rsplit('__', 1)[0], []).append(k)
                keys = []
                for sid, ks in byid.items():
                    if keys and len(keys) + len(ks) > args.batch_size:
                        break
                    keys += ks
            else:
                keys = pool[:args.batch_size]
            led[str(bid)] = keys; save_ledger(led)
        else:
            bid = args.batch; keys = led.get(str(bid))
            if not keys:
                print(f'batch {bid} not in ledger'); return
        out = f'{OUT_DIR}/gaze_label_batch_{bid:02d}.html'
        write_html(build_items(keys), out, f'批次 {bid}')
        print(f'BATCH {bid}: {len(keys)} frames -> {out}')
        return

    # ad-hoc build (all or --tags)
    keys = all_frame_keys()
    if args.tags:
        keys = [k for k in keys if k.rsplit('|', 1)[0] in args.tags]
    write_html(build_items(keys), args.out)


TEMPLATE = r"""<title>Gaze Label Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#12141a; --panel:#1a1e27; --panel2:#20252f; --line:#2c3340;
  --ink:#e7ebf2; --mut:#8b93a5; --dim:#5c6474;
  --c3d:#ff6a5f; --c6d:#37cfe6; --cus:#f6b84a; --keep:#54d38a; --disc:#6b7280;
  --accent:#7b8cff;
  font-family:'IBM Plex Sans',system-ui,sans-serif;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);height:100vh;overflow:hidden}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums}
#app{display:grid;grid-template-columns:1fr 372px;grid-template-rows:auto 1fr;height:100vh}
header{grid-column:1/3;display:flex;align-items:center;gap:18px;padding:10px 18px;
  border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;font-weight:600;letter-spacing:.02em;margin:0}
.batchlbl{font-size:11px;color:var(--accent);border:1px solid var(--line);border-radius:20px;padding:2px 10px;font-weight:600}
header .sub{font-size:12px;color:var(--mut)}
.bar{flex:1;height:6px;border-radius:3px;background:var(--panel2);overflow:hidden;max-width:340px}
.bar>i{display:block;height:100%;background:var(--keep);width:0}
.counts{font-size:12px;color:var(--mut);display:flex;gap:14px}
.counts b{color:var(--ink);font-weight:600}
.counts .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:1px}
#exportBtn{margin-left:auto;background:var(--panel2);color:var(--ink);border:1px solid var(--line);
  padding:6px 13px;border-radius:7px;font:inherit;font-size:12px;cursor:pointer}
#exportBtn:hover{border-color:var(--accent)}

/* stage */
#stage{position:relative;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(120% 120% at 50% 0%,#171b23 0%,#0e1015 100%);overflow:hidden}
#wrap{position:relative;height:min(88%,760px);aspect-ratio:512/768}
#frame{height:100%;display:block;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,.55)}
#ov{position:absolute;inset:0;width:100%;height:100%}
#nav{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);display:flex;gap:10px;
  background:rgba(16,18,24,.82);border:1px solid var(--line);border-radius:10px;padding:6px 10px;font-size:12px;color:var(--mut)}
#nav button{background:none;border:none;color:var(--ink);font:inherit;cursor:pointer;padding:2px 8px;border-radius:6px}
#nav button:hover{background:var(--panel2)}
.tag{position:absolute;top:14px;left:14px;background:rgba(16,18,24,.82);border:1px solid var(--line);
  border-radius:8px;padding:6px 11px;font-size:11px;color:var(--mut)}
.tag b{color:var(--ink);font-weight:600}
.badge{display:inline-block;padding:1px 8px;border-radius:20px;font-size:10.5px;font-weight:600;letter-spacing:.02em}
.badge.warn{background:rgba(246,184,74,.2);color:#f6b84a;border:1px solid rgba(246,184,74,.5)}
.badge.okid{background:rgba(84,211,138,.16);color:#54d38a}
#stage.flagwarn #wrap{box-shadow:0 0 0 2px var(--cus),0 8px 40px rgba(0,0,0,.55)}
#stage.flagwarn #wrap::after{content:'⚠ 身份待确认 — 请重点核对';position:absolute;top:-26px;left:0;right:0;
  text-align:center;color:var(--cus);font-size:11px;font-weight:600}

/* panel */
#panel{border-left:1px solid var(--line);background:var(--panel);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:13px}
.sect-l{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
.est{border:1px solid var(--line);border-radius:9px;padding:11px 12px;cursor:pointer;background:var(--panel2);transition:border-color .1s}
.est:hover{border-color:var(--mut)}
.est.pick{border-color:var(--sel);box-shadow:inset 0 0 0 1px var(--sel)}
.est .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.est .nm{font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:7px}
.est .key{font-size:10px;color:var(--dim);border:1px solid var(--line);border-radius:5px;padding:1px 6px}
.est .val{font-size:16px;letter-spacing:.01em}
.est .na{color:var(--dim);font-style:italic;font-size:13px}
.est .comp{font-size:11px;color:var(--mut);margin-top:3px}
.swatch{width:11px;height:11px;border-radius:3px;display:inline-block}
.custom{border:1px solid var(--line);border-radius:9px;padding:11px 12px;background:var(--panel2)}
.custom.pick{border-color:var(--cus);box-shadow:inset 0 0 0 1px var(--cus)}
.custom .axis{display:grid;grid-template-columns:58px 1fr 52px;align-items:center;gap:8px;margin-top:8px}
.custom .axl{font-size:11px;color:var(--mut)}
.custom input[type=range]{width:100%;accent-color:var(--cus)}
.custom .av{font-size:12.5px;color:var(--cus);text-align:right}
.custom .prefill{display:flex;gap:8px;margin-top:10px}
.custom .prefill button{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--mut);
  border-radius:6px;padding:4px 6px;font:inherit;font-size:11px;cursor:pointer}
.custom .prefill button:hover{border-color:var(--cus);color:var(--ink)}
.disc{border:1px solid var(--line);border-radius:9px;padding:9px 12px;background:var(--panel2);cursor:pointer;
  display:flex;justify-content:space-between;align-items:center;font-size:12.5px;color:var(--mut)}
.disc:hover{border-color:var(--disc)}
.disc.pick{border-color:var(--disc);color:var(--ink);box-shadow:inset 0 0 0 1px var(--disc)}
.hint{font-size:11px;color:var(--dim);line-height:1.7;border-top:1px solid var(--line);padding-top:10px;margin-top:auto}
.hint kbd{font-family:'IBM Plex Mono',monospace;background:var(--panel2);border:1px solid var(--line);
  border-radius:4px;padding:0 5px;color:var(--mut);font-size:10.5px}
/* modal */
#modal{position:fixed;inset:0;background:rgba(8,9,12,.72);display:none;align-items:center;justify-content:center;z-index:9}
#modal.on{display:flex}
#modal .box{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;width:min(680px,90vw)}
#modal h3{margin:0 0 8px;font-size:14px}
#modal p{margin:0 0 10px;font-size:12px;color:var(--mut)}
#modal textarea{width:100%;height:300px;background:var(--bg);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;padding:10px;resize:vertical}
#modal .btns{display:flex;gap:8px;margin-top:10px;justify-content:flex-end}
#modal button{background:var(--panel2);border:1px solid var(--line);color:var(--ink);border-radius:7px;padding:6px 13px;font:inherit;font-size:12px;cursor:pointer}
#modal button.pri{background:var(--accent);border-color:var(--accent);color:#0b0d14;font-weight:600}
</style>

<div id="app">
  <header>
    <h1>Gaze Label Studio</h1>
    <span class="batchlbl">/*__BATCH__*/</span>
    <span class="sub" id="pos"></span>
    <div class="bar"><i id="barfill"></i></div>
    <div class="counts">
      <span><span class="dot" style="background:var(--keep)"></span><b id="cKeep">0</b> kept</span>
      <span><span class="dot" style="background:var(--disc)"></span><b id="cDisc">0</b> discard</span>
      <span><b id="cTodo">0</b> left</span>
    </div>
    <button id="exportBtn">导出 JSON</button>
  </header>

  <section id="stage">
    <div class="tag" id="tag"></div>
    <div id="wrap"><img id="frame" alt="frame"><canvas id="ov" width="512" height="768"></canvas></div>
    <div id="nav">
      <button id="prev">← 上一张</button>
      <span id="navpos" class="mono"></span>
      <button id="next">下一张 →</button>
    </div>
  </section>

  <aside id="panel">
    <div class="sect-l">选择这一帧的 gaze 来源</div>
    <div class="est" id="e3d" data-c="3d">
      <div class="top"><span class="nm"><span class="swatch" style="background:var(--c3d)"></span>3DGazeNet</span><span class="key">1</span></div>
      <div class="val mono" id="v3d"></div><div class="comp" id="c3d_c"></div>
    </div>
    <div class="est" id="e6d" data-c="6d">
      <div class="top"><span class="nm"><span class="swatch" style="background:var(--c6d)"></span>6DRepNet</span><span class="key">2</span></div>
      <div class="val mono" id="v6d"></div><div class="comp" id="c6d_c"></div>
    </div>
    <div class="custom" id="ecus" data-c="custom">
      <div class="top" style="display:flex;justify-content:space-between"><span class="nm"><span class="swatch" style="background:var(--cus)"></span>自设 dx / dy / dz（自动归一化）</span><span class="key">3</span></div>
      <div class="axis"><span class="axl">dx 左右</span><input type="range" class="cs" id="sdx" min="-1" max="1" step="0.02" value="0"><span class="mono av" id="adx">+0.00</span></div>
      <div class="axis"><span class="axl">dy 上下</span><input type="range" class="cs" id="sdy" min="-1" max="1" step="0.02" value="0"><span class="mono av" id="ady">+0.00</span></div>
      <div class="axis"><span class="axl">dz 前后</span><input type="range" class="cs" id="sdz" min="-1" max="1" step="0.02" value="1"><span class="mono av" id="adz">+1.00</span></div>
      <div class="prefill"><button id="pf6d">← 从 6D 载入</button><button id="pf3d">← 从 3D 载入</button></div>
      <div class="comp mono" id="cusvec" style="margin-top:7px"></div>
    </div>
    <div class="disc" id="ediscard" data-c="discard"><span>丢弃这一帧（生成不佳 / 无法判断）</span><span class="key">D</span></div>
    <div class="hint">
      <kbd>1</kbd>用3D &nbsp; <kbd>2</kbd>用6D &nbsp; <kbd>3</kbd>自设 &nbsp; <kbd>D</kbd>丢弃<br>
      <kbd>←</kbd><kbd>→</kbd>上下张 &nbsp; <kbd>Enter</kbd>确认并下一张 &nbsp; 决定后自动前进<br>
      进度自动保存在本机浏览器；随时「导出 JSON」贴回给我。
    </div>
  </aside>
</div>

<div id="modal"><div class="box">
  <h3>导出决定</h3>
  <p>已全选，Ctrl/Cmd+C 复制，贴回给我即可并入训练集。</p>
  <textarea id="exptext" readonly></textarea>
  <div class="btns"><button id="mclose">关闭</button><button class="pri" id="mcopy">复制</button></div>
</div></div>

<script>
const DATA=/*__DATA__*/;
const KEY='gls_decisions_v1';
let dec={}; try{dec=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let idx=0;
const $=id=>document.getElementById(id);
const img=$('frame'), cv=$('ov'), ctx=cv.getContext('2d');

function fmt(v){return v?`${v[0]>=0?'+':''}${v[0].toFixed(2)}, ${v[1]>=0?'+':''}${v[1].toFixed(2)}, ${v[2]>=0?'+':''}${v[2].toFixed(2)}`:''}
function yawstr(v){if(!v)return '';const y=Math.atan2(v[0],v[2])*180/Math.PI;const p=Math.asin(Math.max(-1,Math.min(1,v[1])))*180/Math.PI;return `yaw ${y>=0?'+':''}${y.toFixed(0)}° · pitch ${p>=0?'+':''}${p.toFixed(0)}°`}
// custom: free dx/dy/dz, normalized to a unit vector
function norm3(a,b,c){ const m=Math.hypot(a,b,c)||1; return [a/m,b/m,c/m]; }
function customVec(d){ return norm3(d.dx,d.dy,d.dz); }
function decVec(it,d){ if(!d)return null; if(d.choice==='3d')return it.g3; if(d.choice==='6d')return it.d6;
  if(d.choice==='custom')return customVec(d); return null; }

function drawArrows(){
  ctx.clearRect(0,0,512,768); const it=DATA[idx];
  const h=it.head||[256,150,150]; const ox=h[0], oy=h[1], L=Math.max(70,h[2]*1.3);
  function arrow(v,color,w){ if(!v)return; const dx=v[0], dy=-v[1];
    const m=Math.hypot(dx,dy)||1e-6; const ex=ox+dx/m*L*Math.min(1,Math.hypot(v[0],v[1])+0.15), ey=oy+dy/m*L*Math.min(1,Math.hypot(v[0],v[1])+0.15);
    ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=w; ctx.lineCap='round';
    ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(ex,ey); ctx.stroke();
    const a=Math.atan2(ey-oy,ex-ox), hl=13;
    ctx.beginPath(); ctx.moveTo(ex,ey);
    ctx.lineTo(ex-hl*Math.cos(a-0.4),ey-hl*Math.sin(a-0.4));
    ctx.lineTo(ex-hl*Math.cos(a+0.4),ey-hl*Math.sin(a+0.4)); ctx.closePath(); ctx.fill();
  }
  ctx.beginPath(); ctx.arc(ox,oy,4,0,7); ctx.fillStyle='#fff'; ctx.fill();
  const d=dec[it.i];
  if(d&&d.choice==='custom'){ arrow(customVec(d),'#f6b84a',5); }
  else { arrow(it.g3, '#ff6a5f',4); arrow(it.d6, '#37cfe6',4); }
}

function render(){
  const it=DATA[idx];
  img.onload=drawArrows; img.src=it.img;
  let idb='';
  if(it.id_flag==='low') idb=`<span class="badge warn">⚠ 侧脸·身份待确认${it.id_sim!=null?' '+it.id_sim:''}</span>`;
  else if(it.id_flag==='noface') idb=`<span class="badge warn">⚠ 转身·看不到脸·待确认</span>`;
  else if(it.id_flag==='ok') idb=`<span class="badge okid">ID ✓${it.id_sim!=null?' '+it.id_sim:''}</span>`;
  const sh=it.sharp!=null?` · <span style="color:var(--mut)">sharp ${it.sharp}</span>`:'';
  $('tag').innerHTML=`<b>${it.src}</b> · mode <b>${it.mode}</b> · frame <b>${it.frame}</b>${sh} ${idb}`;
  $('stage').classList.toggle('flagwarn', it.id_flag==='low'||it.id_flag==='noface');
  $('pos').textContent=`${idx+1} / ${DATA.length}`;
  $('navpos').textContent=`${idx+1} / ${DATA.length}`;
  $('barfill').style.width=(100*Object.keys(dec).length/DATA.length)+'%';
  // estimator readouts
  $('v3d').innerHTML=it.g3?fmt(it.g3):'<span class="na">此帧无 3D（侧脸/遮挡）</span>';
  $('c3d_c').textContent=it.g3?yawstr(it.g3):'';
  $('v6d').innerHTML=it.d6?fmt(it.d6):'<span class="na">此帧无 6D</span>';
  $('c6d_c').textContent=it.d6?yawstr(it.d6):'';
  const d=dec[it.i];
  const base=(d&&d.choice==='custom')?[d.dx,d.dy,d.dz]:(it.d6||it.g3||it.chosen);
  setSliders(base);
  // highlight
  for(const [el,c,col] of [['e3d','3d','--c3d'],['e6d','6d','--c6d'],['ecus','custom','--cus'],['ediscard','discard','--disc']]){
    const on=d&&d.choice===c; const e=$(el); e.classList.toggle('pick',!!on);
    if(el!=='ediscard') e.style.setProperty('--sel',getComputedStyle(document.body).getPropertyValue(col));
  }
  $('e3d').style.opacity=it.g3?1:.4; $('e6d').style.opacity=it.d6?1:.4;
  updateCounts(); drawArrows();
}
function updateCounts(){
  let k=0,di=0; for(const i in dec){ dec[i].choice==='discard'?di++:k++; }
  $('cKeep').textContent=k; $('cDisc').textContent=di; $('cTodo').textContent=DATA.length-k-di;
}
function readSliders(){ return [parseFloat($('sdx').value),parseFloat($('sdy').value),parseFloat($('sdz').value)]; }
function setSliders(v){
  $('sdx').value=v[0]; $('sdy').value=v[1]; $('sdz').value=v[2];
  $('adx').textContent=(v[0]>=0?'+':'')+(+v[0]).toFixed(2);
  $('ady').textContent=(v[1]>=0?'+':'')+(+v[1]).toFixed(2);
  $('adz').textContent=(v[2]>=0?'+':'')+(+v[2]).toFixed(2);
  $('cusvec').textContent='归一化 → '+fmt(norm3(v[0],v[1],v[2]))+'　·　'+yawstr(norm3(v[0],v[1],v[2]));
}
function setChoice(c){
  const it=DATA[idx];
  if(c==='custom'){ const v=readSliders(); dec[it.i]={choice:'custom',dx:v[0],dy:v[1],dz:v[2]}; }
  else dec[it.i]={choice:c};
  save(); render(); if(c!=='custom') setTimeout(next,120);
}
function save(){ try{localStorage.setItem(KEY,JSON.stringify(dec))}catch(e){} }
function next(){ if(idx<DATA.length-1){idx++;render();} }
function prev(){ if(idx>0){idx--;render();} }

$('e3d').onclick=()=>DATA[idx].g3&&setChoice('3d');
$('e6d').onclick=()=>DATA[idx].d6&&setChoice('6d');
$('ecus').onclick=e=>{ if(!e.target.classList.contains('cs')&&e.target.tagName!=='BUTTON') setChoice('custom'); };
$('ediscard').onclick=()=>setChoice('discard');
function onSlide(){ const it=DATA[idx]; const v=readSliders(); setSliders(v);
  dec[it.i]={choice:'custom',dx:v[0],dy:v[1],dz:v[2]}; save();
  for(const [el,c] of [['e3d','3d'],['e6d','6d'],['ecus','custom'],['ediscard','discard']]) $(el).classList.toggle('pick',c==='custom');
  drawArrows(); updateCounts(); $('barfill').style.width=(100*Object.keys(dec).length/DATA.length)+'%'; }
['sdx','sdy','sdz'].forEach(id=>$(id).oninput=onSlide);
$('pf6d').onclick=()=>{ if(DATA[idx].d6){ setSliders(DATA[idx].d6); onSlide(); } };
$('pf3d').onclick=()=>{ if(DATA[idx].g3){ setSliders(DATA[idx].g3); onSlide(); } };
$('next').onclick=next; $('prev').onclick=prev;
document.addEventListener('keydown',e=>{
  if($('modal').classList.contains('on'))return;
  const sl=document.activeElement&&document.activeElement.classList.contains('cs');
  if(e.key==='1')setChoice('3d'); else if(e.key==='2')setChoice('6d');
  else if(e.key==='3')setChoice('custom'); else if(e.key.toLowerCase()==='d')setChoice('discard');
  else if(e.key==='Enter')next();
  else if((e.key==='ArrowRight')&&!sl)next(); else if((e.key==='ArrowLeft')&&!sl)prev();
});
// export
$('exportBtn').onclick=()=>{
  const out=[]; for(const it of DATA){ const d=dec[it.i]; if(!d)continue;
    if(d.choice==='discard'){ out.push({tag:it.tag,src:it.src,mode:it.mode,frame:it.frame,choice:'discard'}); continue; }
    const v=decVec(it,d); out.push({tag:it.tag,src:it.src,mode:it.mode,frame:it.frame,img:it.img.slice(0,0)+`frames/${it.tag}/f${String(it.frame).padStart(3,'0')}.jpg`,
      choice:d.choice,dx:+v[0].toFixed(4),dy:+v[1].toFixed(4),dz:+v[2].toFixed(4)}); }
  $('exptext').value=JSON.stringify(out,null,1); $('modal').classList.add('on');
  setTimeout(()=>{$('exptext').focus();$('exptext').select();},50);
};
$('mclose').onclick=()=>$('modal').classList.remove('on');
$('mcopy').onclick=()=>{$('exptext').select();document.execCommand('copy');};
render();
</script>
"""

if __name__ == '__main__':
    main()
