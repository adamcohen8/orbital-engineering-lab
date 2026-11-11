"""Offline interactive placement review. No CDN, network requests, or executable source text."""

from __future__ import annotations

import json
from pathlib import Path

from sim.security.output_writes import write_output_text


def write_viewer(path, model, assessment):
    mesh = []
    geometry = model.get("geometry", {})
    if geometry and assessment.get("runtime_compiled"):
        import numpy as np

        from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, load_stl_triangles

        from .contracts import contained, physical_output_path

        root = Path(path).parent
        profile = GeometryAreaProfile.load(physical_output_path(contained(root, geometry["profile_path"])))
        meta = profile.metadata
        for source in model.get("sources", []):
            if source["sha256"] == meta.get("mesh_sha256"):
                triangles, _ = load_stl_triangles(physical_output_path(contained(root, source["path"])))
                stride = max(1, int(np.ceil(len(triangles) / 3000)))
                triangles = (triangles[::stride] * meta["scale_to_m"]) @ np.asarray(meta["rotation_to_body"]).T + meta[
                    "translation_body_m"
                ]
                mesh = triangles.tolist()
                break
    payload = (
        json.dumps({"model": model, "assessment": assessment, "mesh": mesh}, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OEL Spacecraft Twin Review</title><style>
body{background:#101722;color:#e5edf6;font:16px system-ui;margin:24px}main{display:flex;gap:24px;flex-wrap:wrap}canvas{background:#172335;border:1px solid #53647a;max-width:100%;touch-action:none}aside{max-width:540px}button{padding:9px;margin:4px;background:#263c55;color:white;border:1px solid #8193aa;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}select{padding:8px}h1{font-size:26px}</style>
<h1>Spacecraft Twin Review</h1><p id="subtitle"></p><p>Drag to rotate; use the wheel to zoom. Select a component to inspect placement and source citations. Axes: X red, Y green, Z blue. COM is gold.</p>
<main><section><canvas id="scene" width="720" height="520" aria-label="Rotatable spacecraft component placement diagram"></canvas><div id="buttons"></div></section><aside><h2 id="heading">Readiness</h2><pre id="detail"></pre></aside></main>
<script type="application/json" id="data">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent), m=data.model, a=data.assessment;
const mesh=data.mesh||[], placements=a.compiled?.placements||[], canvas=document.getElementById('scene'),ctx=canvas.getContext('2d');
let yaw=.55,pitch=.4,zoom=1,drag=null,selected=null;
const com=m.center_of_mass_body_m||[0,0,0];
const extent=Math.max(.5,...placements.map(p=>Math.hypot(...p.position_body_m)),Math.hypot(...com),...mesh.flat().map(v=>Math.hypot(...v)));
function project(v){let x=Math.cos(yaw)*v[0]+Math.sin(yaw)*v[2],z=-Math.sin(yaw)*v[0]+Math.cos(yaw)*v[2],y=Math.cos(pitch)*v[1]-Math.sin(pitch)*z;return [360+x*170/extent*zoom,260-y*170/extent*zoom]}
function line(a,b,color,label){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();let angle=Math.atan2(q[1]-p[1],q[0]-p[0]);ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(q[0]-9*Math.cos(angle-.4),q[1]-9*Math.sin(angle-.4));ctx.lineTo(q[0]-9*Math.cos(angle+.4),q[1]-9*Math.sin(angle+.4));ctx.fill();ctx.fillText(label,q[0]+7,q[1]-7)}
function dot(p,color,label){const q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...q,5,0,Math.PI*2);ctx.fill();ctx.fillText(label,q[0]+8,q[1]+15)}
function draw(){ctx.clearRect(0,0,720,520);ctx.font='14px system-ui';ctx.strokeStyle='#354d66';ctx.lineWidth=.6;for(const tri of mesh){ctx.beginPath();tri.forEach((v,i)=>{const p=project(v);if(i)ctx.lineTo(...p);else ctx.moveTo(...p)});ctx.closePath();ctx.stroke()}[['X','#ff8585'],['Y','#83e4a4'],['Z','#87bfff']].forEach(([n,c],i)=>{let v=[0,0,0];v[i]=extent;line([0,0,0],v,c,n)});dot([0,0,0],'white','Body origin');dot(com,'#ffd46c','COM');for(const p of placements){let color=p.id===selected?'#fff':'#81d7df';dot(p.position_body_m,color,selected===p.id?p.id:'');line(p.position_body_m,p.position_body_m.map((x,i)=>x+p.axis_body[i]*extent*.3),color,selected===p.id?p.axis_convention:'')}}
function show(p){selected=p?.id||null;document.getElementById('heading').textContent=p?p.id:'Readiness';document.getElementById('detail').textContent=JSON.stringify(p?{placement:p,evidence:Object.fromEntries(Object.entries(m.evidence||{}).filter(([k])=>k.startsWith('components/'+p.id+'/'))),sources:m.sources}: {ready:a.ready,errors:a.errors,missing:a.missing,warnings:a.warnings,geometry:m.geometry,assumptions:m.assumptions,conflicts:m.conflicts},null,2);draw()}
for(const p of [null,...placements]){let b=document.createElement('button');b.textContent=p?p.id:'Readiness';b.onclick=()=>show(p);document.getElementById('buttons').append(b)}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw()};canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.getElementById('subtitle').textContent=m.id+' / '+m.version+' — '+(a.ready?'Ready for '+a.purpose:'Review required')+(mesh.length?' — source mesh wireframe with component layout':' — component layout; no verified mesh available');show(null);
</script></html>"""
    write_output_text(path, html.replace("PAYLOAD", payload), encoding="utf-8")
