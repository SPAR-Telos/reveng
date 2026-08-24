#!/usr/bin/env python3
"""Serve a small local UI for independent reasoning-DAG edge review."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


DEFAULT_ROOT = Path("outputs/hypothesis_tests/deterministic_reasoning_dag_audit_v1")
EDGE_LABELS = (
    "infer", "restate", "support", "attack", "proceed", "execute",
    "decompose", "verify", "backtrack", "positive", "negative", "uncertain",
    "state_update", "route_update", "action_commit",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reviewer", choices=("a", "b"), required=True)
    parser.add_argument("--calibration-nodes", type=int, default=15)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Reasoning DAG review</title>
<style>
body{font:15px system-ui;margin:0;background:#f5f7f9;color:#17212b} header{background:#18364b;color:white;padding:14px 24px}
main{max-width:1050px;margin:18px auto;padding:0 18px}.card{background:white;border:1px solid #d6dde3;border-radius:9px;padding:16px;margin-bottom:14px}
button,select{font:inherit;padding:7px 10px}.target{font-size:18px;white-space:pre-wrap}.source{display:grid;grid-template-columns:34px 65px 1fr 165px;gap:8px;align-items:start;padding:8px 0;border-top:1px solid #e8ecef}
.source-text{white-space:pre-wrap}.muted{color:#667784}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.ok{color:#19713b}.error{color:#a32626}.context{max-height:480px;overflow:auto}
</style></head><body><header><b>Typed reasoning-DAG calibration</b> — Reviewer __REVIEWER__</header>
<main><div class="card bar"><label>Trace <select id="trace"></select></label><label>Destination <select id="dest"></select></label><span id="progress"></span></div>
<div class="card"><div class="muted" id="meta"></div><p class="target" id="target"></p><div id="labels" class="muted"></div></div>
<div class="card"><b>Select every earlier sentence needed to interpret the destination.</b><p class="muted">Select a relation for each checked source. Proximity alone is not a reason for an edge.</p><label><input type="checkbox" id="none"> Reviewed: this node has no predecessors</label><div class="context" id="sources"></div></div>
<div class="card bar"><button id="prev">Previous</button><button id="save">Save and next</button><span id="status"></span></div></main>
<script>
let state, current, traceKey;
async function load(){state=await (await fetch('/api/state')).json();let t=document.querySelector('#trace');t.innerHTML=state.traces.map(x=>`<option value="${x.key}">${x.label}</option>`).join('');traceKey=t.value;t.onchange=()=>{traceKey=t.value;populateDest()};populateDest()}
function candidates(){return state.nodes.filter(x=>x.trace_key===traceKey&&x.position_index<current.position_index)}
function populateDest(){let ds=state.reviewable.filter(x=>x.trace_key===traceKey);let d=document.querySelector('#dest');d.innerHTML=ds.map(x=>`<option value="${x.node_id}">S${String(x.position_index).padStart(3,'0')} ${x.reviewed?'✓':'○'} ${escapeHtml(x.target_sentence.slice(0,70))}</option>`).join('');d.onchange=()=>show(d.value);show(d.value)}
function show(id){current=state.reviewable.find(x=>x.node_id===id);document.querySelector('#meta').textContent=`${current.trace_label} · sentence ${current.position_index}`;document.querySelector('#target').textContent=current.target_sentence;document.querySelector('#labels').textContent=`Existing semantic labels: ${current.semantic_labels}`;document.querySelector('#progress').textContent=`${state.reviewable.filter(x=>x.reviewed).length}/${state.reviewable.length} calibration nodes reviewed`;let existing=current.edges===null?null:current.edges;document.querySelector('#none').checked=Array.isArray(existing)&&existing.length===0;let selected=new Map((existing||[]).map(x=>[x.source_node_id,x.edge_label]));document.querySelector('#sources').innerHTML=candidates().reverse().map(x=>`<div class="source"><input type="checkbox" class="pick" data-id="${x.node_id}" ${selected.has(x.node_id)?'checked':''}><b>S${String(x.position_index).padStart(3,'0')}</b><span class="source-text">${escapeHtml(x.target_sentence)}</span><select class="edge-label" data-id="${x.node_id}">${state.edge_labels.map(y=>`<option ${selected.get(x.node_id)===y?'selected':''}>${y}</option>`).join('')}</select></div>`).join('');document.querySelector('#status').textContent=''}
function escapeHtml(x){return x.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function save(){let picks=[...document.querySelectorAll('.pick:checked')];let none=document.querySelector('#none').checked;if(!picks.length&&!none){setStatus('Select a predecessor or explicitly mark no predecessors.',true);return}if(picks.length&&none){setStatus('Cannot select edges and no predecessors together.',true);return}let edges=picks.map(x=>({source_node_id:x.dataset.id,edge_label:document.querySelector(`.edge-label[data-id="${x.dataset.id}"]`).value}));let response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({node_id:current.node_id,edges})});let result=await response.json();if(!response.ok){setStatus(result.error||'Save failed',true);return}current.edges=edges;current.reviewed=true;setStatus('Saved',false);let ds=state.reviewable.filter(x=>x.trace_key===traceKey),i=ds.findIndex(x=>x.node_id===current.node_id);populateDest();if(i+1<ds.length){document.querySelector('#dest').value=ds[i+1].node_id;show(ds[i+1].node_id)}}
function setStatus(x,error){let s=document.querySelector('#status');s.textContent=x;s.className=error?'error':'ok'}
document.querySelector('#save').onclick=save;document.querySelector('#prev').onclick=()=>{let ds=state.reviewable.filter(x=>x.trace_key===traceKey),i=ds.findIndex(x=>x.node_id===current.node_id);if(i>0){document.querySelector('#dest').value=ds[i-1].node_id;show(ds[i-1].node_id)}};load();
</script></body></html>'''


def application(args: argparse.Namespace):
    review_path = args.review_dir / "edge_review.csv"
    reviewer_column = f"reviewer_{args.reviewer}_edges_json"

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/":
                body = HTML.replace("__REVIEWER__", args.reviewer.upper()).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if urlparse(self.path).path != "/api/state": self.send_error(404); return
            frame = pd.read_csv(review_path, keep_default_na=False).sort_values(["example_id", "position_index"])
            examples = frame.example_id.drop_duplicates().tolist(); aliases = {value:f"trace_{i+1}" for i,value in enumerate(examples)}
            nodes=[]; reviewable=[]
            for row in frame.itertuples(index=False):
                raw=getattr(row,reviewer_column); edges=json.loads(raw) if str(raw).strip() else None
                item={"node_id":row.node_id,"trace_key":aliases[row.example_id],"trace_label":f"Trace {examples.index(row.example_id)+1}","position_index":int(row.position_index),"target_sentence":str(row.target_sentence),"semantic_labels":str(row.semantic_labels),"edges":edges,"reviewed":edges is not None}
                nodes.append(item)
            for example in examples:
                reviewable.extend([x for x in nodes if x["trace_key"]==aliases[example]][:args.calibration_nodes])
            self.send_json({"traces":[{"key":aliases[x],"label":f"Trace {i+1}"} for i,x in enumerate(examples)],"nodes":nodes,"reviewable":reviewable,"edge_labels":EDGE_LABELS})

        def do_POST(self):
            if urlparse(self.path).path != "/api/save": self.send_error(404); return
            try:
                length=int(self.headers.get("Content-Length","0")); payload=json.loads(self.rfile.read(length)); node_id=payload["node_id"]; edges=payload["edges"]
                if not isinstance(edges,list): raise ValueError("edges must be a list")
                frame=pd.read_csv(review_path,keep_default_na=False); matches=frame.index[frame.node_id.eq(node_id)]
                if len(matches)!=1: raise ValueError("unknown destination node")
                destination=frame.loc[matches[0]]; seen=set()
                for edge in edges:
                    source=edge.get("source_node_id"); label=edge.get("edge_label")
                    if label not in EDGE_LABELS: raise ValueError(f"invalid edge label: {label}")
                    source_rows=frame[frame.node_id.eq(source)]
                    if len(source_rows)!=1: raise ValueError("unknown source node")
                    source_row=source_rows.iloc[0]
                    if source_row.example_id!=destination.example_id or int(source_row.position_index)>=int(destination.position_index): raise ValueError("source must be earlier in the same trace")
                    if source in seen: raise ValueError("duplicate source"); seen.add(source)
                frame.loc[matches[0],reviewer_column]=json.dumps(edges,separators=(",",":"))
                temporary=review_path.with_suffix(".csv.tmp"); frame.to_csv(temporary,index=False); os.replace(temporary,review_path)
                self.send_json({"ok":True})
            except Exception as exc: self.send_json({"error":str(exc)},400)

        def log_message(self, format, *values):
            print(f"review-ui: {format % values}")
    return Handler


def main() -> None:
    args=arguments(); review_path=args.review_dir/"edge_review.csv"
    if not review_path.exists(): raise FileNotFoundError(review_path)
    server=HTTPServer((args.host,args.port),application(args))
    print(f"Reviewer {args.reviewer.upper()} calibration UI: http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop. Saves go directly to",review_path)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__":
    main()
