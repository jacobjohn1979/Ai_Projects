"""
dashboard.py — Real-time Fraud Intelligence Dashboard v2.0
Live risk trend, flag heatmap, score distribution, velocity alerts, recent screenings.
"""
import os, json, logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("fraud_detect.dashboard")
DATABASE_URL = os.getenv("DATABASE_URL","postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
app = FastAPI(title="Fraud Intelligence Dashboard", version="2.0.0")

def _q(sql, params={}):
    db = SessionLocal()
    try:
        return [dict(r._mapping) for r in db.execute(text(sql), params)]
    except Exception as e:
        log.error(f"Query: {e}"); return []
    finally:
        db.close()

@app.get("/api/stats")
def api_stats(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _q("""
        SELECT COUNT(*) AS total,
            COUNT(*) FILTER (WHERE risk_level='HIGH')   AS high,
            COUNT(*) FILTER (WHERE risk_level='MEDIUM') AS medium,
            COUNT(*) FILTER (WHERE risk_level='LOW')    AS low,
            ROUND(AVG(risk_score)::numeric,1)           AS avg_score,
            COUNT(DISTINCT applicant_id)                AS unique_applicants,
            COUNT(*) FILTER (WHERE risk_level='HIGH'
                AND screened_at >= NOW()-INTERVAL '24 hours') AS high_24h
        FROM screening_logs WHERE screened_at >= :since
    """, {"since": since})
    return rows[0] if rows else {}

@app.get("/api/trend")
def api_trend(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _q("""
        SELECT DATE(screened_at) AS date, COUNT(*) AS total,
            COUNT(*) FILTER (WHERE risk_level='HIGH')   AS high,
            COUNT(*) FILTER (WHERE risk_level='MEDIUM') AS medium,
            COUNT(*) FILTER (WHERE risk_level='LOW')    AS low,
            ROUND(AVG(risk_score)::numeric,1)           AS avg_score
        FROM screening_logs WHERE screened_at >= :since
        GROUP BY DATE(screened_at) ORDER BY date
    """, {"since": since})
    for r in rows:
        if r.get("date"): r["date"] = str(r["date"])
    return rows

@app.get("/api/flags")
def api_flags(days: int = 30, limit: int = 12):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _q("""
        SELECT flag, COUNT(*) AS count FROM (
            SELECT jsonb_array_elements_text(flags::jsonb) AS flag
            FROM screening_logs
            WHERE screened_at >= :since AND flags IS NOT NULL
              AND flags != 'null' AND flags != '[]'
        ) sub GROUP BY flag ORDER BY count DESC LIMIT :limit
    """, {"since": since, "limit": limit})
    return rows

@app.get("/api/hourly")
def api_hourly():
    rows = _q("""
        SELECT EXTRACT(HOUR FROM screened_at)::int AS hour,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE risk_level='HIGH') AS high
        FROM screening_logs WHERE screened_at >= NOW()-INTERVAL '7 days'
        GROUP BY EXTRACT(HOUR FROM screened_at) ORDER BY hour
    """)
    return rows

@app.get("/api/recent")
def api_recent(limit: int = 12):
    rows = _q("""
        SELECT id, file_name, doc_type, risk_level, risk_score,
               applicant_id, screened_at
        FROM screening_logs ORDER BY screened_at DESC LIMIT :limit
    """, {"limit": limit})
    for r in rows:
        if r.get("screened_at"): r["screened_at"] = r["screened_at"].isoformat()
    return rows

@app.get("/api/risk-distribution")
def api_risk_dist(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _q("""
        SELECT CASE
            WHEN risk_score < 10 THEN '0-9'
            WHEN risk_score < 20 THEN '10-19'
            WHEN risk_score < 30 THEN '20-29'
            WHEN risk_score < 40 THEN '30-39'
            WHEN risk_score < 50 THEN '40-49'
            WHEN risk_score < 75 THEN '50-74'
            ELSE '75+' END AS bucket,
            COUNT(*) AS count
        FROM screening_logs WHERE screened_at >= :since AND risk_score IS NOT NULL
        GROUP BY bucket ORDER BY MIN(risk_score)
    """, {"since": since})
    return rows

@app.get("/api/doc-types")
def api_doc_types(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _q("""
        SELECT doc_type, COUNT(*) AS count,
               COUNT(*) FILTER (WHERE risk_level='HIGH') AS high,
               ROUND(AVG(risk_score)::numeric,1) AS avg_score
        FROM screening_logs WHERE screened_at >= :since
        GROUP BY doc_type ORDER BY count DESC
    """, {"since": since})
    return rows

@app.get("/api/velocity")
def api_velocity(days: int = 7):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _q("""
        SELECT applicant_id, COUNT(*) AS submissions,
               COUNT(*) FILTER (WHERE risk_level='HIGH') AS high_count,
               MAX(screened_at) AS last_seen
        FROM screening_logs WHERE screened_at >= :since AND applicant_id IS NOT NULL
        GROUP BY applicant_id HAVING COUNT(*) >= 2
        ORDER BY submissions DESC LIMIT 10
    """, {"since": since})
    for r in rows:
        if r.get("last_seen"): r["last_seen"] = r["last_seen"].isoformat()
    return rows

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(HTML)

INLINE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud Intelligence Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--nav:#1a2744;--accent:#2563eb;--surface:#f8fafc;--card:#fff;--border:#e2e8f0;
      --text:#0f172a;--muted:#64748b;
      --high:#dc2626;--high-bg:#fef2f2;--medium:#d97706;--medium-bg:#fffbeb;
      --low:#16a34a;--low-bg:#f0fdf4}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);font-size:14px}
.topbar{background:var(--nav);color:#fff;height:56px;display:flex;align-items:center;
        padding:0 24px;justify-content:space-between;
        box-shadow:0 2px 8px rgba(0,0,0,.2);position:sticky;top:0;z-index:100}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;background:var(--accent);border-radius:7px;
            display:flex;align-items:center;justify-content:center;
            font-size:14px;font-weight:800;color:#fff}
.brand-title{font-size:14px;font-weight:700;color:#f1f5f9}
.brand-sub{font-size:11px;color:#64748b}
.topbar-r{display:flex;align-items:center;gap:10px}
.live-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;
          animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.4)}}
.live-lbl{font-size:12px;color:#86efac;font-weight:700}
select.period{background:#243460;color:#c8dff0;border:1px solid #334e80;
              padding:5px 10px;border-radius:6px;font-size:12px;cursor:pointer}
.link-btn{color:#7dd3fc;font-size:12px;border:1px solid #334e80;
          padding:5px 12px;border-radius:6px;text-decoration:none}
.wrap{padding:20px 24px;max-width:1400px}
.sg{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px}
.sc{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.si{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;
    justify-content:center;margin-bottom:10px;font-size:15px}
.sn{font-size:26px;font-weight:800;letter-spacing:-.5px;line-height:1}
.sl{font-size:11px;color:var(--muted);font-weight:500;margin-top:3px}
.ss{font-size:11px;margin-top:6px;font-weight:500}
.g2{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.g2b{display:grid;grid-template-columns:3fr 2fr;gap:14px;margin-bottom:14px}
.cc{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.ch{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.ct{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.cb{font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px}
.bl{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}
.bd{background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd}
.rt{width:100%;border-collapse:collapse;font-size:12px}
.rt th{padding:8px 10px;text-align:left;font-size:10px;font-weight:700;color:var(--muted);
       text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid var(--border);
       background:#f8fafc}
.rt td{padding:9px 10px;border-bottom:1px solid #f1f5f9;vertical-align:middle}
.rt tr:hover td{background:#fafbff}
.rp{display:inline-flex;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.rH{background:var(--high-bg);color:var(--high)}
.rM{background:var(--medium-bg);color:var(--medium)}
.rL{background:var(--low-bg);color:var(--low)}
.fb{display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:12px}
.fn{flex:1;font-family:monospace;font-size:11px;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis}
.ft{flex:0 0 90px;height:6px;background:#f1f5f9;border-radius:3px;overflow:hidden}
.ff{height:100%;border-radius:3px;transition:width .8s ease}
.fc{flex:0 0 28px;text-align:right;color:var(--muted);font-size:11px;font-weight:600}
.hm{display:grid;grid-template-columns:repeat(24,1fr);gap:3px;margin-top:8px}
.hc{height:28px;border-radius:3px;background:#f1f5f9;cursor:pointer;transition:transform .1s}
.hc:hover{transform:scaleY(1.2)}
.hl{display:grid;grid-template-columns:repeat(24,1fr);gap:3px;font-size:9px;
    color:var(--muted);text-align:center;margin-top:3px}
.vi{display:flex;align-items:center;justify-content:space-between;
    padding:8px 0;border-bottom:1px solid #f1f5f9;font-size:12px}
.vi:last-child{border-bottom:none}
.vid{font-family:monospace;color:var(--accent);font-weight:600}
.vc{font-weight:700;font-size:15px}
.rt2{font-size:11px;color:var(--muted);text-align:center;padding:10px}
</style></head><body>
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">F</div>
    <div><div class="brand-title">Fraud Intelligence Dashboard</div>
         <div class="brand-sub">KYC Screening · Real-time</div></div>
  </div>
  <div class="topbar-r">
    <div class="live-dot"></div><span class="live-lbl">LIVE</span>
    <select class="period" id="period" onchange="setPeriod(this.value)">
      <option value="7">Last 7 days</option>
      <option value="30" selected>Last 30 days</option>
      <option value="90">Last 90 days</option>
    </select>
    <a class="link-btn" href="/portal/">KYC Portal</a>
    <a class="link-btn" href="/loan/">Loan Portal</a>
  </div>
</div>
<div class="wrap">

  <div class="sg">
    <div class="sc">
      <div class="si" style="background:#eff6ff">📋</div>
      <div class="sn" id="s-tot">—</div><div class="sl">Total Screened</div>
      <div class="ss" id="s-app" style="color:var(--muted)"></div>
    </div>
    <div class="sc">
      <div class="si" style="background:#fef2f2">🚨</div>
      <div class="sn" id="s-hi" style="color:var(--high)">—</div><div class="sl">HIGH Risk</div>
      <div class="ss" id="s-hi24" style="color:var(--high)"></div>
    </div>
    <div class="sc">
      <div class="si" style="background:#fffbeb">⚠</div>
      <div class="sn" id="s-me" style="color:var(--medium)">—</div><div class="sl">MEDIUM Risk</div>
    </div>
    <div class="sc">
      <div class="si" style="background:#f0fdf4">✓</div>
      <div class="sn" id="s-lo" style="color:var(--low)">—</div><div class="sl">LOW Risk</div>
    </div>
    <div class="sc">
      <div class="si" style="background:#f5f3ff">📊</div>
      <div class="sn" id="s-avg" style="color:#7c3aed">—</div><div class="sl">Avg Risk Score</div>
    </div>
  </div>

  <div class="g2">
    <div class="cc">
      <div class="ch"><div class="ct">Risk trend</div><span class="cb bl">Auto-refresh 30s</span></div>
      <canvas id="tC" height="90"></canvas>
    </div>
    <div class="cc">
      <div class="ch"><div class="ct">Risk split</div><span class="cb bd">Period</span></div>
      <canvas id="dC" height="150"></canvas>
      <div id="dL" style="display:flex;gap:12px;justify-content:center;margin-top:10px;font-size:12px"></div>
    </div>
  </div>

  <div class="g3">
    <div class="cc">
      <div class="ch"><div class="ct">Top fraud flags</div><span class="cb bd">Period</span></div>
      <div id="fL"></div>
    </div>
    <div class="cc">
      <div class="ch"><div class="ct">Submissions by hour</div><span class="cb bd">Last 7 days</span></div>
      <div class="hm" id="hM"></div>
      <div class="hl" id="hL"></div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:10px;color:var(--muted)">
        <span>Low</span><span>High</span></div>
      <div style="height:5px;border-radius:3px;margin-top:3px;
           background:linear-gradient(90deg,#e0f2fe,#0284c7,#1e3a5f)"></div>
    </div>
    <div class="cc">
      <div class="ch"><div class="ct">Score distribution</div><span class="cb bd">Period</span></div>
      <canvas id="sC" height="150"></canvas>
    </div>
  </div>

  <div class="g2b">
    <div class="cc">
      <div class="ch"><div class="ct">Recent screenings</div><span class="cb bl">Live</span></div>
      <table class="rt">
        <thead><tr><th>File</th><th>Type</th><th>Applicant</th><th>Risk</th><th>Score</th><th>Time</th></tr></thead>
        <tbody id="rB"></tbody>
      </table>
    </div>
    <div class="cc">
      <div class="ch"><div class="ct">Velocity alerts</div><span class="cb bd">Last 7 days</span></div>
      <div id="vL"></div>
      <div style="margin-top:16px">
        <div class="ct" style="margin-bottom:10px">By document type</div>
        <canvas id="dtC" height="100"></canvas>
      </div>
    </div>
  </div>

  <div class="rt2" id="rt"></div>
</div>

<script>
let period=30, tC, dC, sC, dtC;
const BASE = window.location.pathname.replace(/[/]$/, '');

async function jf(p){ const r=await fetch(BASE+p); return r.ok?r.json():{}; }
function ft(iso){ return iso ? new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : ''; }
function rc(l){ return l==='HIGH'?'#dc2626':l==='MEDIUM'?'#d97706':'#16a34a'; }
function setPeriod(v){ period=parseInt(v); loadAll(); }

async function loadStats(){
  const d=await jf(`/api/stats?days=${period}`);
  document.getElementById('s-tot').textContent=(d.total||0).toLocaleString();
  document.getElementById('s-hi').textContent=(d.high||0).toLocaleString();
  document.getElementById('s-me').textContent=(d.medium||0).toLocaleString();
  document.getElementById('s-lo').textContent=(d.low||0).toLocaleString();
  document.getElementById('s-avg').textContent=d.avg_score||'—';
  document.getElementById('s-app').textContent=`${(d.unique_applicants||0).toLocaleString()} applicants`;
  const h24=d.high_24h||0;
  document.getElementById('s-hi24').textContent=h24>0?`▲ ${h24} in 24h`:'0 in 24h';
}

async function loadTrend(){
  const rows=await jf(`/api/trend?days=${period}`);
  const labels=rows.map(r=>r.date?r.date.slice(5):'');
  if(tC) tC.destroy();
  tC=new Chart(document.getElementById('tC').getContext('2d'),{
    type:'bar',
    data:{labels,datasets:[
      {label:'HIGH',  data:rows.map(r=>r.high||0),  backgroundColor:'#fca5a5',stack:'s'},
      {label:'MEDIUM',data:rows.map(r=>r.medium||0),backgroundColor:'#fcd34d',stack:'s'},
      {label:'LOW',   data:rows.map(r=>r.low||0),   backgroundColor:'#86efac',stack:'s'},
      {label:'Avg Score',data:rows.map(r=>r.avg_score||0),type:'line',yAxisID:'y2',
       borderColor:'#7c3aed',backgroundColor:'transparent',borderWidth:2,pointRadius:2,tension:.4},
    ]},
    options:{responsive:true,interaction:{mode:'index'},
      plugins:{legend:{position:'top',labels:{font:{size:10},boxWidth:10}}},
      scales:{x:{grid:{display:false},ticks:{font:{size:10},maxTicksLimit:14}},
              y:{stacked:true,grid:{color:'#f8fafc'},ticks:{font:{size:10}}},
              y2:{position:'right',grid:{display:false},ticks:{font:{size:10}},
                  title:{display:true,text:'Score',font:{size:10}}}}}
  });
}

async function loadDonut(){
  const d=await jf(`/api/stats?days=${period}`);
  const vals=[d.high||0,d.medium||0,d.low||0];
  const cols=['#dc2626','#d97706','#16a34a'];
  const lbls=['HIGH','MEDIUM','LOW'];
  if(dC) dC.destroy();
  dC=new Chart(document.getElementById('dC').getContext('2d'),{
    type:'doughnut',
    data:{labels:lbls,datasets:[{data:vals,backgroundColor:cols,borderWidth:0,hoverOffset:6}]},
    options:{cutout:'68%',responsive:true,plugins:{legend:{display:false},
      tooltip:{callbacks:{label:c=>` ${c.label}: ${c.parsed} (${Math.round(c.parsed/(vals.reduce((a,b)=>a+b,0)||1)*100)}%)`}}}}
  });
  const tot=vals.reduce((a,b)=>a+b,0)||1;
  document.getElementById('dL').innerHTML=lbls.map((l,i)=>
    `<span style="display:flex;align-items:center;gap:4px">
       <span style="width:9px;height:9px;border-radius:2px;background:${cols[i]};flex-shrink:0"></span>
       <span style="color:#64748b;font-size:11px">${l} <b>${Math.round(vals[i]/tot*100)}%</b></span>
     </span>`).join('');
}

async function loadFlags(){
  const rows=await jf(`/api/flags?days=${period}&limit=12`);
  const max=rows[0]?.count||1;
  const cols=['#dc2626','#d97706','#2563eb','#7c3aed','#0891b2','#16a34a'];
  document.getElementById('fL').innerHTML=rows.map((r,i)=>`
    <div class="fb">
      <div class="fn" title="${r.flag}">${r.flag}</div>
      <div class="ft"><div class="ff" style="width:${Math.round(r.count/max*100)}%;background:${cols[i%cols.length]}"></div></div>
      <div class="fc">${r.count}</div>
    </div>`).join('')||'<div style="color:#94a3b8;font-size:12px;text-align:center;padding:20px">No flags</div>';
}

async function loadHeatmap(){
  const rows=await jf('/api/hourly');
  const byHr={};
  rows.forEach(r=>byHr[r.hour]={total:r.total||0,high:r.high||0});
  const max=Math.max(...Object.values(byHr).map(v=>v.total),1);
  const hm=document.getElementById('hM'), hl=document.getElementById('hL');
  hm.innerHTML=''; hl.innerHTML='';
  for(let h=0;h<24;h++){
    const v=byHr[h]?.total||0, hi=byHr[h]?.high||0, pct=Math.round(v/max*100);
    const bg=hi>0?`rgba(220,38,38,${0.1+pct/100*.7})`:pct>0?`rgba(37,99,235,${0.08+pct/100*.55})`:'#f8fafc';
    const c=document.createElement('div'); c.className='hc'; c.style.background=bg;
    c.title=`${h}:00 — ${v} submissions${hi>0?' ('+hi+' HIGH)':''}`;
    hm.appendChild(c);
    const l=document.createElement('div'); l.textContent=h%6===0?h+'h':''; hl.appendChild(l);
  }
}

async function loadDist(){
  const rows=await jf(`/api/risk-distribution?days=${period}`);
  const cols=rows.map(r=>{const b=r.bucket;
    return(b==='0-9'||b==='10-19')?'#86efac':(b==='75+'||b==='50-74')?'#fca5a5':'#fcd34d';});
  if(sC) sC.destroy();
  sC=new Chart(document.getElementById('sC').getContext('2d'),{
    type:'bar',
    data:{labels:rows.map(r=>r.bucket),datasets:[{data:rows.map(r=>r.count||0),backgroundColor:cols,borderRadius:4}]},
    options:{responsive:true,plugins:{legend:{display:false}},
      scales:{x:{grid:{display:false},ticks:{font:{size:10}}},y:{grid:{color:'#f8fafc'},ticks:{font:{size:10}}}}}
  });
}

async function loadRecent(){
  const rows=await jf('/api/recent?limit=12');
  document.getElementById('rB').innerHTML=rows.map(r=>`<tr>
    <td style="font-size:11px;font-family:monospace;color:#374151">${(r.file_name||'').substring(0,20)}</td>
    <td><span style="background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:10px">${r.doc_type||''}</span></td>
    <td style="font-size:11px;color:#64748b">${(r.applicant_id||'—').substring(0,12)}</td>
    <td><span class="rp r${r.risk_level||'LOW'}">${r.risk_level||'?'}</span></td>
    <td style="font-weight:700;color:${rc(r.risk_level)}">${r.risk_score||0}</td>
    <td style="font-size:11px;color:#94a3b8">${ft(r.screened_at)}</td>
  </tr>`).join('')||'<tr><td colspan="6" style="text-align:center;padding:20px;color:#94a3b8">No data</td></tr>';
}

async function loadVelocity(){
  const rows=await jf('/api/velocity');
  document.getElementById('vL').innerHTML=rows.length?rows.map(r=>`
    <div class="vi">
      <div><div class="vid">${(r.applicant_id||'').substring(0,16)}</div>
           <div style="font-size:10px;color:#94a3b8;margin-top:2px">${r.high_count||0} HIGH risk</div></div>
      <div style="text-align:right">
        <div class="vc" style="color:${(r.high_count||0)>0?'#dc2626':'#374151'}">${r.submissions}</div>
        <div style="font-size:10px;color:#94a3b8">submissions</div>
      </div>
    </div>`).join(''):'<div style="color:#94a3b8;font-size:12px;padding:12px;text-align:center">No repeated submissions</div>';
}

async function loadDocTypes(){
  const rows=await jf(`/api/doc-types?days=${period}`);
  if(dtC) dtC.destroy();
  dtC=new Chart(document.getElementById('dtC').getContext('2d'),{
    type:'bar',
    data:{labels:rows.map(r=>r.doc_type||'?'),datasets:[
      {label:'Total',data:rows.map(r=>r.count||0),backgroundColor:'#bfdbfe',borderRadius:3},
      {label:'HIGH', data:rows.map(r=>r.high||0), backgroundColor:'#fca5a5',borderRadius:3},
    ]},
    options:{responsive:true,plugins:{legend:{position:'top',labels:{font:{size:10},boxWidth:10}}},
      scales:{x:{grid:{display:false},ticks:{font:{size:10}}},y:{grid:{color:'#f8fafc'},ticks:{font:{size:10}}}}}
  });
}

async function loadAll(){
  await Promise.all([loadStats(),loadTrend(),loadDonut(),loadFlags(),
    loadHeatmap(),loadDist(),loadRecent(),loadVelocity(),loadDocTypes()]);
  document.getElementById('rt').textContent='Last updated: '+new Date().toLocaleTimeString();
}

loadAll();
setInterval(loadAll,30000);
</script>
</body></html>"""

HTML = INLINE_HTML
