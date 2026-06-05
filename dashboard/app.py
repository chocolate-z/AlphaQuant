# Flask 本地看板：账户总览、持仓、信号（缓存读取）、历史交易（可筛选）、模型表现

import os
import csv
import json
import time
import logging
from datetime import datetime

from flask import Flask, request, Response, jsonify, send_from_directory

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGS_DIR, DASHBOARD_HOST, DASHBOARD_PORT, INIT_CAPITAL, SIGNAL_CACHE_FILE
from dashboard.tasks import manager

logger = logging.getLogger(__name__)
app    = Flask(__name__)

# ── 数据读取 ──────────────────────────────────────────

def _read_account_state() -> dict:
    state_file = os.path.join(LOGS_DIR, "account_state.json")
    if not os.path.exists(state_file):
        return {"cash": INIT_CAPITAL, "holdings": {}, "total_assets": INIT_CAPITAL, "holding_value": 0}
    with open(state_file, encoding="utf-8") as f:
        state = json.load(f)
    state["holding_value"] = sum(
        p.get("market_value", p["shares"] * p["cost"])
        for p in state.get("holdings", {}).values()
    )
    state["total_assets"] = state["cash"] + state["holding_value"]
    return state


def _read_daily_csv() -> list:
    path = os.path.join(LOGS_DIR, "account_daily.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_all_trades(filter_code: str = "", filter_date: str = "") -> list:
    trades = []
    for fn in sorted(os.listdir(LOGS_DIR)):
        if fn.startswith("trades_") and fn.endswith(".csv"):
            if filter_date and filter_date.replace("-", "") not in fn:
                continue
            with open(os.path.join(LOGS_DIR, fn), encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if filter_code and filter_code.lower() not in row.get("code", "").lower():
                        continue
                    trades.append(row)
    return trades


def _read_signal_cache() -> dict:
    """读取信号缓存文件（由 executor 每日更新）。"""
    if not os.path.exists(SIGNAL_CACHE_FILE):
        return {"date": "尚未生成", "time": "", "signals": {}}
    try:
        with open(SIGNAL_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"date": "读取失败", "time": "", "signals": {}}


# ── 基础样式 ──────────────────────────────────────────

_CSS = """
<style>
:root{
  --bg:#f5f5f7; --card:#ffffff; --card-2:#fbfbfd; --text:#1d1d1f; --sub:#6e6e73; --faint:#8e8e93;
  --line:rgba(0,0,0,.08); --hairline:rgba(0,0,0,.06); --hover:rgba(0,0,0,.045);
  --accent:#0071e3; --accent-h:#0077ed; --accent-soft:rgba(0,113,227,.12);
  --up:#1d8a3f; --up-s:rgba(52,199,89,.14); --down:#d70015; --down-s:rgba(255,59,48,.12); --warn:#b25e00;
  --sidebar:rgba(246,246,248,.75); --radius:18px; --radius-sm:12px;
  --shadow:0 1px 2px rgba(0,0,0,.04),0 6px 20px rgba(0,0,0,.05);
  --shadow-lg:0 8px 36px rgba(0,0,0,.12);
}
/* 深色变量：手动选「深色」时强制；「跟随系统」时随 prefers-color-scheme；「浅色」时不生效 */
[data-theme="dark"],
.aq-dark{
    --bg:#000000; --card:#1c1c1e; --card-2:#161618; --text:#f5f5f7; --sub:#a1a1a6; --faint:#8e8e93;
    --line:rgba(255,255,255,.1); --hairline:rgba(255,255,255,.07); --hover:rgba(255,255,255,.06);
    --accent:#0a84ff; --accent-h:#3a9bff; --accent-soft:rgba(10,132,255,.22);
    --up:#30d158; --up-s:rgba(48,209,88,.18); --down:#ff453a; --down-s:rgba(255,69,58,.18); --warn:#ffd60a;
    --sidebar:rgba(28,28,30,.72);
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.4);
    --shadow-lg:0 10px 40px rgba(0,0,0,.6);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#000000; --card:#1c1c1e; --card-2:#161618; --text:#f5f5f7; --sub:#a1a1a6; --faint:#8e8e93;
    --line:rgba(255,255,255,.1); --hairline:rgba(255,255,255,.07); --hover:rgba(255,255,255,.06);
    --accent:#0a84ff; --accent-h:#3a9bff; --accent-soft:rgba(10,132,255,.22);
    --up:#30d158; --up-s:rgba(48,209,88,.18); --down:#ff453a; --down-s:rgba(255,69,58,.18); --warn:#ffd60a;
    --sidebar:rgba(28,28,30,.72);
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.4);
    --shadow-lg:0 10px 40px rgba(0,0,0,.6);
  }
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","PingFang SC","Helvetica Neue",Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);margin:0;-webkit-font-smoothing:antialiased;letter-spacing:-.012em;
  font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
a{color:inherit}
/* ── 应用骨架：侧栏 + 内容 ── */
.app{display:flex;min-height:100vh}
.sidebar{width:248px;flex-shrink:0;position:sticky;top:0;align-self:flex-start;height:100vh;overflow-y:auto;
  background:var(--sidebar);backdrop-filter:saturate(180%) blur(24px);-webkit-backdrop-filter:saturate(180%) blur(24px);
  border-right:1px solid var(--line);padding:18px 14px 22px;display:flex;flex-direction:column}
.brand{display:flex;align-items:center;gap:11px;padding:6px 10px 16px}
.brand .logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(145deg,#0a84ff,#0a40d8);
  display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:1.05em;
  box-shadow:0 3px 8px rgba(10,90,255,.4);flex-shrink:0}
.brand .name{font-weight:600;font-size:1.04em;letter-spacing:-.02em;line-height:1.1}
.brand .sub{font-size:.7em;color:var(--faint);margin-top:1px}
.nav-group{font-size:.68em;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--faint);
  padding:14px 12px 5px}
.nav-item{display:flex;align-items:center;gap:11px;padding:8px 12px;border-radius:9px;color:var(--text);
  text-decoration:none;font-size:.91em;font-weight:450;transition:background .15s,color .15s;margin:1px 0}
.nav-item .ic{width:18px;height:18px;display:flex;align-items:center;justify-content:center;color:var(--sub);flex-shrink:0}
.nav-item:hover{background:var(--hover)}
.nav-item.active{background:var(--accent);color:#fff;font-weight:500;box-shadow:0 2px 8px rgba(0,113,227,.3)}
.nav-item.active .ic{color:#fff}
.content{flex:1;min-width:0;padding:46px 52px 70px;max-width:1240px;width:100%}
/* ── 标题 ── */
h1{color:var(--text);font-weight:700;font-size:2.5em;margin:0 0 4px;letter-spacing:-.028em;line-height:1.05}
h2{color:var(--text);font-size:1.2em;font-weight:600;margin:30px 0 12px;letter-spacing:-.018em}
/* ── 表格（分组内嵌圆角）── */
table{width:100%;border-collapse:separate;border-spacing:0;margin:14px 0;font-size:.9em;
  background:var(--card);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
th{background:var(--card-2);color:var(--sub);padding:13px 18px;text-align:left;font-weight:500;font-size:.92em;
   border-bottom:1px solid var(--hairline)}
td{padding:13px 18px;border-bottom:1px solid var(--hairline)}
tr:last-child td{border-bottom:none}
tbody tr{transition:background .12s}
tbody tr:hover td{background:var(--hover)}
/* ── 数据卡片 ── */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px;margin:20px 0}
.card{background:var(--card);border-radius:var(--radius);padding:18px 22px;box-shadow:var(--shadow);
  transition:transform .25s cubic-bezier(.2,.7,.3,1),box-shadow .25s}
.card:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}
.val{font-size:1.75em;font-weight:600;color:var(--text);margin:5px 0 0;letter-spacing:-.025em}
.lbl{color:var(--sub);font-size:.8em;font-weight:500}
.up{color:var(--up)}.down{color:var(--down)}.neutral{color:var(--sub)}
.bar{display:inline-block;background:var(--accent);height:8px;border-radius:4px;vertical-align:middle}
.badge-buy{background:var(--up-s);color:var(--up);padding:3px 11px;border-radius:20px;font-size:.82em;font-weight:600}
.badge-sell{background:var(--down-s);color:var(--down);padding:3px 11px;border-radius:20px;font-size:.82em;font-weight:600}
.badge-hold{background:var(--hover);color:var(--sub);padding:3px 11px;border-radius:20px;font-size:.82em;font-weight:600}
/* ── 表单控件 ── */
input,select{background:var(--card);border:1px solid var(--line);color:var(--text);padding:8px 12px;
  border-radius:10px;font-size:.9em;font-family:inherit;transition:border-color .15s,box-shadow .15s}
input::placeholder{color:var(--faint)}
input:focus,select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
button{background:var(--accent);color:#fff;border:none;padding:8px 20px;border-radius:980px;
  cursor:pointer;font-size:.9em;font-weight:500;font-family:inherit;transition:background .2s,transform .08s,box-shadow .2s;
  box-shadow:0 1px 3px rgba(0,0,0,.12)}
button:hover{background:var(--accent-h);box-shadow:0 2px 8px rgba(0,113,227,.3)}
button:active{transform:scale(.96)}
button:disabled{background:var(--hover);color:var(--faint);cursor:not-allowed;transform:none;box-shadow:none}
button.danger{background:var(--down)}button.danger:hover{background:#ff5147}
.filter-bar{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.note{color:var(--sub);font-size:.86em;margin:8px 0;line-height:1.55}
canvas{max-width:100%}
/* ── 面板 ── */
.panel{background:var(--card);border-radius:var(--radius);padding:18px 20px;box-shadow:var(--shadow);
  transition:box-shadow .25s,transform .25s}
.panel:hover{box-shadow:var(--shadow-lg)}
.panel h2{margin:0 0 4px 0;color:var(--text);font-size:1.02em}
.panel .desc{color:var(--sub);font-size:.82em;margin:0 0 12px 0}
.panel form{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}
/* ── 操作中心：左操作 + 右固定日志 ── */
.control-layout{display:grid;grid-template-columns:1fr 440px;gap:24px;align-items:start;margin-top:10px}
.control-actions{min-width:0}
.control-side{position:sticky;top:24px;display:flex;flex-direction:column;gap:14px}
#console{background:#1d1d1f;border-radius:14px;padding:16px 18px;
  font-family:"SF Mono",ui-monospace,"Menlo","Consolas",monospace;font-size:.78em;color:#e8e8ed;
  white-space:pre-wrap;word-break:break-all;height:340px;overflow-y:auto;line-height:1.6;
  box-shadow:inset 0 1px 4px rgba(0,0,0,.5)}
#console::-webkit-scrollbar{width:8px}#console::-webkit-scrollbar-thumb{background:#48484a;border-radius:4px}
.status-running{color:var(--warn);font-weight:600}.status-done{color:var(--up);font-weight:600}
.status-error{color:var(--down);font-weight:600}
.spin{display:inline-block;width:11px;height:11px;border:2px solid var(--warn);border-top-color:transparent;
  border-radius:50%;animation:sp .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
/* ── 侧栏底部主题切换 ── */
.side-foot{margin-top:auto;padding:14px 8px 2px}
.side-foot .ttl{font-size:.68em;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);padding:0 4px 6px}
.seg{display:flex;background:var(--hover);border-radius:9px;padding:3px;gap:2px}
.seg button{flex:1;background:transparent;color:var(--sub);box-shadow:none;border-radius:7px;
  padding:5px 0;font-size:.76em;font-weight:500}
.seg button:hover{background:transparent;color:var(--text);box-shadow:none;transform:none}
.seg button.active{background:var(--card);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.18)}
/* ── 账户 Hero（苹果股票风格大数字）── */
.hero{background:var(--card);border-radius:22px;padding:30px 34px;box-shadow:var(--shadow);margin:22px 0}
.hero-label{color:var(--sub);font-size:.92em;font-weight:500}
.hero-num{font-size:3.4em;font-weight:700;letter-spacing:-.035em;margin:6px 0 16px;line-height:1}
.hero-row{display:flex;gap:12px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:baseline;gap:6px;padding:8px 15px;border-radius:980px;
  font-weight:600;font-size:.95em}
.pill small{font-weight:500;opacity:.65;margin-left:2px;font-size:.82em}
.pill-up{background:var(--up-s);color:var(--up)}
.pill-down{background:var(--down-s);color:var(--down)}
.pill-flat{background:var(--hover);color:var(--sub)}
/* ── 表格微观可视化 ── */
.track{display:inline-block;width:110px;height:8px;background:var(--hover);border-radius:4px;
  vertical-align:middle;overflow:hidden}
.track .fill{display:block;height:100%;border-radius:4px;transition:width .3s}
.minibar{display:inline-block;width:54px;height:6px;background:var(--hover);border-radius:3px;
  vertical-align:middle;overflow:hidden;margin-left:9px}
.minibar i{display:block;height:100%;border-radius:3px}
/* ── 侧栏折叠（Xcode 风）── */
.collapse-btn{align-self:flex-end;background:transparent;color:var(--sub);box-shadow:none;
  padding:3px 9px;border-radius:7px;font-size:1.15em;line-height:1;margin-bottom:2px;transition:transform .25s,background .15s}
.collapse-btn:hover{background:var(--hover);color:var(--text);box-shadow:none;transform:none}
html.aq-collapsed .collapse-btn:hover{transform:rotate(180deg)}
html.aq-collapsed .sidebar{width:66px;padding:18px 9px 22px}
html.aq-collapsed .brand{justify-content:center;gap:0}
html.aq-collapsed .brand .bwrap{display:none}
html.aq-collapsed .nav-item{justify-content:center;padding:9px 0}
html.aq-collapsed .nav-item span:not(.ic){display:none}
html.aq-collapsed .nav-group{display:none}
html.aq-collapsed .side-foot .ttl{display:none}
html.aq-collapsed .seg{flex-direction:column}
html.aq-collapsed .collapse-btn{align-self:center;transform:rotate(180deg)}
/* ── 响应式 ── */
@media(max-width:1080px){.control-layout{grid-template-columns:1fr}.control-side{position:static}}
@media(max-width:820px){
  .app{flex-direction:column}
  .sidebar{width:100%;height:auto;position:sticky;top:0;flex-direction:row;align-items:center;gap:4px;
    overflow-x:auto;border-right:none;border-bottom:1px solid var(--line);padding:8px 12px}
  .brand{padding:4px 8px;flex-shrink:0}.brand .sub{display:none}
  .nav-group{display:none}
  .nav-item{flex-shrink:0;padding:7px 11px}.nav-item span:not(.ic){font-size:.85em}
  .content{padding:28px 22px 60px}h1{font-size:2em}
}
</style>"""

_CHARTJS = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>'

# 单色线性图标（SF Symbols 风格，stroke=currentColor，激活时自动转白）
_ICONS = {
    "overview": '<rect x="3" y="3" width="7" height="9" rx="1.6"/><rect x="14" y="3" width="7" height="5" rx="1.6"/><rect x="14" y="11" width="7" height="10" rx="1.6"/><rect x="3" y="15" width="7" height="6" rx="1.6"/>',
    "holdings": '<rect x="3" y="7" width="18" height="13" rx="2.2"/><path d="M8 7V5.5A2 2 0 0 1 10 3.5h4A2 2 0 0 1 16 5.5V7"/>',
    "signal":   '<circle cx="12" cy="12" r="1.8"/><path d="M16.6 7.4a6.5 6.5 0 0 1 0 9.2M7.4 16.6a6.5 6.5 0 0 1 0-9.2"/>',
    "perf":     '<polyline points="3 16.5 9 10.5 13 14.5 21 6.5"/><polyline points="15.5 6.5 21 6.5 21 12"/>',
    "trades":   '<path d="M5.5 3h13v18l-2.6-1.8L13.3 21 11 19.2 8.7 21 6.1 19.2 3.5 21V5z" transform="translate(1 0)"/><line x1="9" y1="8.5" x2="16" y2="8.5"/><line x1="9" y1="12.5" x2="16" y2="12.5"/>',
    "control":  '<line x1="4" y1="8" x2="20" y2="8"/><line x1="4" y1="16" x2="20" y2="16"/><circle cx="9" cy="8" r="2.3"/><circle cx="15" cy="16" r="2.3"/>',
}

def _icon(name: str) -> str:
    return (f'<span class="ic"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
            f'{_ICONS.get(name, "")}</svg></span>')

# 导航定义：(key, 路由, 图标, 文字)
_NAV_ITEMS = [
    ("group", "概览", "", ""),
    ("overview",  "/",            "overview", "账户总览"),
    ("holdings",  "/holdings",    "holdings", "当前持仓"),
    ("group", "智能", "", ""),
    ("signals",   "/signals",     "signal",   "今日信号"),
    ("performance", "/performance","perf",     "模型表现"),
    ("group", "记录", "", ""),
    ("trades",    "/trades",      "trades",   "历史交易"),
    ("group", "操作", "", ""),
    ("control",   "/control",     "control",  "操作中心"),
]

def _sidebar(active: str = "") -> str:
    rows = []
    for key, href, icon, label in _NAV_ITEMS:
        if key == "group":
            rows.append(f'<div class="nav-group">{href}</div>')
        else:
            cls = "nav-item active" if key == active else "nav-item"
            rows.append(f'<a class="{cls}" href="{href}">{_icon(icon)}<span>{label}</span></a>')
    return (f'<aside class="sidebar">'
            f'  <button type="button" class="collapse-btn" id="collapseBtn" title="折叠/展开侧栏">‹</button>'
            f'  <div class="brand"><div class="logo">A</div>'
            f'    <div class="bwrap"><div class="name">AlphaQuant</div><div class="sub">量化交易系统</div></div></div>'
            f'  <nav>{"".join(rows)}</nav>'
            f'  <div class="side-foot"><div class="ttl">外观</div>'
            f'    <div class="seg" id="themeSeg">'
            f'      <button type="button" data-theme-val="light">浅色</button>'
            f'      <button type="button" data-theme-val="dark">深色</button>'
            f'      <button type="button" data-theme-val="auto">跟随</button>'
            f'    </div></div>'
            f'</aside>')


# 头部内联脚本：渲染前先套用已保存的主题与折叠状态，避免明暗/布局闪烁
_THEME_HEAD = ("<script>(function(){try{var t=localStorage.getItem('aq-theme');"
               "if(t&&t!=='auto')document.documentElement.setAttribute('data-theme',t);"
               "if(localStorage.getItem('aq-sidebar')==='collapsed')"
               "document.documentElement.classList.add('aq-collapsed');}catch(e){}})();</script>")

# 尾部脚本：高亮当前选项 + 点击切换并持久化
_THEME_BODY = """
<script>
(function(){
  function apply(t){
    if(t==='auto'){document.documentElement.removeAttribute('data-theme');}
    else{document.documentElement.setAttribute('data-theme',t);}
  }
  var saved='auto';try{saved=localStorage.getItem('aq-theme')||'auto';}catch(e){}
  apply(saved);
  document.querySelectorAll('#themeSeg button').forEach(function(b){
    if(b.dataset.themeVal===saved)b.classList.add('active');
    b.addEventListener('click',function(){
      var t=b.dataset.themeVal;
      try{localStorage.setItem('aq-theme',t);}catch(e){}
      apply(t);
      document.querySelectorAll('#themeSeg button').forEach(function(x){x.classList.remove('active');});
      b.classList.add('active');
    });
  });
  var cb=document.getElementById('collapseBtn');
  if(cb)cb.addEventListener('click',function(){
    var on=document.documentElement.classList.toggle('aq-collapsed');
    try{localStorage.setItem('aq-sidebar',on?'collapsed':'expanded');}catch(e){}
  });
})();
</script>"""


def _page(title: str, body: str, active: str = "", auto_refresh: int = 0) -> str:
    refresh = f'<meta http-equiv="refresh" content="{auto_refresh}">' if auto_refresh else ""
    return (f"<!DOCTYPE html><html lang='zh-CN'><head>"
            f"<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title} — AlphaQuant</title>{refresh}{_THEME_HEAD}{_CSS}</head>"
            f"<body><div class='app'>{_sidebar(active)}"
            f"<main class='content'>{body}</main></div>{_THEME_BODY}</body></html>")


# ── 路由 ──────────────────────────────────────────────

@app.route("/")
def index():
    state   = _read_account_state()
    daily   = _read_daily_csv()
    total   = state.get("total_assets", INIT_CAPITAL)
    cum_pnl = total - INIT_CAPITAL
    cum_pct = cum_pnl / INIT_CAPITAL * 100

    today_pnl = 0.0
    if len(daily) >= 2:
        try:
            today_pnl = total - float(daily[-2]["total_assets"])
        except Exception:
            pass

    nav_dates = [r["date"] for r in daily[-90:]]
    nav_vals  = [float(r["total_assets"]) for r in daily[-90:]]

    def card(label, value, css_class=""):
        return f"<div class='card'><div class='lbl'>{label}</div><div class='val {css_class}'>{value}</div></div>"

    def pill(amount, pct, label):
        cls  = "pill-up" if amount > 0 else ("pill-down" if amount < 0 else "pill-flat")
        arr  = "↑" if amount > 0 else ("↓" if amount < 0 else "→")
        pcts = f" ({pct:+.2f}%)" if pct is not None else ""
        return (f"<span class='pill {cls}'>{arr} ¥{abs(amount):,.0f}{pcts}"
                f"<small>{label}</small></span>")

    today_pct = (today_pnl / max(total - today_pnl, 1)) * 100 if today_pnl else 0.0
    hero = (f"<div class='hero'>"
            f"  <div class='hero-label'>虚拟总资产</div>"
            f"  <div class='hero-num'>¥{total:,.0f}</div>"
            f"  <div class='hero-row'>{pill(today_pnl, today_pct, '今日')}"
            f"      {pill(cum_pnl, cum_pct, '累计')}</div>"
            f"</div>")

    cards = (hero + "<div class='cards'>"
             + card("现金余额",   f"¥{state.get('cash',0):,.0f}")
             + card("持仓市值",   f"¥{state.get('holding_value',0):,.0f}")
             + card("持仓数量",   f"{len(state.get('holdings',{}))} 只")
             + "</div>")

    chart = f"""
    <h2>资产净值曲线（近90日）</h2>
    {_CHARTJS}
    <div class="panel"><canvas id="navChart" height="100"></canvas></div>
    <script>
    new Chart(document.getElementById('navChart'),{{
      type:'line',
      data:{{labels:{nav_dates},datasets:[{{
        label:'总资产',data:{nav_vals},borderColor:'#0071e3',borderWidth:2,
        fill:true,backgroundColor:'rgba(0,113,227,0.08)',tension:.3,pointRadius:0
      }}]}},
      options:{{
        scales:{{
          y:{{ticks:{{color:'#6e6e73',callback:v=>'¥'+v.toLocaleString()}},grid:{{color:'#f0f0f2'}}}},
          x:{{ticks:{{color:'#6e6e73',maxTicksLimit:12}},grid:{{color:'#f0f0f2'}}}}
        }},
        plugins:{{legend:{{labels:{{color:'#1d1d1f',usePointStyle:true,boxWidth:12}}}}}}
      }}
    }});
    </script>"""

    return _page("账户总览", f"<h1>账户总览</h1>{cards}{chart}", active="overview", auto_refresh=300)


@app.route("/holdings")
def holdings():
    from data.loader import get_stock_name
    state = _read_account_state()
    rows  = ""
    if not state.get("holdings"):
        rows = "<tr><td colspan='7' style='text-align:center;color:var(--sub)'>当前无持仓</td></tr>"
    total_holding_pnl = 0.0
    for code, pos in state.get("holdings", {}).items():
        name   = get_stock_name(code)
        shares = pos.get("shares", 0)
        cost   = pos.get("cost", 0)
        curr   = pos.get("current_price", cost)
        pnl    = (curr - cost) * shares
        pct    = (curr - cost) / max(cost, 0.01) * 100
        cls    = "up" if pnl >= 0 else "down"
        total_holding_pnl += pnl
        barpx  = min(abs(pct), 10) / 10 * 54   # |涨跌幅| 满格 10%
        fillc  = "var(--up)" if pnl >= 0 else "var(--down)"
        minibar = f"<span class='minibar'><i style='width:{barpx:.0f}px;background:{fillc}'></i></span>"
        rows += (f"<tr><td><b>{code}</b></td><td style='color:var(--sub)'>{name}</td>"
                 f"<td>{shares:,}</td>"
                 f"<td>¥{cost:.2f}</td><td>¥{curr:.2f}</td>"
                 f"<td class='{cls}'>{'+'if pnl>=0 else ''}¥{pnl:,.0f} ({pct:+.1f}%){minibar}</td>"
                 f"<td>{pos.get('buy_date','')}</td></tr>")

    cls_total = "up" if total_holding_pnl >= 0 else "down"
    summary = (f"<p style='margin:8px 0;color:var(--sub)'>持仓总浮盈亏："
               f"<span class='{cls_total}'>{'+'if total_holding_pnl>=0 else ''}¥{total_holding_pnl:,.0f}</span></p>")
    body = (f"<h1>当前持仓</h1>{summary}"
            f"<table><thead><tr><th>代码</th><th>名称</th><th>持仓数量</th><th>成本价</th>"
            f"<th>当前价</th><th>浮动盈亏</th><th>买入日期</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("持仓", body, active="holdings", auto_refresh=120)


@app.route("/signals")
def signals():
    from data.loader import get_stock_name
    cached    = _read_signal_cache()
    sigs      = cached.get("signals", {})
    cache_dt  = f"{cached.get('date','')} {cached.get('time','')}".strip()
    is_stale  = cached.get("date", "") != datetime.today().strftime("%Y-%m-%d")

    note = (f"<p class='note'>⚠️ 信号为昨日数据（{cache_dt}），当日执行后自动更新</p>" if is_stale
            else f"<p class='note'>✓ 信号更新时间：{cache_dt}（每日 15:30 后自动刷新）</p>")

    rows = ""
    if not sigs:
        rows = "<tr><td colspan='4' style='text-align:center;color:var(--sub)'>暂无信号缓存，请先运行模拟盘或手动触发：python main.py --mode signal</td></tr>"
    for code, prob in sorted(sigs.items(), key=lambda x: -x[1]):
        name  = get_stock_name(code)
        if prob > 0.65:
            badge = f"<span class='badge-buy'>★ 买入</span>"; col = "var(--up)"
        elif prob < 0.35:
            badge = f"<span class='badge-sell'>▼ 卖出</span>"; col = "var(--down)"
        else:
            badge = f"<span class='badge-hold'>— 观望</span>"; col = "var(--accent)"
        barfill = f"<span class='track'><span class='fill' style='width:{prob*100:.0f}%;background:{col}'></span></span>"
        rows += (f"<tr><td><b>{code}</b></td>"
                 f"<td style='color:var(--sub)'>{name}</td>"
                 f"<td>{barfill} <b>{prob:.1%}</b></td>"
                 f"<td>{badge}</td></tr>")

    body = (f"<h1>今日信号</h1>{note}"
            f"<table><thead><tr><th>股票代码</th><th>名称</th><th>买入概率</th><th>建议</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("今日信号", body, active="signals")


@app.route("/trades")
def trades():
    filter_code = request.args.get("code", "").strip()
    filter_date = request.args.get("date", "").strip()

    all_t = _read_all_trades(filter_code=filter_code, filter_date=filter_date)

    filter_form = f"""
    <div class='filter-bar'>
      <form method='get' style='display:flex;gap:8px;flex-wrap:wrap;align-items:center'>
        <label style='color:var(--sub)'>股票代码</label>
        <input name='code' placeholder='如 sh600519' value='{filter_code}' style='width:140px'>
        <label style='color:var(--sub)'>日期</label>
        <input name='date' type='date' value='{filter_date}'>
        <button type='submit'>筛选</button>
        <a href='/trades' style='color:var(--accent);font-size:.9em;text-decoration:none'>清除筛选</a>
      </form>
      <span class='note'>共 {len(all_t)} 条记录</span>
    </div>"""

    rows = ""
    for t in reversed(all_t[-500:]):
        action = t.get("action", "")
        cls    = "up" if action == "buy" else "down"
        badge  = (f"<span class='badge-buy'>↗ 买入</span>" if action == "buy"
                  else f"<span class='badge-sell'>↘ 卖出</span>")
        rows += (f"<tr><td>{t.get('time','')}</td><td><b>{t.get('code','')}</b></td>"
                 f"<td>{badge}</td><td>¥{t.get('price','')}</td>"
                 f"<td>{t.get('shares','')}</td><td>{t.get('commission','')}</td>"
                 f"<td style='color:var(--sub)'>{t.get('reason','')}</td></tr>")

    if not rows:
        rows = "<tr><td colspan='7' style='text-align:center;color:var(--sub)'>暂无交易记录</td></tr>"

    body = (f"<h1>历史交易记录</h1>{filter_form}"
            f"<table><thead><tr><th>时间</th><th>代码</th><th>操作</th>"
            f"<th>价格</th><th>数量</th><th>手续费</th><th>原因</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("历史交易", body, active="trades")


@app.route("/performance")
def performance():
    all_t    = _read_all_trades()
    daily    = _read_daily_csv()

    from collections import defaultdict, deque
    buy_queues  = defaultdict(deque)
    wins        = 0
    total_sells = 0
    total_pnl   = 0.0

    for t in all_t:
        if t.get("action") == "buy":
            buy_queues[t["code"]].append(float(t.get("price", 0)))
        elif t.get("action") == "sell":
            total_sells += 1
            q = buy_queues.get(t["code"])
            sell_price = float(t.get("price", 0))
            shares     = int(t.get("shares", 0))
            if q:
                bp = q.popleft()
                pnl = (sell_price - bp) * shares
                total_pnl += pnl
                if sell_price > bp:
                    wins += 1

    win_rate = wins / total_sells * 100 if total_sells > 0 else 0

    # 最大连胜/连败统计
    results = []
    buy_map2 = {}
    for t in all_t:
        if t.get("action") == "buy":
            buy_map2[t["code"]] = float(t.get("price", 0))
        elif t.get("action") == "sell" and t["code"] in buy_map2:
            results.append(1 if float(t.get("price", 0)) > buy_map2[t["code"]] else 0)

    max_win_streak = max_lose_streak = cur = 0
    if results:
        cur = streak_val = results[0]
        for r in results[1:]:
            if r == streak_val:
                cur += 1 if streak_val == 1 else -1
            else:
                streak_val = r
                cur = 1 if r == 1 else -1
            if cur > 0:
                max_win_streak  = max(max_win_streak, cur)
            else:
                max_lose_streak = max(max_lose_streak, -cur)

    def card(label, value, css=""):
        return f"<div class='card'><div class='lbl'>{label}</div><div class='val {css}'>{value}</div></div>"

    cards = (f"<div class='cards'>"
             + card("信号胜率",   f"{win_rate:.1f}%", "up" if win_rate > 50 else "down")
             + card("已完成交易", f"{total_sells} 笔")
             + card("盈利次数",   f"{wins}", "up")
             + card("亏损次数",   f"{total_sells-wins}", "down")
             + card("累计实现盈亏", f"{'+'if total_pnl>=0 else ''}¥{total_pnl:,.0f}", "up" if total_pnl>=0 else "down")
             + card("最大连胜",   f"{max_win_streak} 次", "up")
             + card("最大连败",   f"{max_lose_streak} 次", "down")
             + "</div>")

    # 每日胜率趋势（最近30天）
    win_trend_dates = [r["date"] for r in daily[-30:]]
    win_trend_vals  = []
    for r in daily[-30:]:
        # 简化：用当日盈亏正负表示
        try:
            win_trend_vals.append(1 if float(r.get("daily_pnl", 0)) >= 0 else 0)
        except Exception:
            win_trend_vals.append(0)

    chart = f"""
    <h2>每日盈亏状态（近30日，1=盈利，0=亏损）</h2>
    {_CHARTJS}
    <div class="panel"><canvas id="wChart" height="80"></canvas></div>
    <script>
    new Chart(document.getElementById('wChart'),{{
      type:'bar',
      data:{{labels:{win_trend_dates},datasets:[{{
        label:'盈亏',data:{win_trend_vals},
        backgroundColor:ctx=>ctx.raw===1?'rgba(52,199,89,.85)':'rgba(255,59,48,.85)',
        borderWidth:0,borderRadius:4
      }}]}},
      options:{{
        scales:{{
          y:{{min:0,max:1,ticks:{{color:'#6e6e73',stepSize:1}},grid:{{color:'#f0f0f2'}}}},
          x:{{ticks:{{color:'#6e6e73',maxTicksLimit:10}},grid:{{color:'#f0f0f2'}}}}
        }},
        plugins:{{legend:{{display:false}}}}
      }}
    }});
    </script>"""

    return _page("模型表现", f"<h1>模型表现</h1>{cards}{chart}", active="performance")


# ── 操作中心（把命令行菜单搬到网页）────────────────────────────────────

# 操作名 → (中文名, 是否危险)，用于前端展示
_ACTION_META = {
    "train_full":     ("训练模型（完整模式）", False),
    "train_quick":    ("训练模型（快速模式）", False),
    "train_resume":   ("增量训练（继续学习）", False),
    "train_ensemble": ("集成训练（多模型）",   False),
    "backtest":       ("历史回测",            False),
    "single_bt":      ("单股买卖点图",         False),
    "diagnose":       ("个股诊断",            False),
    "portfolio_diag": ("持仓一键诊断",         False),
    "export":         ("导出今日信号 Excel",   False),
    "signal":         ("刷新今日信号",         False),
    "config":         ("查看当前配置",         False),
    "reset":          ("重置虚拟账户",         True),
}


def _backtest_params(f) -> dict:
    """从表单提取回测参数覆盖，仅保留用户实际填写的项（空值回退 config 默认）。"""
    spec = {
        "init_capital":    float, "buy_threshold":  float, "sell_threshold": float,
        "stop_loss":       float, "take_profit":    float, "max_position":   float,
        "rank_sell_bottom": float, "max_holdings":  int,   "top_n_buy":      int,
    }
    out = {}
    for key, cast in spec.items():
        v = (f.get(key) or "").strip()
        if v:
            try:
                out[key] = cast(v)
            except ValueError:
                pass
    rr = f.get("relative_rank")
    if rr in ("0", "1"):
        out["relative_rank"] = (rr == "1")
    return out


def _build_action(action: str, f):
    """根据表单参数构造对应的无交互调用闭包。返回 None 表示未知操作。"""
    import main

    def _float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def _int(x, default):
        try:
            return int(x)
        except (TypeError, ValueError):
            return default

    table = {
        "train_full":     lambda: main.run_train(quick=False, force_refresh=(f.get("refresh") == "1")),
        "train_quick":    lambda: main.run_train(quick=True),
        "train_resume":   lambda: main.run_train(resume=True),
        "train_ensemble": lambda: main.run_train_ensemble(),
        "backtest":       lambda: main.run_backtest(f.get("start") or None, f.get("end") or None,
                                                    params=_backtest_params(f)),
        "single_bt":      lambda: main.run_single_backtest(code=(f.get("code") or "").strip() or None,
                                                           years=_int(f.get("years"), 2)),
        "diagnose":       lambda: main.run_diagnose(stock_input=(f.get("code") or "").strip(),
                                                    cost=_float(f.get("cost"))),
        "portfolio_diag": lambda: main.run_portfolio_diagnose(),
        "export":         lambda: main.run_export_excel(),
        "signal":         lambda: main.run_signal(),
        "config":         lambda: main.run_show_config(),
        "reset":          lambda: main.reset_account(),
    }
    return table.get(action)


@app.route("/run/<action>", methods=["POST"])
def run_action(action):
    if manager.is_busy():
        cur = manager.current()
        return jsonify({"ok": False, "error": f"已有任务在运行：{cur.name if cur else ''}"}), 409
    func = _build_action(action, request.form)
    if func is None:
        return jsonify({"ok": False, "error": "未知操作"}), 400
    name = _ACTION_META.get(action, (action, False))[0]
    task_id = manager.start(name, func)
    if task_id is None:
        return jsonify({"ok": False, "error": "已有任务在运行"}), 409
    return jsonify({"ok": True, "task_id": task_id, "name": name})


@app.route("/task_status")
def task_status():
    """供页面加载时判断是否有任务在跑，便于自动重新挂接控制台。"""
    cur = manager.current()
    if cur is not None and cur.status == "running":
        return jsonify({"busy": True, "task_id": cur.id, "name": cur.name})
    return jsonify({"busy": False})


@app.route("/stream/<task_id>")
def stream(task_id):
    task = manager.get(task_id)
    if task is None:
        return "no such task", 404

    def gen():
        i = 0
        while True:
            n = len(task.lines)
            while i < n:
                yield f"data: {json.dumps(task.lines[i])}\n\n"
                i += 1
            if task.status != "running" and i >= len(task.lines):
                yield f"event: done\ndata: {json.dumps(task.status)}\n\n"
                break
            time.sleep(0.25)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/report/<path:filename>")
def report_file(filename):
    """提供 reports/ 目录下的图表文件。"""
    from config import REPORTS_DIR
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/task_reports/<task_id>")
def task_reports(task_id):
    """返回任务运行期间新生成（或更新）的图表，供完成后内嵌预览。"""
    import glob
    from config import REPORTS_DIR

    task = manager.get(task_id)
    if task is None:
        return jsonify({"images": []})

    cutoff = task.start - 2  # 留 2s 余量，避免临界遗漏
    imgs = []
    for p in glob.glob(os.path.join(REPORTS_DIR, "*.png")):
        mtime = os.path.getmtime(p)
        if mtime >= cutoff:
            name = os.path.basename(p)
            imgs.append({"name": name,
                         "url": f"/report/{name}?t={int(mtime)}",
                         "mtime": mtime})
    imgs.sort(key=lambda x: -x["mtime"])
    return jsonify({"images": imgs})


@app.route("/control")
def control():
    from config import (START_DATE, INIT_CAPITAL as _IC, BUY_THRESHOLD as _BT,
                        SELL_THRESHOLD as _ST, STOP_LOSS_RATIO as _SL, TAKE_PROFIT_RATIO as _TP,
                        MAX_HOLDINGS as _MH, MAX_POSITION_RATIO as _MP, TOP_N_BUY as _TN,
                        RANK_SELL_BOTTOM as _RB, RELATIVE_RANK_MODE as _RR)
    _today = datetime.today().strftime("%Y%m%d")
    _start = START_DATE.replace("-", "")

    def _num(name, label, default, step="any", w=90):
        return (f"<label style='color:var(--sub)'>{label}</label>"
                f"<input name='{name}' type='number' step='{step}' "
                f"placeholder='{default}' style='width:{w}px'>")

    # 回测可调参数（留空=用 config 默认值，占位符显示默认）
    _bt_params = (
        "<details style='margin-top:10px;width:100%'>"
        "<summary style='color:var(--accent);cursor:pointer;font-weight:500'>⚙ 可调参数（留空=用默认值）</summary>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px'>"
        + _num("buy_threshold",   "买入阈值", _BT)
        + _num("sell_threshold",  "卖出阈值", _ST)
        + _num("stop_loss",       "止损线",   _SL)
        + _num("take_profit",     "止盈线",   _TP)
        + _num("max_holdings",    "最大持仓", _MH, step="1", w=70)
        + _num("max_position",    "单股仓位", _MP)
        + _num("top_n_buy",       "每日买入数", _TN, step="1", w=80)
        + _num("rank_sell_bottom", "排名卖出底部比例", _RB, w=120)
        + (f"<label style='color:var(--sub)'>选股模式</label>"
           f"<select name='relative_rank' style='width:130px'>"
           f"<option value=''>默认({'相对排名' if _RR else '绝对阈值'})</option>"
           f"<option value='1'>相对排名(买Top-N)</option>"
           f"<option value='0'>绝对阈值(超买入线)</option></select>")
        + "</div></details>"
    )

    def panel(action, fields_html=""):
        name, danger = _ACTION_META[action]
        btn_cls = "danger" if danger else ""
        confirm = (" onsubmit=\"return confirm('确认执行：" + name + "？')\""
                   if danger else "")
        return (f"<div class='panel'><h2>{name}</h2>"
                f"<form data-action='{action}'{confirm}>{fields_html}"
                f"<button class='{btn_cls}' type='submit'>执行</button></form></div>")

    # ── 训练类 ──
    train_panels = (
        panel("train_full",
              "<label style='color:var(--sub)'><input type='checkbox' name='refresh' value='1'> 强制重新下载数据</label>")
        + panel("train_quick")
        + panel("train_resume")
        + panel("train_ensemble")
    )

    # ── 回测类 ──
    bt_panels = (
        panel("backtest",
              f"<label style='color:var(--sub)'>起始</label><input name='start' value='{_start}' style='width:110px'>"
              f"<label style='color:var(--sub)'>结束</label><input name='end' value='{_today}' style='width:110px'>"
              + _bt_params)
        + panel("single_bt",
                "<input name='code' placeholder='如 sh600519' style='width:130px'>"
                "<label style='color:var(--sub)'>年数</label><input name='years' value='2' style='width:60px'>")
    )

    # ── 工具类 ──
    tool_panels = (
        panel("diagnose",
              "<input name='code' placeholder='代码，逗号分隔多只' style='width:200px'>"
              "<input name='cost' placeholder='成本价(可选)' style='width:110px'>")
        + panel("portfolio_diag")
        + panel("signal")
        + panel("export")
        + panel("config")
        + panel("reset")
    )

    console = """
    <div class="panel">
      <h2 id="taskTitle">运行控制台</h2>
      <p class="note" id="taskState">空闲中——点击左侧任意「执行」按钮开始</p>
      <div id="liveBox" style="display:none;margin:6px 0 12px">
        <div class="lbl" style="margin-bottom:6px">📈 训练实时曲线</div>
        """ + _CHARTJS + """
        <canvas id="liveChart" height="120"></canvas>
      </div>
      <div id="console">（任务输出会实时显示在这里）</div>
      <div id="reports"></div>
    </div>"""

    script = """
    <script>
    const consoleEl = document.getElementById('console');
    const stateEl   = document.getElementById('taskState');
    const reportsEl = document.getElementById('reports');
    const liveBox   = document.getElementById('liveBox');
    let evtSource = null;

    // ── 训练实时曲线（解析控制台每轮日志，无需后端改动）──
    let lineBuf = '', liveChart = null;
    function resetLive(){
      lineBuf = '';
      liveBox.style.display = 'none';
      if(liveChart){ liveChart.destroy(); liveChart = null; }
    }
    function ensureChart(){
      if(liveChart) return liveChart;
      liveBox.style.display = 'block';
      liveChart = new Chart(document.getElementById('liveChart'), {
        type:'line',
        data:{labels:[],datasets:[
          {label:'损失',yAxisID:'yL',data:[],borderColor:'#0071e3',backgroundColor:'#0071e3',pointRadius:0,tension:.3,borderWidth:2},
          {label:'训练识别率',yAxisID:'yR',data:[],borderColor:'#34c759',backgroundColor:'#34c759',pointRadius:0,tension:.3,borderWidth:2},
          {label:'验证识别率',yAxisID:'yR',data:[],borderColor:'#ff9f0a',backgroundColor:'#ff9f0a',pointRadius:0,tension:.3,borderWidth:2}
        ]},
        options:{animation:false,interaction:{mode:'index',intersect:false},
          scales:{
            yL:{position:'left',title:{display:true,text:'损失',color:'#0071e3'},
                ticks:{color:'#6e6e73'},grid:{color:'#f0f0f2'}},
            yR:{position:'right',min:0.3,max:1.0,title:{display:true,text:'识别率AUC',color:'#34c759'},
                ticks:{color:'#6e6e73'},grid:{drawOnChartArea:false}},
            x:{ticks:{color:'#6e6e73',maxTicksLimit:15},grid:{color:'#f0f0f2'}}
          },
          plugins:{legend:{labels:{color:'#1d1d1f',boxWidth:12,usePointStyle:true}}}
        }
      });
      return liveChart;
    }
    function feedChart(text){
      lineBuf += text;
      let idx;
      while((idx = lineBuf.indexOf('\\n')) >= 0){
        const line = lineBuf.slice(0, idx); lineBuf = lineBuf.slice(idx+1);
        const m = line.match(/第\\s*(\\d+)轮.*?损失.*?:\\s*([\\d.]+).*?训练识别率:\\s*([\\d.]+).*?验证识别率:\\s*([\\d.]+)/);
        if(m){
          const ch = ensureChart();
          ch.data.labels.push(m[1]);
          ch.data.datasets[0].data.push(parseFloat(m[2]));
          ch.data.datasets[1].data.push(parseFloat(m[3]));
          ch.data.datasets[2].data.push(parseFloat(m[4]));
          ch.update('none');
        }
      }
    }

    function setButtons(disabled){
      document.querySelectorAll('form[data-action] button').forEach(b=>b.disabled=disabled);
    }
    function showReports(taskId){
      fetch('/task_reports/'+taskId).then(r=>r.json()).then(d=>{
        if(!d.images || !d.images.length){ reportsEl.innerHTML=''; return; }
        let h = '<h2 style="color:#58a6ff;margin-top:16px">📊 生成的图表</h2>';
        d.images.forEach(img=>{
          h += '<div style="margin:10px 0"><div class="note">'+img.name+'</div>'
             + '<a href="'+img.url+'" target="_blank">'
             + '<img src="'+img.url+'" style="max-width:100%;border:1px solid var(--line);border-radius:10px"></a></div>';
        });
        reportsEl.innerHTML = h;
      });
    }
    function attach(taskId, name){
      consoleEl.textContent = '';
      reportsEl.innerHTML = '';
      resetLive();
      stateEl.innerHTML = '<span class="spin"></span><span class="status-running">运行中：'+name+'</span>';
      setButtons(true);
      if(evtSource) evtSource.close();
      evtSource = new EventSource('/stream/'+taskId);
      evtSource.onmessage = e=>{
        const txt = JSON.parse(e.data);
        consoleEl.textContent += txt;
        consoleEl.scrollTop = consoleEl.scrollHeight;
        feedChart(txt);
      };
      evtSource.addEventListener('done', e=>{
        const st = JSON.parse(e.data);
        stateEl.innerHTML = st==='done'
          ? '<span class="status-done">✓ 已完成</span>'
          : '<span class="status-error">✗ 出错（详见上方输出）</span>';
        setButtons(false);
        evtSource.close();
        showReports(taskId);
      });
      evtSource.onerror = ()=>{ setButtons(false); };
    }
    document.querySelectorAll('form[data-action]').forEach(form=>{
      form.addEventListener('submit', async ev=>{
        ev.preventDefault();
        const action = form.dataset.action;
        const res = await fetch('/run/'+action, {method:'POST', body:new FormData(form)});
        const data = await res.json();
        if(!data.ok){ stateEl.innerHTML='<span class="status-error">'+data.error+'</span>'; return; }
        attach(data.task_id, data.name);
      });
    });
    // 页面加载时若已有任务在跑，自动挂接
    fetch('/task_status').then(r=>r.json()).then(d=>{ if(d.busy) attach(d.task_id, d.name); });
    </script>"""

    body = (f"<h1>⚙ 操作中心</h1>"
            f"<p class='note'>所有命令行菜单操作均可在此执行，耗时任务（训练/回测）会在右侧控制台实时滚动日志。"
            f"同一时刻只允许一个任务运行。</p>"
            f"<div class='control-layout'>"
            f"  <div class='control-actions'>"
            f"    <h2 style='margin-top:8px'>训练</h2><div class='grid'>{train_panels}</div>"
            f"    <h2>回测</h2><div class='grid'>{bt_panels}</div>"
            f"    <h2>工具</h2><div class='grid'>{tool_panels}</div>"
            f"  </div>"
            f"  <div class='control-side'>{console}</div>"
            f"</div>"
            f"{script}")
    return _page("操作中心", body, active="control")


def start_dashboard():
    """启动 Flask 看板服务。"""
    print(f"[AlphaQuant] 看板启动: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False,
            use_reloader=False, threaded=True)
