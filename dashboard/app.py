# Flask 本地看板：账户总览、持仓、信号（缓存读取）、历史交易（可筛选）、模型表现

import os
import csv
import json
import time
import logging
from datetime import datetime

from flask import Flask, request, Response, jsonify

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
*{box-sizing:border-box}
body{font-family:'Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;margin:0}
nav{background:#161b22;padding:12px 24px;display:flex;gap:20px;align-items:center;
    border-bottom:1px solid #30363d;position:sticky;top:0;z-index:100}
nav .brand{color:#f0f6fc;font-weight:bold;font-size:1.1em}
nav a{color:#58a6ff;text-decoration:none;font-size:.95em}
nav a:hover{color:#79c0ff;text-decoration:underline}
.wrap{max-width:1200px;margin:24px auto;padding:0 20px}
h1{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:8px;margin-top:0}
h2{color:#58a6ff;font-size:1.1em}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9em}
th{background:#161b22;color:#8b949e;padding:8px 12px;text-align:left;
   position:sticky;top:52px;z-index:10}
td{padding:8px 12px;border-bottom:1px solid #21262d}
tr:hover td{background:#161b22}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;
      padding:16px 20px;min-width:160px}
.val{font-size:1.7em;font-weight:bold;color:#f0f6fc;margin:4px 0}
.lbl{color:#8b949e;font-size:.82em}
.up{color:#3fb950}.down{color:#f85149}.neutral{color:#8b949e}
.bar{display:inline-block;background:#1f6feb;height:10px;border-radius:2px;vertical-align:middle}
.badge-buy{background:#1a4a2e;color:#3fb950;padding:2px 8px;border-radius:4px;font-size:.85em}
.badge-sell{background:#3d1a1a;color:#f85149;padding:2px 8px;border-radius:4px;font-size:.85em}
.badge-hold{background:#2a2a2a;color:#8b949e;padding:2px 8px;border-radius:4px;font-size:.85em}
input,select{background:#161b22;border:1px solid #30363d;color:#c9d1d9;
             padding:6px 10px;border-radius:4px;font-size:.9em}
button{background:#1f6feb;color:#fff;border:none;padding:6px 16px;
       border-radius:4px;cursor:pointer;font-size:.9em}
button:hover{background:#388bfd}
.filter-bar{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.note{color:#8b949e;font-size:.85em;margin:8px 0}
canvas{max-width:100%}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;margin:14px 0}
.panel h2{margin:0 0 4px 0;color:#f0f6fc;font-size:1em}
.panel .desc{color:#8b949e;font-size:.82em;margin:0 0 12px 0}
.panel form{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
button.danger{background:#a13232}button.danger:hover{background:#c44}
button:disabled{background:#30363d;color:#8b949e;cursor:not-allowed}
#console{background:#010409;border:1px solid #30363d;border-radius:8px;padding:14px;
  font-family:'Consolas','Courier New',monospace;font-size:.85em;color:#c9d1d9;
  white-space:pre-wrap;word-break:break-all;height:420px;overflow-y:auto;margin-top:8px}
.console-wrap{position:sticky;bottom:0}
.status-running{color:#d29922}.status-done{color:#3fb950}.status-error{color:#f85149}
.spin{display:inline-block;width:10px;height:10px;border:2px solid #d29922;border-top-color:transparent;
  border-radius:50%;animation:sp 0.8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
</style>"""

_NAV = """
<nav>
  <span class="brand">📈 AlphaQuant</span>
  <a href="/">账户总览</a>
  <a href="/holdings">当前持仓</a>
  <a href="/signals">今日信号</a>
  <a href="/trades">历史交易</a>
  <a href="/performance">模型表现</a>
  <a href="/control">⚙ 操作中心</a>
</nav>"""

_CHARTJS = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>'

def _page(title: str, body: str, auto_refresh: int = 0) -> str:
    refresh = f'<meta http-equiv="refresh" content="{auto_refresh}">' if auto_refresh else ""
    return (f"<!DOCTYPE html><html lang='zh-CN'><head>"
            f"<meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{title} — AlphaQuant</title>{refresh}{_CSS}</head>"
            f"<body>{_NAV}<div class='wrap'>{body}</div></body></html>")


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

    cards = (f"<div class='cards'>"
             + card("虚拟总资产", f"¥{total:,.0f}")
             + card("现金余额",   f"¥{state.get('cash',0):,.0f}")
             + card("持仓市值",   f"¥{state.get('holding_value',0):,.0f}")
             + card("今日盈亏",   f"{'+'if today_pnl>=0 else ''}¥{today_pnl:,.0f}", "up" if today_pnl>=0 else "down")
             + card("累计盈亏",   f"{'+'if cum_pnl>=0 else ''}¥{cum_pnl:,.0f} ({cum_pct:+.2f}%)", "up" if cum_pnl>=0 else "down")
             + "</div>")

    chart = f"""
    <h2>资产净值曲线（近90日）</h2>
    {_CHARTJS}
    <canvas id="navChart" height="100"></canvas>
    <script>
    new Chart(document.getElementById('navChart'),{{
      type:'line',
      data:{{labels:{nav_dates},datasets:[{{
        label:'总资产',data:{nav_vals},borderColor:'#58a6ff',
        fill:true,backgroundColor:'rgba(88,166,255,0.08)',tension:.2,pointRadius:1
      }}]}},
      options:{{
        scales:{{
          y:{{ticks:{{color:'#8b949e',callback:v=>'¥'+v.toLocaleString()}},grid:{{color:'#21262d'}}}},
          x:{{ticks:{{color:'#8b949e',maxTicksLimit:12}},grid:{{color:'#21262d'}}}}
        }},
        plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}}
      }}
    }});
    </script>"""

    return _page("账户总览", f"<h1>账户总览</h1>{cards}{chart}", auto_refresh=300)


@app.route("/holdings")
def holdings():
    from data.loader import get_stock_name
    state = _read_account_state()
    rows  = ""
    if not state.get("holdings"):
        rows = "<tr><td colspan='7' style='text-align:center;color:#8b949e'>当前无持仓</td></tr>"
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
        rows += (f"<tr><td><b>{code}</b></td><td style='color:#8b949e'>{name}</td>"
                 f"<td>{shares:,}</td>"
                 f"<td>¥{cost:.2f}</td><td>¥{curr:.2f}</td>"
                 f"<td class='{cls}'>{'+'if pnl>=0 else ''}¥{pnl:,.0f} ({pct:+.1f}%)</td>"
                 f"<td>{pos.get('buy_date','')}</td></tr>")

    cls_total = "up" if total_holding_pnl >= 0 else "down"
    summary = (f"<p style='margin:8px 0;color:#8b949e'>持仓总浮盈亏："
               f"<span class='{cls_total}'>{'+'if total_holding_pnl>=0 else ''}¥{total_holding_pnl:,.0f}</span></p>")
    body = (f"<h1>当前持仓</h1>{summary}"
            f"<table><thead><tr><th>代码</th><th>名称</th><th>持仓数量</th><th>成本价</th>"
            f"<th>当前价</th><th>浮动盈亏</th><th>买入日期</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("持仓", body, auto_refresh=120)


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
        rows = "<tr><td colspan='4' style='text-align:center;color:#8b949e'>暂无信号缓存，请先运行模拟盘或手动触发：python main.py --mode signal</td></tr>"
    for code, prob in sorted(sigs.items(), key=lambda x: -x[1]):
        bar_w = int(prob * 120)
        name  = get_stock_name(code)
        if prob > 0.65:
            badge = f"<span class='badge-buy'>★ 买入</span>"
        elif prob < 0.35:
            badge = f"<span class='badge-sell'>▼ 卖出</span>"
        else:
            badge = f"<span class='badge-hold'>— 观望</span>"
        rows += (f"<tr><td><b>{code}</b></td>"
                 f"<td style='color:#8b949e'>{name}</td>"
                 f"<td><span class='bar' style='width:{bar_w}px'></span> {prob:.1%}</td>"
                 f"<td>{badge}</td></tr>")

    body = (f"<h1>今日信号</h1>{note}"
            f"<table><thead><tr><th>股票代码</th><th>名称</th><th>买入概率</th><th>建议</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("今日信号", body)


@app.route("/trades")
def trades():
    filter_code = request.args.get("code", "").strip()
    filter_date = request.args.get("date", "").strip()

    all_t = _read_all_trades(filter_code=filter_code, filter_date=filter_date)

    filter_form = f"""
    <div class='filter-bar'>
      <form method='get' style='display:flex;gap:8px;flex-wrap:wrap;align-items:center'>
        <label style='color:#8b949e'>股票代码</label>
        <input name='code' placeholder='如 sh600519' value='{filter_code}' style='width:140px'>
        <label style='color:#8b949e'>日期</label>
        <input name='date' type='date' value='{filter_date}'>
        <button type='submit'>筛选</button>
        <a href='/trades' style='color:#8b949e;font-size:.9em'>清除筛选</a>
      </form>
      <span class='note'>共 {len(all_t)} 条记录</span>
    </div>"""

    rows = ""
    for t in reversed(all_t[-500:]):
        action = t.get("action", "")
        cls    = "up" if action == "buy" else "down"
        badge  = f"<span class='badge-buy'>买入</span>" if action == "buy" else f"<span class='badge-sell'>卖出</span>"
        rows += (f"<tr><td>{t.get('time','')}</td><td><b>{t.get('code','')}</b></td>"
                 f"<td>{badge}</td><td>¥{t.get('price','')}</td>"
                 f"<td>{t.get('shares','')}</td><td>{t.get('commission','')}</td>"
                 f"<td>{t.get('reason','')}</td></tr>")

    if not rows:
        rows = "<tr><td colspan='7' style='text-align:center;color:#8b949e'>暂无交易记录</td></tr>"

    body = (f"<h1>历史交易记录</h1>{filter_form}"
            f"<table><thead><tr><th>时间</th><th>代码</th><th>操作</th>"
            f"<th>价格</th><th>数量</th><th>手续费</th><th>原因</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("历史交易", body)


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
    <canvas id="wChart" height="80"></canvas>
    <script>
    new Chart(document.getElementById('wChart'),{{
      type:'bar',
      data:{{labels:{win_trend_dates},datasets:[{{
        label:'盈亏',data:{win_trend_vals},
        backgroundColor:ctx=>ctx.raw===1?'rgba(63,185,80,.7)':'rgba(248,81,73,.7)',
        borderWidth:0
      }}]}},
      options:{{
        scales:{{
          y:{{min:0,max:1,ticks:{{color:'#8b949e',stepSize:1}},grid:{{color:'#21262d'}}}},
          x:{{ticks:{{color:'#8b949e',maxTicksLimit:10}},grid:{{color:'#21262d'}}}}
        }},
        plugins:{{legend:{{display:false}}}}
      }}
    }});
    </script>"""

    return _page("模型表现", f"<h1>模型表现</h1>{cards}{chart}")


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
        "backtest":       lambda: main.run_backtest(f.get("start") or None, f.get("end") or None),
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


@app.route("/control")
def control():
    from config import START_DATE
    _today = datetime.today().strftime("%Y%m%d")
    _start = START_DATE.replace("-", "")

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
              "<label style='color:#8b949e'><input type='checkbox' name='refresh' value='1'> 强制重新下载数据</label>")
        + panel("train_quick")
        + panel("train_resume")
        + panel("train_ensemble")
    )

    # ── 回测类 ──
    bt_panels = (
        panel("backtest",
              f"<label style='color:#8b949e'>起始</label><input name='start' value='{_start}' style='width:110px'>"
              f"<label style='color:#8b949e'>结束</label><input name='end' value='{_today}' style='width:110px'>")
        + panel("single_bt",
                "<input name='code' placeholder='如 sh600519' style='width:130px'>"
                "<label style='color:#8b949e'>年数</label><input name='years' value='2' style='width:60px'>")
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
    <div class="console-wrap">
      <h2 id="taskTitle" style="color:#58a6ff">运行控制台</h2>
      <p class="note" id="taskState">空闲中——点击上方任意「执行」按钮开始</p>
      <div id="console">（任务输出会实时显示在这里）</div>
    </div>"""

    script = """
    <script>
    const consoleEl = document.getElementById('console');
    const stateEl   = document.getElementById('taskState');
    let evtSource = null;

    function setButtons(disabled){
      document.querySelectorAll('form[data-action] button').forEach(b=>b.disabled=disabled);
    }
    function attach(taskId, name){
      consoleEl.textContent = '';
      stateEl.innerHTML = '<span class="spin"></span><span class="status-running">运行中：'+name+'</span>';
      setButtons(true);
      if(evtSource) evtSource.close();
      evtSource = new EventSource('/stream/'+taskId);
      evtSource.onmessage = e=>{
        consoleEl.textContent += JSON.parse(e.data);
        consoleEl.scrollTop = consoleEl.scrollHeight;
      };
      evtSource.addEventListener('done', e=>{
        const st = JSON.parse(e.data);
        stateEl.innerHTML = st==='done'
          ? '<span class="status-done">✓ 已完成</span>'
          : '<span class="status-error">✗ 出错（详见上方输出）</span>';
        setButtons(false);
        evtSource.close();
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
            f"<p class='note'>所有命令行菜单操作均可在此执行，耗时任务（训练/回测）会在下方控制台实时滚动日志。"
            f"同一时刻只允许一个任务运行。</p>"
            f"<h2>训练</h2><div class='grid'>{train_panels}</div>"
            f"<h2>回测</h2><div class='grid'>{bt_panels}</div>"
            f"<h2>工具</h2><div class='grid'>{tool_panels}</div>"
            f"{console}{script}")
    return _page("操作中心", body)


def start_dashboard():
    """启动 Flask 看板服务。"""
    print(f"[AlphaQuant] 看板启动: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False,
            use_reloader=False, threaded=True)
