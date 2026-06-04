# Flask 本地看板：账户总览、持仓、信号、历史交易、模型表现

import os
import csv
import json
import logging

from flask import Flask, render_template_string

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGS_DIR, DASHBOARD_HOST, DASHBOARD_PORT, INIT_CAPITAL

logger = logging.getLogger(__name__)
app    = Flask(__name__)

# ── 数据读取 ──────────────────────────────────────────

def _read_account_state() -> dict:
    state_file = os.path.join(LOGS_DIR, "account_state.json")
    if not os.path.exists(state_file):
        return {"cash": INIT_CAPITAL, "holdings": {}}
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


def _read_all_trades() -> list:
    trades = []
    for fn in sorted(os.listdir(LOGS_DIR)):
        if fn.startswith("trades_") and fn.endswith(".csv"):
            with open(os.path.join(LOGS_DIR, fn), encoding="utf-8") as f:
                trades.extend(list(csv.DictReader(f)))
    return trades


# ── 基础布局 ──────────────────────────────────────────

_NAV = """
<nav style="background:#161b22;padding:12px 24px;display:flex;gap:24px;border-bottom:1px solid #30363d">
  <span style="color:#f0f6fc;font-weight:bold">AlphaQuant</span>
  <a href="/" style="color:#58a6ff;text-decoration:none">账户总览</a>
  <a href="/holdings" style="color:#58a6ff;text-decoration:none">当前持仓</a>
  <a href="/signals" style="color:#58a6ff;text-decoration:none">今日信号</a>
  <a href="/trades" style="color:#58a6ff;text-decoration:none">历史交易</a>
  <a href="/performance" style="color:#58a6ff;text-decoration:none">模型表现</a>
</nav>"""

_STYLE = """
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;margin:0}
.wrap{max-width:1200px;margin:24px auto;padding:0 16px}
h1{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:8px}
h2{color:#58a6ff}
table{width:100%;border-collapse:collapse;margin:16px 0}
th{background:#161b22;color:#8b949e;padding:8px 12px;text-align:left}
td{padding:8px 12px;border-bottom:1px solid #21262d}
tr:hover td{background:#161b22}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;
      padding:16px;margin:8px;display:inline-block;min-width:180px;vertical-align:top}
.val{font-size:1.8em;font-weight:bold;color:#f0f6fc}
.lbl{color:#8b949e;font-size:.85em}
.up{color:#3fb950}.down{color:#f85149}
.bar{display:inline-block;background:#1f6feb;height:10px;border-radius:2px;vertical-align:middle}
</style>"""


def _page(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title>{_STYLE}</head><body>{_NAV}<div class='wrap'>{body}</div></body></html>"


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

    cards = f"""
    <div>
      <div class="card"><div class="lbl">虚拟总资产</div><div class="val">¥{total:,.0f}</div></div>
      <div class="card"><div class="lbl">现金余额</div><div class="val">¥{state.get('cash',0):,.0f}</div></div>
      <div class="card"><div class="lbl">持仓市值</div><div class="val">¥{state.get('holding_value',0):,.0f}</div></div>
      <div class="card">
        <div class="lbl">今日盈亏</div>
        <div class="val {'up' if today_pnl>=0 else 'down'}">{'+'if today_pnl>=0 else ''}¥{today_pnl:,.0f}</div>
      </div>
      <div class="card">
        <div class="lbl">累计盈亏</div>
        <div class="val {'up' if cum_pnl>=0 else 'down'}">
          {'+'if cum_pnl>=0 else ''}¥{cum_pnl:,.0f}<br>
          <span style="font-size:.6em">({'+'if cum_pct>=0 else ''}{cum_pct:.2f}%)</span>
        </div>
      </div>
    </div>"""

    chart = f"""
    <h2>资产净值曲线（近90日）</h2>
    <canvas id="c" width="900" height="300"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
    <script>
    new Chart(document.getElementById('c'),{{
      type:'line',
      data:{{labels:{nav_dates},datasets:[{{label:'总资产',data:{nav_vals},
        borderColor:'#58a6ff',fill:false,tension:.1,pointRadius:2}}]}},
      options:{{scales:{{
        y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
        x:{{ticks:{{color:'#8b949e',maxTicksLimit:12}},grid:{{color:'#21262d'}}}}
      }},plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}}}}
    }});
    </script>"""

    return _page("AlphaQuant 看板", f"<h1>账户总览</h1>{cards}{chart}")


@app.route("/holdings")
def holdings():
    state = _read_account_state()
    rows  = ""
    for code, pos in state.get("holdings", {}).items():
        shares = pos.get("shares", 0)
        cost   = pos.get("cost", 0)
        curr   = pos.get("current_price", cost)
        pnl    = (curr - cost) * shares
        pct    = (curr - cost) / max(cost, 0.01) * 100
        cls    = "up" if pnl >= 0 else "down"
        rows += (f"<tr><td>{code}</td><td>{shares}</td><td>¥{cost:.2f}</td>"
                 f"<td>¥{curr:.2f}</td>"
                 f"<td class='{cls}'>{'+'if pnl>=0 else ''}¥{pnl:,.0f} ({pct:+.1f}%)</td>"
                 f"<td>{pos.get('buy_date','')}</td></tr>")

    body = (f"<h1>当前持仓</h1>"
            f"<table><thead><tr><th>代码</th><th>持仓数量</th><th>成本价</th>"
            f"<th>当前价</th><th>浮动盈亏</th><th>买入日期</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("持仓", body)


@app.route("/signals")
def signals():
    try:
        from paper_trading.executor import get_current_signals
        sigs = get_current_signals()
    except Exception as e:
        sigs = {}
        logger.error(f"信号获取失败: {e}")

    rows = ""
    for code, prob in sorted(sigs.items(), key=lambda x: -x[1]):
        bar_w  = int(prob * 120)
        sug    = "★ 买入" if prob > 0.65 else ("▼ 卖出" if prob < 0.35 else "— 观望")
        cls    = "up" if prob > 0.65 else ("down" if prob < 0.35 else "")
        rows += (f"<tr><td>{code}</td>"
                 f"<td><span class='bar' style='width:{bar_w}px'></span> {prob:.1%}</td>"
                 f"<td class='{cls}'>{sug}</td></tr>")

    body = (f"<h1>今日信号</h1>"
            f"<p style='color:#8b949e'>计算中，首次加载可能较慢...</p>"
            f"<table><thead><tr><th>股票代码</th><th>买入概率</th><th>建议</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("今日信号", body)


@app.route("/trades")
def trades():
    all_t = _read_all_trades()
    rows  = ""
    for t in reversed(all_t[-300:]):
        cls  = "up" if t.get("action") == "buy" else "down"
        rows += (f"<tr><td>{t.get('time','')}</td><td>{t.get('code','')}</td>"
                 f"<td class='{cls}'>{t.get('action','')}</td>"
                 f"<td>¥{t.get('price','')}</td><td>{t.get('shares','')}</td>"
                 f"<td>{t.get('reason','')}</td></tr>")

    body = (f"<h1>历史交易记录（最近300笔）</h1>"
            f"<table><thead><tr><th>时间</th><th>代码</th><th>操作</th>"
            f"<th>价格</th><th>数量</th><th>原因</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return _page("历史交易", body)


@app.route("/performance")
def performance():
    all_t  = _read_all_trades()
    buy_map = {t["code"]: float(t["price"]) for t in all_t if t.get("action") == "buy"}
    sells   = [t for t in all_t if t.get("action") == "sell"]
    wins    = sum(1 for t in sells if float(t.get("price", 0)) > buy_map.get(t["code"], 9e9))
    total   = len(sells)
    wr      = wins / total * 100 if total else 0

    body = (f"<h1>模型表现</h1>"
            f"<div>"
            f"<div class='card'><div class='lbl'>信号胜率</div><div class='val'>{wr:.1f}%</div></div>"
            f"<div class='card'><div class='lbl'>已完成交易</div><div class='val'>{total} 笔</div></div>"
            f"<div class='card'><div class='lbl'>盈利次数</div><div class='val up'>{wins}</div></div>"
            f"<div class='card'><div class='lbl'>亏损次数</div><div class='val down'>{total-wins}</div></div>"
            f"</div>")
    return _page("模型表现", body)


def start_dashboard():
    """启动 Flask 看板服务。"""
    print(f"[AlphaQuant] 看板启动: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)
