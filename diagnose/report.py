# 终端格式化输出诊断报告

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _prob_bar(prob: float, width: int = 10) -> str:
    """概率进度条：████░░ 样式。"""
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _level(prob: float) -> str:
    if prob >= 0.75: return "极高"
    if prob >= 0.65: return "高"
    if prob >= 0.50: return "中"
    if prob >= 0.35: return "低"
    return "极低"


def print_report(result: dict):
    """
    在终端打印格式化诊断报告。

    Args:
        result: diagnose.analyzer.diagnose() 的返回值
    """
    W = 44

    def line(s: str = "") -> str:
        # 中文字符占2个宽度，需要手动计算填充
        display_len = sum(2 if '一' <= c <= '鿿' or '　' <= c <= '〿'
                          or '＀' <= c <= '￯' else 1 for c in s)
        pad = W - 4 - display_len
        return f"║  {s}{' ' * max(pad, 0)}║"

    sep = "╠" + "═" * (W - 2) + "╣"
    top = "╔" + "═" * (W - 2) + "╗"
    bot = "╚" + "═" * (W - 2) + "╝"

    code  = result["stock_code"]
    name  = result.get("stock_name", code)
    price = result["current_price"]
    pct   = result["pct_change"]
    prob  = result["ai_prob"]
    trend = result["trend"]
    vol   = result["volume"]
    act   = result["action"]
    sl    = result["stop_loss_price"]
    tp    = result["take_profit_price"]
    main  = result["main_net_inflow"]
    pos   = act["position_pct"]

    pct_str  = f"{'+'if pct>=0 else ''}{pct:.2f}%"
    main_str = (f"净流入 +{main/1e8:.2f}亿" if main >= 1e4
                else f"净流出 {main/1e8:.2f}亿" if main <= -1e4
                else "数据不足")

    lines = [
        top,
        line("    AlphaQuant 单股诊断报告"),
        sep,
        line(f"股票：{name} ({code[2:]})"),
        line(f"当前价：{price:.2f}  涨跌：{pct_str}"),
        sep,
        line(f"AI买入概率：{prob*100:.0f}%  {_prob_bar(prob)}  {_level(prob)}"),
        line(f"趋势判断：{trend['description']}"),
        line(f"量能状态：{vol['description']}"),
        line(f"主力资金：{main_str}"),
        sep,
        line(f"操作建议：{act['action']}"),
        line(f"建议仓位：不超过总资产 {pos}%"),
        line(f"止损价格：{sl:,.2f}（-7%）"),
        line(f"止盈目标：{tp:,.2f}（+5%）"),
    ]

    if result.get("holdings_cost"):
        cost = result["holdings_cost"]
        pnl  = (price - cost) / cost * 100
        lines.append(sep)
        lines.append(line(f"持仓成本：{cost:.2f}  当前盈亏：{pnl:+.1f}%"))

    lines.append(bot)
    print("\n" + "\n".join(lines) + "\n")


def print_batch_summary(results: list):
    """批量诊断时打印简洁汇总表格。"""
    print("\n" + "=" * 72)
    print(f"  {'代码':<14} {'名称':<10} {'当前价':>8} {'涨跌':>8} {'AI概率':>8}  {'建议'}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: -x["ai_prob"]):
        code   = r["stock_code"]
        name   = r.get("stock_name", code)[:6]
        price  = r["current_price"]
        pct    = r["pct_change"]
        prob   = r["ai_prob"]
        action = r["action"]["action"]
        print(f"  {code:<14} {name:<10} {price:>8.2f} {pct:>+7.2f}% {prob:>8.1%}  {action}")
    print("=" * 72 + "\n")
