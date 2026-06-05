# AlphaQuant 主入口：交互式菜单 + 命令行两种使用方式

import argparse
import logging
import sys
import os

from config import LOGS_DIR as _LOGS_DIR
_log_file = os.path.join(_LOGS_DIR, "alphaquant.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),  # 详细日志写文件
        logging.StreamHandler(),                            # 终端只显示 WARNING+
    ],
)
# 终端只显示 WARNING 及以上，避免刷屏；详细 INFO 看 logs/alphaquant.log
# 按类型找控制台 handler，避免硬编码下标 [1]（若 logging 已被预先配置，
# basicConfig 会变成空操作、handlers 数量不定，硬编码下标会 IndexError）
for _h in logging.getLogger().handlers:
    if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
        _h.setLevel(logging.WARNING)
logger = logging.getLogger("main")

# ── 菜单工具 ─────────────────────────────────────────────────────────

_W = 50  # 菜单宽度（字节宽，中文占2字节）

def _box_top():  print("╔" + "═" * (_W - 2) + "╗")
def _box_sep():  print("╠" + "═" * (_W - 2) + "╣")
def _box_bot():  print("╚" + "═" * (_W - 2) + "╝")

def _box_line(s: str = ""):
    # 按字节宽度计算填充（中文2字节，英文1字节）
    byte_w = len(s.encode("utf-8", errors="replace")) - len(s.encode("ascii", errors="ignore")) // 2
    try:
        from wcwidth import wcswidth
        display_w = wcswidth(s)
        display_w = display_w if display_w >= 0 else len(s)
    except ImportError:
        display_w = sum(2 if ord(c) > 127 else 1 for c in s)
    pad = _W - 4 - display_w
    print(f"║  {s}{' ' * max(pad, 0)}║")

def _clear():
    os.system("cls" if os.name == "nt" else "clear")

def _pause():
    input("\n  按 Enter 返回菜单...")

def _ask(prompt: str, default: str = "") -> str:
    tip = f"（回车默认 {default}）" if default else ""
    val = input(f"  {prompt}{tip}：").strip()
    return val if val else default

def _confirm(prompt: str) -> bool:
    return _ask(prompt + " (yes/no)", "no").lower() == "yes"


def _print_menu():
    _clear()
    _box_top()
    _box_line()
    _box_line("    AlphaQuant — A股量化 AI 系统")
    _box_line()
    _box_sep()
    _box_line("  【训练 & 回测】")
    _box_line("  1  训练模型（完整模式，全部股票）")
    _box_line("  2  训练模型（快速模式，随机10只/近2年）")
    _box_line("  3  继续训练（在已有模型基础上增量学习）")
    _box_line("  f  集成训练（多模型，提升稳定性）")
    _box_line("  4  历史回测")
    _box_sep()
    _box_line("  【实盘 & 看板】")
    _box_line("  5  启动模拟盘调度器")
    _box_line("  6  启动网页看板")
    _box_line("  7  查看今日信号排行")
    _box_sep()
    _box_line("  【工具】")
    _box_line("  8  个股诊断")
    _box_line("  9  单股买卖点图（K线 + 模型信号）")
    _box_line("  p  持仓一键诊断（模拟盘所有持仓）")
    _box_line("  v  查看训练/回测报告图表")
    _box_line("  e  导出今日信号到 Excel")
    _box_line("  c  查看当前配置参数")
    _box_line("  r  重置虚拟账户")
    _box_line()
    _box_line("  0  退出")
    _box_bot()
    print()


# ── 各功能 ───────────────────────────────────────────────────────────

def _load_training_stocks(force_refresh: bool = False) -> dict:
    """
    完整训练的数据加载：优先使用本地缓存池（离线、数据更多、不触发限流），
    缓存不足或强制刷新时回退到联网随机抽样。
    """
    from config import PREFER_CACHED_POOL, MAX_TRAIN_STOCKS
    from data.loader import load_cached_stocks, load_all_stocks
    if PREFER_CACHED_POOL and not force_refresh:
        # 只用沪深主板(sh/sz)：北交所流动性差、走势特殊，混入会拖累模型质量
        cached = load_cached_stocks(limit=MAX_TRAIN_STOCKS, boards=("sh", "sz"))
        if len(cached) >= 50:
            logger.info(f"训练数据来源：本地缓存池 {len(cached)} 只 sh/sz 主板（离线、数据多、不限流）")
            return cached
        logger.warning(f"本地缓存仅 {len(cached)} 只，不足以训练，回退到联网随机抽样")
    return load_all_stocks(force_refresh=force_refresh, quick=False)


def run_train(quick: bool = False, force_refresh: bool = False, resume: bool = False):
    from data.loader import load_all_stocks
    from features.builder import build_all_stocks
    from models.trainer import train_model

    mode_tag = "快速模式" if quick else ("增量训练" if resume else "完整模式")
    logger.info(f"=== 训练模型 [{mode_tag}] ===")

    logger.info("正在下载/加载数据...")
    if quick:
        stock_data = load_all_stocks(force_refresh=force_refresh, quick=True)
    else:
        stock_data = _load_training_stocks(force_refresh=force_refresh)

    if not stock_data:
        logger.error("未获取到任何数据，请检查网络连接")
        return

    logger.info(f"共加载 {len(stock_data)} 只股票，正在构建特征...")
    X, y, scaler, dates = build_all_stocks(stock_data, fit_scaler=True)

    if len(X) == 0:
        logger.error("特征构建失败，数据不足（每只股票至少需要 40 条记录）")
        return

    logger.info(f"特征维度: {X.shape}，正样本比例: {y.mean():.3f}")
    logger.info("开始训练...")
    train_model(X, y, resume=resume, dates=dates)
    logger.info("训练完成！")


def run_backtest(start_str: str = None, end_str: str = None, params: dict = None):
    """历史回测。start_str/end_str 为 None 时进入交互问答（命令行），
    传入则直接使用（供网页调用，避免阻塞 stdin）。
    params 为可选的回测参数覆盖（买卖阈值/止损止盈等），供网页参数面板使用。"""
    import pandas as pd
    from data.loader import load_all_stocks
    from features.builder import load_scaler
    from models.lstm_model import load_best_available
    from backtest.engine import BacktestEngine
    from data.index_fetcher import fetch_all_benchmarks
    from config import MODEL_SAVE_DIR, START_DATE, RELATIVE_RANK_MODE, WINDOW_SIZE

    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        logger.error("模型文件不存在，请先训练模型（选项 1 或 2）")
        return

    mode_tag = "相对排名模式" if RELATIVE_RANK_MODE else "绝对阈值模式"
    logger.info(f"=== 历史回测 [{mode_tag}] ===")

    from datetime import datetime
    _default_end   = datetime.today().strftime("%Y%m%d")
    _default_start = START_DATE.replace("-", "")

    if start_str is None:
        print()
        start_str = _ask("回测起始日期（格式 YYYYMMDD，默认 20150101）", _default_start)
    if end_str is None:
        end_str = _ask(f"回测结束日期（格式 YYYYMMDD，默认 {_default_end}）", _default_end)
    start_str = start_str or _default_start
    end_str   = end_str or _default_end

    for _label, _val in [("起始日期", start_str), ("结束日期", end_str)]:
        try:
            datetime.strptime(_val, "%Y%m%d")
        except ValueError:
            print(f"  ✗ {_label} 格式错误：{_val}，应为 YYYYMMDD")
            return

    start_dt = pd.Timestamp(datetime.strptime(start_str, "%Y%m%d"))
    end_dt   = pd.Timestamp(datetime.strptime(end_str,   "%Y%m%d"))

    logger.info("正在拉取5个基准指数历史数据...")
    benchmarks = fetch_all_benchmarks(start_str, end_str)
    if benchmarks:
        logger.info(f"成功加载基准指数: {list(benchmarks.keys())}")
    else:
        logger.warning("基准指数拉取失败，将跳过对比图")

    # 优先使用本地缓存（避免每次回测都重新下载），再按日期筛选
    print("\n  正在加载股票数据（优先使用缓存）...")
    stock_data_all = load_all_stocks()
    stock_data = {}
    for code, df in stock_data_all.items():
        filtered = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
        if len(filtered) >= WINDOW_SIZE + 5:
            stock_data[code] = filtered
    print(f"  日期筛选后：{len(stock_data)} 只股票在回测区间内有足够数据")

    if not stock_data:
        print("  ✗ 无有效数据，请先运行训练（选项 1）下载股票数据后再回测")
        return

    model  = load_best_available()
    scaler = load_scaler()
    if params:
        logger.info(f"使用自定义回测参数: {params}")
    BacktestEngine(stock_data, model, scaler, benchmarks=benchmarks, params=params).run()
    logger.info("回测完成")


def run_paper():
    logger.info("=== 模拟盘调度器（Ctrl+C 停止）===")
    from paper_trading.scheduler import start_scheduler
    start_scheduler()


def run_dashboard():
    from config import DASHBOARD_HOST, DASHBOARD_PORT
    logger.info("=== 网页看板 ===")
    print(f"\n  看板地址：http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("  在浏览器中打开上述地址，Ctrl+C 停止\n")
    from dashboard.app import start_dashboard
    start_dashboard()


def run_signal():
    from paper_trading.executor import get_current_signals
    from data.loader import get_stock_name
    from config import BUY_THRESHOLD, SELL_THRESHOLD

    logger.info("=== 今日信号排行 ===")
    signals = get_current_signals()
    if not signals:
        logger.error("未获取到信号，请先训练模型并运行一次模拟盘")
        return

    print(f"\n  {'代码':<14} {'名称':<8} {'买入概率':>10} {'建议':>10}")
    print("  " + "─" * 48)
    for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
        name = get_stock_name(code)[:6]
        sug  = "★ 买入" if prob > BUY_THRESHOLD else ("▼ 观望" if prob < SELL_THRESHOLD else "─ 持仓")
        print(f"  {code:<14} {name:<8} {prob:>10.1%} {sug:>10}")
    print()


def run_diagnose(stock_input: str = None, cost: float = None):
    from diagnose.analyzer import diagnose
    from diagnose.report import print_report, print_batch_summary

    if stock_input is None:
        print()
        stock_input = _ask("输入股票代码（多只用逗号分隔，如 sh600519,sz000858）")
        if not stock_input:
            print("  已取消")
            return
        cost_str = _ask("持仓成本价（无则直接回车）")
        cost = float(cost_str) if cost_str else None

    codes = [s.strip() for s in stock_input.split(",") if s.strip()]
    results = []
    for code in codes:
        try:
            r = diagnose(code, holdings_cost=cost)
            print_report(r)
            results.append(r)
        except Exception as e:
            logger.error(f"[{code}] 诊断失败: {e}")
    if len(results) > 1:
        print_batch_summary(results)


def run_single_backtest(code: str = None, years: int = None):
    """单股回测：绘制K线图 + 模型买卖点标注。
    code/years 为 None 时进入交互问答；传入则直接使用（供网页调用）。"""
    from data.loader import load_stock_data, _fetch_kline, get_stock_name
    from features.builder import build_sequences
    from models.lstm_model import load_best_available
    from models.trainer import predict_proba
    from config import MODEL_SAVE_DIR, REPORTS_DIR, START_DATE, BUY_THRESHOLD, SELL_THRESHOLD

    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        print("\n  ⚠ 模型文件不存在，请先训练模型\n")
        return

    if code is None:
        print()
        code = _ask("输入股票代码（如 sh600519）").strip()
    if not code:
        return
    years_str = str(years) if years is not None else _ask("回测多少年历史（默认 2）", "2")
    try:
        years = int(years_str)
    except ValueError:
        years = 2

    from datetime import datetime, timedelta
    end_str   = datetime.today().strftime("%Y%m%d")
    start_str = (datetime.today() - timedelta(days=years * 365)).strftime("%Y%m%d")

    print(f"\n  正在加载 {code} 数据...")
    df = _fetch_kline(code, start_str, end_str)
    if df is None or df.empty:
        print(f"\n  ✗ 无法获取 {code} 的数据，请检查代码是否正确\n")
        return

    print(f"  获取到 {len(df)} 条，正在计算模型信号...")
    try:
        X, _, _, dates = build_sequences(df)
    except Exception as e:
        print(f"\n  ✗ 特征构建失败: {e}\n")
        return

    if len(X) == 0:
        print("\n  ✗ 数据量不足以构建特征序列\n")
        return

    model = load_best_available()
    probs = predict_proba(model, X)

    # 对齐日期
    import numpy as np
    import pandas as pd
    date_series = pd.to_datetime(dates)
    prob_df = pd.DataFrame({"date": date_series, "prob": probs})
    merged  = df.merge(prob_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

    buy_pts  = merged[merged["prob"] >= BUY_THRESHOLD]
    sell_pts = merged[merged["prob"] <= SELL_THRESHOLD]

    # ── 绘图 ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from utils.viz import setup_chinese_font
        setup_chinese_font()

        stock_name = get_stock_name(code)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)

        # ── 收盘价走势 ──
        ax1.plot(merged["date"], merged["close"], color="#455A64", linewidth=1.2, label="收盘价")
        if not buy_pts.empty:
            ax1.scatter(buy_pts["date"], buy_pts["close"],
                        color="#E53935", marker="^", s=80, zorder=5, label=f"买入信号(≥{BUY_THRESHOLD})")
        if not sell_pts.empty:
            ax1.scatter(sell_pts["date"], sell_pts["close"],
                        color="#1E88E5", marker="v", s=80, zorder=5, label=f"卖出信号(≤{SELL_THRESHOLD})")

        ax1.set_title(f"{stock_name}（{code}）买卖点分析  |  共 {len(buy_pts)} 次买入信号，{len(sell_pts)} 次卖出信号",
                      fontsize=13, fontweight="bold")
        ax1.set_ylabel("价格（元）")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(alpha=0.25)

        # ── 模型概率 ──
        ax2.plot(merged["date"], merged["prob"], color="#7B1FA2", linewidth=1.0, label="买入概率")
        ax2.axhline(BUY_THRESHOLD,  color="#E53935", linestyle="--", alpha=0.7, label=f"买入线 {BUY_THRESHOLD}")
        ax2.axhline(SELL_THRESHOLD, color="#1E88E5", linestyle="--", alpha=0.7, label=f"卖出线 {SELL_THRESHOLD}")
        ax2.fill_between(merged["date"], merged["prob"], BUY_THRESHOLD,
                         where=merged["prob"] >= BUY_THRESHOLD, alpha=0.25, color="#E53935")
        ax2.fill_between(merged["date"], merged["prob"], SELL_THRESHOLD,
                         where=merged["prob"] <= SELL_THRESHOLD, alpha=0.25, color="#1E88E5")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("买入概率")
        ax2.set_xlabel("日期")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.25)

        plt.tight_layout()
        out_path = os.path.join(REPORTS_DIR, f"signal_{code}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  图表已保存: {out_path}")

        # 尝试自动打开
        import subprocess, sys as _sys
        try:
            if os.name == "nt":
                os.startfile(out_path)
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", out_path])
            else:
                subprocess.Popen(["xdg-open", out_path])
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"绘图失败: {e}")

    print()


def run_view_reports():
    """列出并用系统默认程序打开已生成的报告图表。"""
    from config import REPORTS_DIR
    import glob

    pngs = sorted(glob.glob(os.path.join(REPORTS_DIR, "*.png")))
    if not pngs:
        print("\n  暂无报告文件，请先运行训练或回测\n")
        return

    print("\n  已生成的报告文件：")
    for i, p in enumerate(pngs, 1):
        size_kb = os.path.getsize(p) // 1024
        print(f"  {i}. {os.path.basename(p)}  ({size_kb} KB)  {p}")

    choice = _ask(f"\n  输入编号打开（1-{len(pngs)}），直接回车全部打开", "all")
    import subprocess, sys as _sys

    def _open(path):
        try:
            if os.name == "nt":
                os.startfile(path)
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            print(f"  已打开: {os.path.basename(path)}")
        except Exception as e:
            print(f"  无法自动打开，请手动查看: {path}")

    if choice == "all":
        for p in pngs:
            _open(p)
    elif choice.isdigit() and 1 <= int(choice) <= len(pngs):
        _open(pngs[int(choice) - 1])
    else:
        print("  无效输入")


def run_export_excel():
    """导出今日信号到 Excel 文件（Sheet1: 今日信号，Sheet2: 交易明细）。"""
    from data.loader import load_all_stocks, get_stock_name
    from features.builder import build_sequences
    from models.lstm_model import load_best_available
    from models.trainer import predict_proba
    from config import MODEL_SAVE_DIR, REPORTS_DIR, BUY_THRESHOLD, SELL_THRESHOLD

    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        print("\n  模型文件不存在，请先训练模型\n")
        return

    print("\n  正在加载股票数据（快速模式）...")
    stock_data = load_all_stocks(quick=True)
    if not stock_data:
        print("  未能加载任何股票数据，请检查网络连接\n")
        return

    print(f"  已加载 {len(stock_data)} 只股票，正在推理信号...")
    model = load_best_available()
    signals = {}
    for code, df in stock_data.items():
        if df is None or len(df) < 35:
            continue
        try:
            X, _, _, _ = build_sequences(df)
            if len(X) == 0:
                continue
            probs = predict_proba(model, X[-1:])
            signals[code] = float(probs[0])
        except Exception:
            pass

    if not signals:
        print("  未获取到有效信号\n")
        return

    from datetime import datetime
    today = datetime.today().strftime("%Y%m%d")
    out_path = os.path.join(REPORTS_DIR, f"signals_{today}.xlsx")

    rows = []
    for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
        name = get_stock_name(code)
        if prob >= BUY_THRESHOLD:
            suggestion = "买入"
        elif prob <= SELL_THRESHOLD:
            suggestion = "卖出"
        else:
            suggestion = "持仓"
        rows.append({"代码": code, "名称": name, "买入概率": round(prob, 4),
                     "建议": suggestion, "日期": today})

    try:
        from openpyxl import Workbook
        wb = Workbook()

        # Sheet 1: 今日信号
        ws1 = wb.active
        ws1.title = "今日信号"
        headers1 = ["代码", "名称", "买入概率", "建议", "日期"]
        ws1.append(headers1)
        for row in rows:
            ws1.append([row[h] for h in headers1])

        # Sheet 2: 交易明细
        ws2 = wb.create_sheet("交易明细")
        trade_log = os.path.join(REPORTS_DIR, "trade_log.csv")
        if os.path.exists(trade_log):
            import csv
            with open(trade_log, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for line in reader:
                    ws2.append(line)
        else:
            ws2.append(["暂无交易记录"])

        wb.save(out_path)
        print(f"\n  已保存到: {out_path}\n")

    except ImportError:
        # fallback to CSV
        import csv
        csv_path = out_path.replace(".xlsx", ".csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["代码", "名称", "买入概率", "建议", "日期"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  openpyxl 未安装，已保存为 CSV: {csv_path}\n")


def run_train_ensemble():
    """训练集成模型（多个不同随机种子，取平均提升稳定性）。"""
    from data.loader import load_all_stocks
    from features.builder import build_all_stocks
    from models.trainer import train_ensemble
    from config import ENSEMBLE_N_MODELS

    logger.info(f"=== 集成训练（共 {ENSEMBLE_N_MODELS} 个模型）===")

    print("\n  正在加载数据...")
    stock_data = _load_training_stocks()
    X, y, _, dates = build_all_stocks(stock_data)

    if len(X) == 0:
        logger.error("特征构建失败，数据不足")
        return

    logger.info(f"特征维度: {X.shape}，正样本比例: {y.mean():.3f}")
    print(f"\n  数据加载完成，共 {len(X)} 条样本，开始训练 {ENSEMBLE_N_MODELS} 个模型...\n")
    train_ensemble(X, y, dates=dates)
    logger.info("集成训练完成！")


def run_portfolio_diagnose():
    """诊断模拟盘当前所有持仓。"""
    import json
    from config import LOGS_DIR
    from diagnose.analyzer import diagnose
    from diagnose.report import print_report, print_batch_summary

    state_file = os.path.join(LOGS_DIR, "account_state.json")
    if not os.path.exists(state_file):
        print("\n  虚拟账户文件不存在，请先启动模拟盘\n")
        return

    with open(state_file, encoding="utf-8") as f:
        state = json.load(f)

    holdings = state.get("holdings", {})
    if not holdings:
        print("\n  当前无持仓，无需诊断\n")
        return

    print(f"\n  正在诊断 {len(holdings)} 只持仓股票...\n")
    results = []
    for code, pos in holdings.items():
        cost = pos.get("cost")
        try:
            r = diagnose(code, holdings_cost=cost)
            print_report(r)
            results.append(r)
        except Exception as e:
            logger.error(f"[{code}] 诊断失败: {e}")

    if len(results) > 1:
        print_batch_summary(results)


def run_show_config():
    """展示当前 config.py 中所有可调参数。"""
    import config as _cfg

    fields = {
        "── 数据 ──────────────────────────": None,
        "START_DATE":         (_cfg.START_DATE,         "历史数据起始日期"),
        "WINDOW_SIZE":        (_cfg.WINDOW_SIZE,         "时间窗口大小（天）"),
        "FEATURE_DIM":        (_cfg.FEATURE_DIM,         "特征维度数"),
        "FULL_STOCK_COUNT":   (_cfg.FULL_STOCK_COUNT,    "完整模式随机抽取股票数"),
        "QUICK_STOCK_COUNT":  (_cfg.QUICK_STOCK_COUNT,   "快速模式随机抽取股票数"),
        "QUICK_HISTORY_YEARS":(_cfg.QUICK_HISTORY_YEARS, "快速模式历史年数"),
        "── 模型 ──────────────────────────": None,
        "LSTM_HIDDEN1":       (_cfg.LSTM_HIDDEN1,        "GRU第1层隐藏单元数"),
        "LSTM_HIDDEN2":       (_cfg.LSTM_HIDDEN2,        "GRU第2层隐藏单元数"),
        "DROPOUT":            (_cfg.DROPOUT,             "Dropout比率"),
        "LEARNING_RATE":      (_cfg.LEARNING_RATE,       "初始学习率"),
        "BATCH_SIZE":         (_cfg.BATCH_SIZE,          "训练批大小"),
        "MAX_EPOCHS":         (_cfg.MAX_EPOCHS,          "最大训练轮数"),
        "EARLY_STOP_PATIENCE":(_cfg.EARLY_STOP_PATIENCE, "早停耐心（轮）"),
        "CPU_THREAD_RATIO":   (_cfg.CPU_THREAD_RATIO,    "CPU线程比率（0=全用，0.5=一半）"),
        "── 回测 & 风控 ────────────────────": None,
        "INIT_CAPITAL":       (_cfg.INIT_CAPITAL,        "初始资金（元）"),
        "BUY_THRESHOLD":      (_cfg.BUY_THRESHOLD,       "买入概率阈值"),
        "SELL_THRESHOLD":     (_cfg.SELL_THRESHOLD,      "卖出概率阈值"),
        "STOP_LOSS_RATIO":    (_cfg.STOP_LOSS_RATIO,     "单股止损线"),
        "TAKE_PROFIT_RATIO":  (_cfg.TAKE_PROFIT_RATIO,   "单股止盈线"),
        "PORTFOLIO_STOP":     (_cfg.PORTFOLIO_STOP,      "组合止损线"),
        "MAX_HOLDINGS":       (_cfg.MAX_HOLDINGS,        "最大持仓数量"),
        "MAX_POSITION_RATIO": (_cfg.MAX_POSITION_RATIO,  "单股最大仓位比例"),
        "RELATIVE_RANK_MODE": (_cfg.RELATIVE_RANK_MODE,  "相对排名模式"),
        "TOP_N_BUY":          (_cfg.TOP_N_BUY,           "每日最多买入候选数"),
    }

    print("\n" + "=" * 62)
    print("  AlphaQuant 当前配置（修改 config.py 生效）")
    print("=" * 62)
    for key, val in fields.items():
        if val is None:
            print(f"\n  {key}")
            continue
        v, desc = val
        print(f"  {key:<22} {str(v):<12}  # {desc}")
    print("=" * 62)
    print("\n  提示：直接编辑 config.py 修改参数，重启程序后生效\n")


def reset_account():
    """实际执行账户重置（无交互，供网页/命令行复用）。"""
    import json
    from config import LOGS_DIR, INIT_CAPITAL

    state_file   = os.path.join(LOGS_DIR, "account_state.json")
    suspend_file = os.path.join(LOGS_DIR, "suspend_state.json")
    peak_file    = os.path.join(LOGS_DIR, "peak_assets.json")

    new_state = {
        "cash": float(INIT_CAPITAL),
        "holdings": {},
        "today_bought": [],
        "updated_at": __import__("datetime").datetime.now().isoformat(),
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(new_state, f, ensure_ascii=False, indent=2)

    for fp in [suspend_file, peak_file]:
        if os.path.exists(fp):
            os.remove(fp)

    print(f"\n  ✓ 账户已重置，初始资金 ¥{INIT_CAPITAL:,.0f}")


def run_reset():
    """命令行交互式重置（带确认）。"""
    from config import INIT_CAPITAL

    print()
    print(f"  ⚠️  此操作将重置虚拟账户：")
    print(f"     · 现金恢复为 ¥{INIT_CAPITAL:,.0f}")
    print(f"     · 清空所有持仓、暂停状态和峰值记录")
    print()

    if not _confirm("确认重置"):
        print("  已取消")
        return

    reset_account()


# ── 交互式菜单主循环 ──────────────────────────────────────────────────

def interactive_menu():
    handlers = {
        "1": lambda: run_train(quick=False, force_refresh=_confirm("是否强制重新下载数据")),
        "2": lambda: (_clear(), print("\n  快速模式：随机抽取 10 只股票，近 2 年历史\n"), run_train(quick=True)),
        "3": lambda: (_clear(), print("\n  增量训练：加载已有模型，在原基础上继续学习\n"), run_train(resume=True)),
        "f": run_train_ensemble,
        "4": run_backtest,
        "5": run_paper,
        "6": run_dashboard,
        "7": run_signal,
        "8": run_diagnose,
        "9": run_single_backtest,
        "p": run_portfolio_diagnose,
        "v": run_view_reports,
        "e": run_export_excel,
        "c": run_show_config,
        "r": run_reset,
    }

    while True:
        _print_menu()
        choice = input("  请输入选项编号：").strip()

        if choice == "0":
            print("\n  再见！\n")
            break

        if choice not in handlers:
            print("  无效选项，请重新输入")
            import time; time.sleep(1)
            continue

        print()
        try:
            handlers[choice]()
        except KeyboardInterrupt:
            print("\n  已中断")
        except Exception as e:
            logger.error(f"执行失败: {e}")

        if choice not in ("5", "8"):   # 这些选项操作完直接回菜单，不需要暂停
            _pause()


# ── 命令行模式（脚本 / 自动化使用）──────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        interactive_menu()
        return

    parser = argparse.ArgumentParser(
        description="AlphaQuant — A股量化AI系统（无参数直接运行进入交互菜单）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py --mode train\n"
            "  python main.py --mode train --quick\n"
            "  python main.py --mode train --refresh\n"
            "  python main.py --mode backtest\n"
            "  python main.py --mode diagnose --stock sh600519\n"
            "  python main.py --mode diagnose --stock sh600519 --cost 1650\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "paper", "dashboard", "signal", "diagnose", "reset"],
        required=True,
    )
    parser.add_argument("--stock",   type=str, default=None)
    parser.add_argument("--cost",    type=str, default=None)
    parser.add_argument("--refresh", action="store_true", help="强制重新下载数据（train）")
    parser.add_argument("--quick",   action="store_true", help="快速模式（train）")

    args = parser.parse_args()

    dispatch = {
        "train":     lambda: run_train(quick=args.quick, force_refresh=args.refresh),
        "backtest":  run_backtest,
        "paper":     run_paper,
        "dashboard": run_dashboard,
        "signal":    run_signal,
        "diagnose":  lambda: run_diagnose(
                         stock_input=args.stock,
                         cost=float(args.cost) if args.cost else None
                     ),
        "reset":     run_reset,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
