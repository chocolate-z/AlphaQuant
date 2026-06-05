# 每日定时任务入口：刷新全市场缓存(增量) + 跑一次对齐后的模拟盘。
# 配 Windows 计划任务 / cron，在每个交易日收盘后(约 15:05)运行即可。
# 用法：python daily_paper.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import glob
import re
import logging

from config import LOGS_DIR, DATA_CACHE_DIR, MAX_TRAIN_STOCKS
_log = os.path.join(LOGS_DIR, "daily_paper.log")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(_log, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("daily_paper")

from data.realtime import is_trade_day
if not is_trade_day():
    log.info("今日非交易日，跳过")
    raise SystemExit(0)

# 1) 增量刷新「可交易池规模」的 sh/sz 缓存，保证用最新数据决策（无未来函数：只取到昨日收盘）
from data.loader import load_stock_data, load_cached_stocks
universe = list(load_cached_stocks(boards=("sh", "sz"), limit=MAX_TRAIN_STOCKS))
log.info(f"刷新 {len(universe)} 只 sh/sz 缓存（增量）...")
ok = 0
for code in universe:
    try:
        load_stock_data(code)   # 缓存新鲜则直接用，过期则增量补最近缺失
        ok += 1
    except Exception as e:
        log.warning(f"[{code}] 刷新失败: {e}")
log.info(f"缓存刷新完成 {ok}/{len(universe)}")

# 2) 跑一次对齐后的模拟盘（信号扫全池、择时闸、动态流动池、无未来函数）
from paper_trading.executor import run_daily_execution
run_daily_execution()
log.info("每日模拟盘执行结束")
