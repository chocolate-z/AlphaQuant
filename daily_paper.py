# 每日定时任务入口：①生长缓存(抓新票) ②增量刷新已缓存 sh/sz ③跑一次对齐后的模拟盘。
# 配 Windows 计划任务 / cron，在每个交易日收盘后(约 15:05)运行即可。
# 随着每天生长，候选池(已缓存的 sh/sz)会逐步覆盖全市场，并由模拟盘动态流动性过滤后下单。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import glob
import re
import logging

from config import LOGS_DIR, DATA_CACHE_DIR, GROW_CACHE_PER_DAY
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

from data.loader import fetch_all_stock_codes, load_stock_data

# 已缓存的 sh/sz 代码
cached_shsz = {
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(DATA_CACHE_DIR, "*.csv"))
}
cached_shsz = {c for c in cached_shsz if re.match(r"^(sh|sz)\d{6}$", c)}

# 1) 生长缓存：从全A股列表抓 GROW_CACHE_PER_DAY 只还没缓存的 sh/sz 新票
if GROW_CACHE_PER_DAY > 0:
    try:
        allc = [c for c in fetch_all_stock_codes() if re.match(r"^(sh|sz)\d{6}$", c)]
        new = [c for c in allc if c not in cached_shsz][:GROW_CACHE_PER_DAY]
        if new:
            log.info(f"生长缓存：新增 {len(new)} 只（候选池逐步覆盖全市场）...")
            for c in new:
                try:
                    load_stock_data(c)
                    cached_shsz.add(c)
                except Exception as e:
                    log.warning(f"[{c}] 新票抓取失败: {e}")
    except Exception as e:
        log.warning(f"生长缓存步骤跳过: {e}")

# 2) 增量刷新已缓存 sh/sz（缓存新鲜则秒过，过期才补最近缺失，无未来函数：只取到昨日收盘）
log.info(f"增量刷新 {len(cached_shsz)} 只 sh/sz 缓存...")
ok = 0
for c in sorted(cached_shsz):
    try:
        load_stock_data(c)
        ok += 1
    except Exception:
        pass
log.info(f"刷新完成 {ok}/{len(cached_shsz)}")

# 3) 跑一次对齐后的模拟盘（候选池=全部缓存 sh/sz，动态流动性过滤后下单）
from paper_trading.executor import run_daily_execution
run_daily_execution()
log.info("每日模拟盘执行结束")
