# 一键「诚实样本外回测」：用现成模型 + 本地缓存，只测训练截止日之后的未来段。
# 不重训、不联网。用法：python oos_backtest.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re, glob, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger().handlers[0].setLevel(logging.WARNING)   # 控制台只显示要点
log = logging.getLogger("oos")
log.setLevel(logging.INFO)
_h = logging.StreamHandler(); _h.setLevel(logging.INFO)
log.addHandler(_h); log.propagate = False

from config import DATA_CACHE_DIR
from features.builder import build_all_stocks
from models.trainer import _temporal_split_masks
from models.lstm_model import load_best_available
from backtest.engine import BacktestEngine

# 1) 选与训练同口径的 sh/sz 主板股票（约 120 只）
files = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "*.csv")))
codes = [os.path.splitext(os.path.basename(f))[0] for f in files]
codes = [c for c in codes if re.match(r"^(sh|sz)\d{6}$", c)]
sel = codes[::max(1, len(codes) // 120)][:120]
sd = {}
for c in sel:
    df = pd.read_csv(os.path.join(DATA_CACHE_DIR, c + ".csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) >= 120:
        sd[c] = df
log.info(f"加载 {len(sd)} 只 sh/sz 股票")

# 2) 求训练截止日（样本外回测的起点）
X, y, _, dates = build_all_stocks(sd)
_, _, cutoff = _temporal_split_masks(dates)
log.info(f"训练截止日 = {cutoff.date()}，样本外回测区间 = 该日 ~ 至今")

# 3) 基准（缓存里的沪深300）
bench = {}
mf = os.path.join(DATA_CACHE_DIR, "market_features.csv")
if os.path.exists(mf):
    m = pd.read_csv(mf, parse_dates=["date"])
    if "mkt_close" in m.columns:
        bench["沪深300"] = m[["date", "mkt_close"]].rename(columns={"mkt_close": "close"})

# 4) 用现成模型跑样本外回测（config 默认风控；start_date 让它只从截止日后开仓）
model = load_best_available()
metrics = BacktestEngine(sd, model, benchmarks=bench,
                         params={"start_date": cutoff.strftime("%Y-%m-%d")}).run()

print("\n" + "=" * 46)
print("  诚实样本外(OOS)回测结果")
print("=" * 46)
for k, v in metrics.items():
    print(f"  {k:<10} {v}")
print("=" * 46)
print("  图表已存到 reports/backtest_report.png")
