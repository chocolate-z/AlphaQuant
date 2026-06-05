# 训练「正式模型」：sh/sz 主板全历史训练 + 诚实样本外回测。
# 已应用全部验证过的稳定性改造：择时闸MA20 + 动态可交易池(point-in-time) + 无未来函数。
# 信号扫全训练池，下单由引擎按"昨日流动性"动态筛选。用法：python train_production.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger().handlers[0].setLevel(logging.WARNING)
log = logging.getLogger("prod"); log.setLevel(logging.INFO)
_h = logging.StreamHandler(); _h.setLevel(logging.INFO); log.addHandler(_h); log.propagate = False

from config import DATA_CACHE_DIR
from data.loader import load_cached_stocks
from features.builder import build_all_stocks
import models.trainer as T
from models.trainer import train_model, _temporal_split_masks
from models.lstm_model import load_best_available
from backtest.engine import BacktestEngine

# 1) 训练池：150 只 sh/sz 主板（信号扫这些，下单由引擎按昨日流动性动态过滤）
N = 150
sd = load_cached_stocks(limit=N, boards=("sh", "sz"))
log.info(f"训练池 {len(sd)} 只 sh/sz 主板，构建特征...")
X, y, _, dates = build_all_stocks(sd)
log.info(f"X={X.shape} 正样本={y.mean():.3f}")

# 2) 训练正式单模型（质量优先，靠早停；如需更稳可改 train_ensemble，约 3 倍时间）
T.MAX_EPOCHS = 60
T.EARLY_STOP_PATIENCE = 12
log.info("开始训练正式模型（约 30-40 分钟）...")
train_model(X, y, dates=dates, _save_name="lstm_best")
log.info("训练完成，正式模型已存为 lstm_best.pt")

# 3) 诚实样本外回测：择时闸 + 动态池 + 无未来函数（全用 config 默认）
_, _, cutoff = _temporal_split_masks(dates)
log.info(f"训练截止日 = {cutoff.date()}，样本外回测从此日起（模型没见过）")
bench = {}
mf = os.path.join(DATA_CACHE_DIR, "market_features.csv")
if os.path.exists(mf):
    m = pd.read_csv(mf, parse_dates=["date"])
    if "mkt_close" in m.columns:
        bench["沪深300"] = m[["date", "mkt_close"]].rename(columns={"mkt_close": "close"})

model = load_best_available()
metrics = BacktestEngine(sd, model, benchmarks=bench,
                         params={"start_date": cutoff.strftime("%Y-%m-%d")}).run()
print("\n" + "=" * 50)
print("  正式模型 · 诚实样本外回测（无未来函数）")
print("=" * 50)
for k, v in metrics.items():
    print(f"  {k:<10} {v}")
print("=" * 50)
print("  说明：择时闸MA20 + 动态可交易池 + 信号滞后1日(无未来函数)")
print("  ⚠️ 上线前务必先用模拟盘前向空跑数月验证，别直接上真钱")
