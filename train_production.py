# 训练「正式模型」：在流动性好的可交易池上、用全历史训练集成模型，并做诚实样本外回测。
# 训练池 = 回测池 = 将来实盘池（完全一致）。跑完 lstm_best.pt 即正式模型。
# 用法：python train_production.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger().handlers[0].setLevel(logging.WARNING)
log = logging.getLogger("prod"); log.setLevel(logging.INFO)
_h = logging.StreamHandler(); _h.setLevel(logging.INFO); log.addHandler(_h); log.propagate = False

from config import DATA_CACHE_DIR, ENSEMBLE_N_MODELS
from data.loader import get_tradeable_pool, load_cached_stocks
from features.builder import build_all_stocks
import models.trainer as T
from models.trainer import train_ensemble, _temporal_split_masks
from models.lstm_model import load_best_available
from backtest.engine import BacktestEngine

# 1) 可交易池（流动性最好的 100 只）——训练/回测/实盘统一用它
pool = get_tradeable_pool(limit=100, min_amount_yi=2.0)
log.info(f"可交易池 {len(pool)} 只（流动性最好），加载全历史...")
sd = {}
for code in pool:
    fp = os.path.join(DATA_CACHE_DIR, code + ".csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if len(df) >= 120:
            sd[code] = df
log.info(f"有效 {len(sd)} 只，构建特征...")
X, y, _, dates = build_all_stocks(sd)
log.info(f"X={X.shape} 正样本={y.mean():.3f}")

# 2) 集成训练（多模型取平均，更稳）。耐心给足，质量优先
T.MAX_EPOCHS = 80
T.EARLY_STOP_PATIENCE = 15
log.info(f"开始集成训练 {ENSEMBLE_N_MODELS} 个模型（约 30-45 分钟）...")
train_ensemble(X, y, dates=dates)
log.info("集成训练完成，模型已存为 lstm_best.pt + 集成清单")

# 3) 诚实样本外回测（用 config 当前风控默认值，含择时闸）
_, _, cutoff = _temporal_split_masks(dates)
log.info(f"训练截止日 = {cutoff.date()}，样本外回测从此日起")
bench = {}
mf = os.path.join(DATA_CACHE_DIR, "market_features.csv")
if os.path.exists(mf):
    m = pd.read_csv(mf, parse_dates=["date"])
    if "mkt_close" in m.columns:
        bench["沪深300"] = m[["date", "mkt_close"]].rename(columns={"mkt_close": "close"})

model = load_best_available()
metrics = BacktestEngine(sd, model, benchmarks=bench,
                         params={"start_date": cutoff.strftime("%Y-%m-%d")}).run()
print("\n" + "=" * 48)
print("  正式模型 · 诚实样本外回测")
print("=" * 48)
for k, v in metrics.items():
    print(f"  {k:<10} {v}")
print("=" * 48)
print("  ⚠️ 上线前必须先用模拟盘前向空跑数月验证，别直接上真钱")
