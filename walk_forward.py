# 滚动前向检验 v2：跨牛熊多周期，每折训练一次，对比「无择时闸 / 闸MA20 / 闸MA40」。
# 信号每折只算一次、三套配置复用，省时间。专看熊市(2018/2022)能否从大亏救回来。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import glob, logging, io, contextlib
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("wf"); log.setLevel(logging.INFO)
_h = logging.StreamHandler(); _h.setLevel(logging.INFO); log.addHandler(_h); log.propagate = False

from config import DATA_CACHE_DIR
from data.loader import load_cached_stocks
from features.builder import build_all_stocks
import models.trainer as T
from models.lstm_model import load_model
from backtest.engine import BacktestEngine

T.MAX_EPOCHS = 25
T.EARLY_STOP_PATIENCE = 8

bench_full = None
mfp = os.path.join(DATA_CACHE_DIR, "market_features.csv")
if os.path.exists(mfp):
    m = pd.read_csv(mfp, parse_dates=["date"])
    if "mkt_close" in m.columns:
        bench_full = m[["date", "mkt_close"]].rename(columns={"mkt_close": "close"})

def bench_ret(a, b):
    if bench_full is None: return None
    s = bench_full[(bench_full["date"] >= a) & (bench_full["date"] <= b)]
    return (s["close"].iloc[-1] / s["close"].iloc[0] - 1) * 100 if len(s) >= 2 else None

FULL = load_cached_stocks(limit=150, boards=("sh", "sz"))
log.info(f"股票池 {len(FULL)} 只 sh/sz 主板")

FOLDS = [
    ("2018-01-01", "2019-01-01", "2018熊市"),
    ("2020-01-01", "2021-01-01", "2020疫情V"),
    ("2022-01-01", "2023-01-01", "2022熊市"),
    ("2024-01-01", "2026-06-01", "2024-25牛市"),
]
CFGS = [
    ("无闸",   {"use_market_filter": False}),
    ("闸MA20", {"use_market_filter": True, "market_ma_days": 20}),
    ("闸MA40", {"use_market_filter": True, "market_ma_days": 40}),
]

rows = []
for cutoff_s, oos_end_s, tag in FOLDS:
    cutoff = pd.Timestamp(cutoff_s); oos_end = pd.Timestamp(oos_end_s)
    sd_train = {c: df[df["date"] < cutoff].reset_index(drop=True)
                for c, df in FULL.items() if len(df[df["date"] < cutoff]) >= 120}
    if len(sd_train) < 40:
        log.info(f"[{tag}] 训练数据不足，跳过"); continue
    log.info(f"[{tag}] 训练({len(sd_train)}只, <{cutoff.date()}) ...")
    X, y, _, dates = build_all_stocks(sd_train)
    name = f"wf_{cutoff.date()}"
    with contextlib.redirect_stdout(io.StringIO()):
        T.train_model(X, y, dates=dates, _save_name=name)
    model = load_model(os.path.join("models", "saved", f"{name}.pt"))

    sd_bt = {c: df[df["date"] <= oos_end].reset_index(drop=True)
             for c, df in FULL.items() if len(df[df["date"] <= oos_end]) >= 40}
    bench = {"沪深300": bench_full} if bench_full is not None else {}
    base = BacktestEngine(sd_bt, model, benchmarks=bench)
    base._build_price_index()
    with contextlib.redirect_stdout(io.StringIO()):
        sig = base._precompute_signals()

    bm = bench_ret(cutoff, oos_end)
    for cfg_name, cfg in CFGS:
        p = {"start_date": cutoff_s}; p.update(cfg)
        eng = BacktestEngine(sd_bt, model, benchmarks=bench, params=p)
        eng._precompute_signals = lambda _s=sig: _s
        with contextlib.redirect_stdout(io.StringIO()):
            mt = eng.run()
        rows.append((tag, bm, cfg_name, mt))
        log.info(f"  [{tag}] {cfg_name:<6} 收益={str(mt.get('总收益率')):>9} "
                 f"超额={str(mt.get('超额收益')):>9} 回撤={str(mt.get('最大回撤')):>9} "
                 f"夏普={str(mt.get('夏普比率')):>7}")
    for f in glob.glob(os.path.join("models", "saved", f"{name}*")):
        try: os.remove(f)
        except Exception: pass

log.info("\n" + "=" * 86)
log.info("  滚动前向检验汇总：每折「用历史训练、测未来」，对比有无大盘择时闸")
log.info("=" * 86)
log.info(f"  {'区间':<12}{'沪深300':>9}  {'配置':<7}{'策略收益':>10}{'超额':>10}{'最大回撤':>10}{'夏普':>8}")
last = None
for tag, bm, cfg_name, mt in rows:
    head = f"  {tag:<12}{(f'{bm:.1f}%' if bm is not None else 'NA'):>9}" if tag != last else "  " + " " * 19
    log.info(f"{head}  {cfg_name:<7}{str(mt.get('总收益率')):>10}{str(mt.get('超额收益')):>10}"
             f"{str(mt.get('最大回撤')):>10}{str(mt.get('夏普比率')):>8}")
    last = tag
log.info("=" * 86)
log.info("  关键：熊市(2018/2022)加了择时闸后，回撤和亏损是否被大幅削减？")
log.info("=== walk-forward v2 完成 ===")
