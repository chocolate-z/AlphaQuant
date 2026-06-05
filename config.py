# 全局配置：股票池、路径、模型超参、交易参数

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 路径 ─────────────────────────────────────────────
DATA_CACHE_DIR   = os.path.join(BASE_DIR, "data", "cache")
MODEL_SAVE_DIR   = os.path.join(BASE_DIR, "models", "saved")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")
REPORTS_DIR      = os.path.join(BASE_DIR, "reports")
SIGNAL_CACHE_FILE = os.path.join(LOGS_DIR, "signals_cache.json")
PEAK_ASSETS_FILE  = os.path.join(LOGS_DIR, "peak_assets.json")

for _d in [DATA_CACHE_DIR, MODEL_SAVE_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── 股票池（沪深300精选20只，覆盖金融/消费/科技/医药/新能源）──
STOCK_POOL = [
    "sh600519",  # 贵州茅台  消费
    "sh600036",  # 招商银行  金融
    "sh601318",  # 中国平安  金融
    "sz000858",  # 五粮液    消费
    "sz000333",  # 美的集团  消费/科技
    "sh600900",  # 长江电力  能源
    "sz300750",  # 宁德时代  新能源
    "sh601888",  # 中国中免  消费
    "sz000001",  # 平安银行  金融
    "sh600276",  # 恒瑞医药  医药
    "sh601166",  # 兴业银行  金融
    "sz002594",  # 比亚迪    新能源
    "sh600309",  # 万华化学  材料
    "sz300059",  # 东方财富  科技/金融
    "sh601012",  # 隆基绿能  新能源
    "sz002415",  # 海康威视  科技
    "sh600031",  # 三一重工  工业
    "sz000568",  # 泸州老窖  消费
    "sh601628",  # 中国人寿  金融
    "sh600048",  # 保利发展  地产
]

# ── 股票抽样 ──────────────────────────────────────────
FULL_STOCK_COUNT    = 100   # 完整模式：从全A股随机抽取数量
QUICK_STOCK_COUNT   = 20    # 快速模式：随机抽取数量
QUICK_HISTORY_YEARS = 3     # 快速模式：只拉取近 N 年数据
STOCK_LIST_CACHE    = "stock_list.json"  # 全A股列表缓存文件名（存于 DATA_CACHE_DIR）

# ── 数据 ─────────────────────────────────────────────
START_DATE       = "20150101"
WINDOW_SIZE      = 30             # 时间窗口（天）— 增加上下文
FEATURE_DIM      = 16             # 特征维度（16个技术特征）
LABEL_HORIZON    = 5              # 标签：未来N日
LABEL_THRESHOLD  = 0.05           # 标签阈值（5%）

# ── 模型 ─────────────────────────────────────────────
# 缩小容量 + 加强正则：金融数据信噪比极低，小模型反而泛化更好（防过拟合）
LSTM_HIDDEN1         = 64         # 128→64：减少参数，抑制记忆噪声
LSTM_HIDDEN2         = 32         # 64→32
FC_HIDDEN            = 24
DROPOUT              = 0.45       # 0.3→0.45：更强随机失活
WEIGHT_DECAY         = 1e-3       # 1e-4→1e-3：更强 L2 正则
NOISE_STD            = 0.05       # 训练时给输入加高斯噪声做数据增强（0=关闭）
LEARNING_RATE        = 3e-4       # 5e-4→3e-4：放慢拟合，延迟过拟合
BATCH_SIZE           = 512        # 更大batch：减少迭代次数，加快每轮速度
MAX_EPOCHS           = 300        # 足够多轮次，靠早停控制
EARLY_STOP_PATIENCE  = 20         # 验证早早见顶，缩短耐心避免空跑
LR_PATIENCE          = 7
TRAIN_RATIO          = 0.8

# ── 回测 & 模拟盘 ────────────────────────────────────
INIT_CAPITAL       = 1_000_000   # 初始资金 100万
COMMISSION_BUY     = 0.0003      # 买入手续费
COMMISSION_SELL    = 0.0013      # 卖出手续费（含印花税）
RISK_FREE_RATE     = 0.03        # 无风险利率

BUY_THRESHOLD      = 0.65        # 买入概率阈值（绝对模式）
SELL_THRESHOLD     = 0.35        # 卖出概率阈值（绝对模式）
MAX_POSITION_RATIO = 0.20        # 单股最大仓位
MAX_HOLDINGS       = 5           # 最大持仓数量

# ── 相对排名交易模式 ─────────────────────────────────
# True = 每天按概率排名，买前 TOP_N 只（无论绝对概率高低）
# False = 传统绝对阈值模式（需要模型训练充分才有交易）
RELATIVE_RANK_MODE  = True
TOP_N_BUY           = 3          # 每日最多买入候选数（实际受 MAX_HOLDINGS 限制）
RANK_SELL_BOTTOM    = 0.40       # 相对模式下：概率低于此值触发卖出
STOP_LOSS_RATIO    = -0.07       # 单股止损
TAKE_PROFIT_RATIO  =  0.20       # 单股止盈（涨幅超过此值自动卖出）
PORTFOLIO_STOP     = -0.12       # 组合止损
SUSPEND_DAYS       = 3           # 触发组合止损后暂停天数

# ── CPU 线程 ─────────────────────────────────────────
# 训练时使用的 CPU 线程数（0 = 自动用满所有核，0.5 = 一半核，正整数 = 固定数量）
CPU_THREAD_RATIO    = 0.5

# ── 调度 ────────────────────────────────────────────
DAILY_RUN_TIME     = "15:30"     # 每日触发时间

# ── 看板 ────────────────────────────────────────────
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
