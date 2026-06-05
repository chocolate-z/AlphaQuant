# AlphaQuant 项目交接

你将接手一个 A股量化交易系统（AlphaQuant）。请先通读本文件，再开始工作。

## 工作约定（重要）
- 开发分支：`claude/modest-goodall-v5Egd`，所有改动提交并推送到此分支
- 仓库：chocolate-z/AlphaQuant
- 不要主动创建 PR，除非用户明确要求
- 提交信息用中文，描述清楚「改了什么 + 为什么」
- 该环境无 GPU、无 akshare/yfinance、训练在 CPU 上跑

## 项目架构
A股 GRU 选股模型 + 回测 + 模拟盘 + 网页控制台。核心流程：
拉数据 → 构建特征序列 → 训练模型 → 回测 / 生成每日信号。

关键文件：
- `config.py` — 全局配置（股票池、超参、交易参数）
- `data/loader.py` — 个股日线抓取（搜狐为主，多源回退）
- `data/index_fetcher.py` — 指数抓取（搜狐 + 腾讯备用源）
- `features/builder.py` — 16个技术特征 + 3个市场特征(沪深300)，逐窗口Z-Score归一化；标签构建
- `models/lstm_model.py` — 模型（名为LSTMModel，实为2层GRU + Attention + 3层FC + Sigmoid）
- `models/trainer.py` — 训练循环、Focal Loss、早停、AUC监控、集成训练
- `backtest/engine.py` — 回测引擎（支持 params dict 覆盖交易参数）
- `main.py` — 命令行菜单入口，各操作函数
- `dashboard/app.py` — Flask 网页控制台（苹果风UI、SSE实时日志、实时训练曲线、可调回测参数）
- `dashboard/tasks.py` — 后台任务运行器（捕获 stdout/logging 经 SSE 推送前端）
- `paper_trading/executor.py` — 模拟盘信号生成

## 关键设计
- **标签**：超额收益标签（`USE_EXCESS_LABEL=True`）——个股5日收益 − 沪深300同期收益 > `EXCESS_THRESHOLD`(当前0.02) 记为1，剥离大盘beta，专注alpha
- **特征维度** `FEATURE_DIM=19`：16个个股技术特征 + 3个市场特征(mkt_ret_1d/5d, mkt_ma20_dev)
- **集成训练**：N个不同种子模型(`ENSEMBLE_N_MODELS=3`，种子 42+i*17)，推理时概率平均。`models/lstm_model.py` 的 `load_best_available()` 统一加载器（有集成清单则返回列表，否则单模型），`predict_proba()` 接受列表会自动平均
- **网页训练曲线**：前端用正则解析 SSE 日志里的「第N轮 | 损失 | 训练识别率 | 验证识别率」行实时画图，无需改 trainer
- **回测参数化**：`BacktestEngine(params=dict)` 可从网页覆盖买卖阈值/止损止盈等，所有交易常量存为实例属性 `self.*`

## 本轮（claude）完成的优化（已提交并推送，最新 f7d7d15）
共 12 个 commit，每个独立。要点：
1. **修复模型加载崩溃**：torch.compile 包装使权重键带 `_orig_mod.` 前缀，普通 LSTMModel 加载即崩
   （回测/模拟盘/选股全挂）。加 `strip_compile_prefix` 保存/加载都剥前缀。后续直接移除了 torch.compile
   （实测 aot_eager 零提速、还反复重编译）。
2. **真·时序验证切分**：原按股票横截面切却自称"时序"，验证 AUC 虚高。改为按时间分位点切 + 边界 purge
   + 训练集 shuffle。诚实验证 AUC≈0.52~0.53。
3. **缓存池训练**：`load_cached_stocks()` + `PREFER_CACHED_POOL`，优先用本地缓存（离线、数据多、不限流）；
   训练池只取 **sh/sz 主板**（剔除流动性差的北交所），均匀抽样。
4. **数据源大修**（依 leek-fund 文档）：腾讯 fqkline 此前因 maxBars=3000 报 param error 解析崩 →
   **腾讯源一直没生效**。改 maxBars=800 翻页取全历史 + 容错；K线主源切腾讯、板块自适应路由（北交所走搜狐）；
   指数源同切腾讯；实时行情加腾讯兜底。全部联网实测通过。
5. **大盘择时闸 MA20**（最关键风控）：沪深300 跌破均线→清仓避熊。walk-forward 实测四折全降回撤
   （2018熊回撤 -42%→-18%、2022熊 -28%→-9%）。`USE_MARKET_FILTER=True, MARKET_MA_DAYS=20`。
6. **Point-in-time 动态可交易池**：买入只允许「截至昨日近 60 日成交额中位 ≥2 亿」的票，随时间更新、
   只用历史数据。`get_tradeable_pool()` + 引擎 `_is_liquid()`。实现"信号扫全市场、下单只在流动池"。
7. **去未来函数**：回测改为「用昨日信号/择时/流动性，今日成交」（信号滞后1日）。这是分水岭——见下。
8. **横截面特征实现后又关闭**：动量排名 walk-forward 实测在 2018/2022 熊市**净减分**，`USE_CROSS_SECTIONAL=False`
   （代码保留在 features/builder.py，将来可换均值回归/价值排名再试）。
9. **正式模型**：`train_production.py` 在 150 只 sh/sz 全历史训练 + 诚实 OOS。`lstm_best.pt` 即正式模型。
10. **模拟盘对齐回测**：executor 复用引擎的择时/流动性判定，跑和回测一致的策略（信号全池、下单流动池、
    择时闸、最短持有/冷却、无未来函数）。`fetch_all_realtime` 支持任意股票池。

> 🔴 **最重要的诚实结论**：第 7 步去掉未来函数后，真相暴露——之前"好看"的超额大多是未来函数 + 幸存者偏差
> 堆出来的。**正式模型诚实 OOS（2024-09~2026-06，同期沪深300 +51%）：总收益 +54%、但超额仅 +0.78%、
> 回撤 -28%、夏普 0.93。** 即：**这个纯价量模型目前≈"高波动版指数跟随"，没有可用的稳定 alpha。**
> 这不是失败，是诚实——纯价量预测 A 股日线本就极难有超额。**未上线资格**（无超额）。

## 已知问题 / 待办（按优先级，含诚实判断）
- **🧱 免费数据的天花板已到**：价量特征(已榨干，AUC0.53/超额~0)、横截面动量(净减分,已关)、个股资金流
  (东财免费接口**只有~120天历史**，无法训练长史模型——已验证不可行)。**要真 alpha 基本只能上付费数据**
  （Wind/聚宫/米筐：无幸存者偏差历史 + 个股资金流史 + 基本面）。
- **未做的真正检验**：模拟盘**前向空跑数月**——这是唯一不能造假的标准。现模拟盘已对齐回测，可开始跑。
- **幸存者偏差**：本地缓存只有"活到今天"的股票，所有回测都偏乐观，去不掉（除非付费 point-in-time 数据）。
- **换手率(turnover)**：是 16 特征之一，但腾讯源不带（填0），近端增量数据会丢；想保留需对它优先用搜狐补。
- **网页回测未走 OOS**：dashboard 回测仍是全区间（含样本内、偏乐观）；想看诚实结果，起始日填训练截止日之后。
- **torch线程坑**：`set_num_interop_threads` 只能在并行前调用一次 → 已 try/except 容错，勿删。

## 可直接用的脚本
- `python train_production.py` — 训正式模型 + 诚实 OOS（约 30-40 分钟）。
- `python oos_backtest.py` — 用现成模型跑诚实样本外回测（不重训，2-3 分钟）。
- `python walk_forward.py` — 跨 2018/2020/2022/2024 多折、对比有无择时闸的前向检验。

## 当前关键配置值
```
USE_EXCESS_LABEL=True, EXCESS_THRESHOLD=0.02, FOCAL_GAMMA=2.0, LABEL_SMOOTHING=0.05
PREFER_CACHED_POOL=True, MAX_TRAIN_STOCKS=300, WINDOW_SIZE=30, LABEL_HORIZON=5
FEATURE_DIM=19, USE_CROSS_SECTIONAL=False（关；开则23维）
DROPOUT=0.45, WEIGHT_DECAY=1e-3, NOISE_STD=0.05, LEARNING_RATE=3e-4, EARLY_STOP_PATIENCE=20
风控: MAX_HOLDINGS=10, MAX_POSITION_RATIO=0.10, STOP_LOSS=-0.12, TAKE_PROFIT=0.30, TOP_N_BUY=5
     MIN_HOLD_DAYS=10, COOLDOWN_DAYS=5, RELATIVE_RANK_MODE=True
择时/流动性: USE_MARKET_FILTER=True, MARKET_MA_DAYS=20, LIQ_WINDOW_DAYS=60, LIQ_MIN_AMOUNT_YI=2.0
回测无未来函数: 信号/择时/流动性均用上一交易日数据，今日成交（engine.run 内 prev_day 映射）
```

## 用户的长期目标
持续优化「模型 + 策略本身」，让选股模型预测更准、回测收益/回撤更好。用户偏好中文交流、看重可解释性（日志要让小白能看懂）。

请先 `git log --oneline -10` 和通读 config.py、features/builder.py、models/trainer.py 熟悉现状，再动手。
