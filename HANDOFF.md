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

## 本轮（claude）刚完成的优化（已提交并推送，最新 42aca23）
按时间顺序，每条都是独立 commit：
1. **修复模型加载崩溃**：训练用 `torch.compile` 包装后保存的权重键带 `_orig_mod.` 前缀，
   普通 LSTMModel 加载即崩（回测/模拟盘/选股全挂）。加 `strip_compile_prefix` 保存/加载都剥前缀。
2. **真·时序验证切分**（关键）：原来按股票横截面切分却自称"时序"，训练/验证覆盖同一段日历、
   只是股票不同，且全市场同涨同跌会泄漏 → 验证 AUC 虚高(0.59)。改为按时间分位点切（较早训练、
   较晚验证）+ 边界 purge + 训练集 shuffle。诚实验证 AUC≈0.52~0.53（真实"预测未来"能力）。
3. **缓存池训练**：`load_cached_stocks()` + `PREFER_CACHED_POOL`，完整训练优先用本地全部缓存
   （离线、数据多、不被限流），不再每次从全A股随机抽 300 只去联网。
4. **移除 torch.compile**：实测 aot_eager 在本项目零提速，反而触发反复重编译 + 制造前缀 bug。
5. **数据源大修**（依据 leek-fund 接口文档）：发现腾讯 fqkline 因 maxBars=3000 报 param error、
   解析按 dict 取值而崩——**腾讯源一直没生效**，多源回退名存实亡。改为 maxBars=800 翻页取全历史 +
   容错；K线主源切腾讯、按板块自适应路由（北交所走搜狐）；搜狐降级快速失败；指数源同样切腾讯主源；
   实时行情加腾讯兜底。**全部联网实测通过**。
6. **诚实样本外(OOS)回测 + 反刷单 + 风控默认值**：引擎加 `start_date`（信号用完整历史算、只从训练
   截止日后开仓），加"最短持有期+卖出冷却"杜绝来回刷单，修了交易日志 O(n²)。用 OOS 扫描选出新默认：
   仓位 0.20→0.10、持仓 5→10、止损 -0.07→-0.12、止盈 0.20→0.30、TOP_N 3→5、MIN_HOLD_DAYS=10、COOLDOWN_DAYS=5。

> 📌 重要发现：旧的"好看回测"是**样本内**自欺。诚实 OOS 一照——80 只(大半北交所)模型 + 截断特征
> 是 -48% 回撤/-64% 超额；换 120 只 sh/sz 主板 + 完整历史特征 + 新风控后，OOS(2024-09~2026-06)
> 总收益 143%/夏普 1.56/超额 +40%/回撤 -21%。⚠️ 但该区间是强牛市(沪深300 +51%)且缓存池有幸存者
> 偏差，绝对收益偏乐观；稳健结论是"分散+低换手"显著改善夏普与回撤。当前模型信号仍偏弱(AUC≈0.53)。

## 已知问题 / 待办（按优先级）
- **信号偏弱是最大瓶颈**：AUC≈0.53、概率挤在 [0.46,0.63]。最该做的是**横截面特征**（同一天该股相对
  全市场的动量/量能/RSI 排名）——这是新增信息、最可能提升 alpha。需改 build_all_stocks/_precompute_signals
  让特征带上横截面排名（注意训练与推理要用同一套排名口径），会动 FEATURE_DIM、要重训。
- **幸存者偏差 & 单一牛市区间**：OOS 只测了 2024-09 后的牛市。应找更早 cutoff 或加入已退市股票再验。
- **生产模型**：当前 lstm_best.pt 是 120 只 sh/sz 的验证模型，够用；正式可在更多股票上重训/集成。
- **网页回测未走 OOS**：dashboard 回测仍是全区间（含样本内）。可加"仅样本外"开关 + 暴露 min_hold/cooldown。
- **torch线程坑**：`set_num_interop_threads` 只能在并行工作前调用一次 → 已用 try/except 容错，勿删。

## 当前关键配置值
```
USE_EXCESS_LABEL=True, EXCESS_THRESHOLD=0.02, FOCAL_GAMMA=2.0, LABEL_SMOOTHING=0.05
PREFER_CACHED_POOL=True, MAX_TRAIN_STOCKS=500, WINDOW_SIZE=30, LABEL_HORIZON=5, FEATURE_DIM=19
DROPOUT=0.45, WEIGHT_DECAY=1e-3, NOISE_STD=0.05, LEARNING_RATE=3e-4, EARLY_STOP_PATIENCE=20
风控(OOS扫描选优): MAX_HOLDINGS=10, MAX_POSITION_RATIO=0.10, STOP_LOSS=-0.12, TAKE_PROFIT=0.30
              TOP_N_BUY=5, MIN_HOLD_DAYS=10, COOLDOWN_DAYS=5, RELATIVE_RANK_MODE=True
```

## 用户的长期目标
持续优化「模型 + 策略本身」，让选股模型预测更准、回测收益/回撤更好。用户偏好中文交流、看重可解释性（日志要让小白能看懂）。

请先 `git log --oneline -10` 和通读 config.py、features/builder.py、models/trainer.py 熟悉现状，再动手。
