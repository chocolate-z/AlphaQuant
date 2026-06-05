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

## 上一任刚完成的优化（已提交 d091612）
1. **市场数据鲁棒性**：index_fetcher 加腾讯财经备用源；market_features 缓存 3→7天；在线失败时用过期缓存而非填0（之前填0会让超额标签退化成绝对涨跌标签）
2. **Focal Loss(γ=2)** 替换加权BCE，alpha按类别比例自动算
3. **标签平滑** `LABEL_SMOOTHING=0.05`
4. **超额阈值** `EXCESS_THRESHOLD` 0.03→0.02，提高正样本密度

> ⚠️ 这些改动后还没有重新训练验证。第一件事建议：重新跑一次集成训练，确认新损失函数下验证AUC是否改善、正样本比例是否到了~20%。

## 已知问题 / 待办
- **过拟合**：历史上训练AUC升到~0.70但验证AUC约epoch 2见顶后下滑。可继续尝试：更强正则、减小模型容量、特征精简、更多样本
- **数据抓取限流**：搜狐频繁503，已有4次指数退避+多源回退，但拖慢训练
- **特征工程**：可考虑加 板块轮动、波动率regime 等市场相对特征
- **回测策略调参**：`RELATIVE_RANK_MODE=True`、`TOP_N_BUY=3`、`RANK_SELL_BOTTOM=0.40`、`STOP_LOSS_RATIO=-0.07`、`TAKE_PROFIT_RATIO=0.20` 都还可优化
- **torch线程坑**：`set_num_interop_threads` 只能在并行工作前调用一次，网页常驻进程二次训练会崩 → 已用 try/except 容错，勿删

## 当前关键配置值
```
USE_EXCESS_LABEL=True, EXCESS_THRESHOLD=0.02, FOCAL_GAMMA=2.0, LABEL_SMOOTHING=0.05
FULL_STOCK_COUNT=300, WINDOW_SIZE=30, LABEL_HORIZON=5, FEATURE_DIM=19
DROPOUT=0.45, WEIGHT_DECAY=1e-3, NOISE_STD=0.05, LEARNING_RATE=3e-4
EARLY_STOP_PATIENCE=20, ENSEMBLE_N_MODELS=3, BUY_THRESHOLD=0.65, SELL_THRESHOLD=0.35
```

## 用户的长期目标
持续优化「模型 + 策略本身」，让选股模型预测更准、回测收益/回撤更好。用户偏好中文交流、看重可解释性（日志要让小白能看懂）。

请先 `git log --oneline -10` 和通读 config.py、features/builder.py、models/trainer.py 熟悉现状，再动手。
