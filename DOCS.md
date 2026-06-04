# AlphaQuant 使用文档

> A股量化AI系统 · 完整使用手册  
> 版本：v2.0 · 更新日期：2026-06

---

## 目录

1. [项目简介](#1-项目简介)
2. [系统要求与环境搭建](#2-系统要求与环境搭建)
3. [项目结构说明](#3-项目结构说明)
4. [全局配置参数](#4-全局配置参数)
5. [数据层](#5-数据层)
6. [特征构建](#6-特征构建)
7. [模型训练](#7-模型训练)
8. [历史回测](#8-历史回测)
9. [模拟盘（实盘演练）](#9-模拟盘实盘演练)
10. [本地看板](#10-本地看板)
11. [单股诊断](#11-单股诊断)
12. [完整工作流](#12-完整工作流)
13. [命令行参数速查](#13-命令行参数速查)
14. [常见问题与排错](#14-常见问题与排错)
15. [性能与资源说明](#15-性能与资源说明)
16. [研究扩展方向](#16-研究扩展方向)

---

## 1. 项目简介

AlphaQuant 是一套面向 A 股市场的端到端量化 AI 系统，包含以下四大模块：

| 模块 | 功能 |
|------|------|
| **数据层** | 通过 AKShare 获取日K线数据，本地 CSV 缓存 + 每日增量更新 |
| **模型层** | 两层 LSTM 神经网络，基于纯原始数据预测买入概率 |
| **回测引擎** | 严格 T+1、手续费、涨跌停、止损约束的历史回测 |
| **模拟盘** | 每日 15:30 自动执行的虚拟账户交易系统 |

**核心设计原则：**
- 特征全部来自原始行情数据，不使用 RSI、MACD 等人工指标
- 让 LSTM 模型自行发现价格序列中的规律
- 严格禁止前视偏差（Look-ahead Bias），scaler 仅在训练集上 fit
- 回测信号预计算（O(S×N) 复杂度），单次回测约 2 分钟

---

## 2. 系统要求与环境搭建

### 2.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 2 GB 可用空间 | SSD 10 GB+ |
| GPU | 不需要 | NVIDIA GPU（训练加速） |

> 无 GPU 也可正常运行，训练约 10-30 分钟，CPU 推理约 2 秒/股。

### 2.2 创建 conda 环境

```bash
# 1. 创建独立环境（Python 3.10）
conda create -n alphaquant python=3.10 -y

# 2. 激活环境
conda activate alphaquant

# 3. 安装 PyTorch（CPU 版，适用 Windows/Linux）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 如有 NVIDIA GPU，改用：
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; import akshare; print('环境OK, PyTorch:', torch.__version__)"
```

### 2.3 依赖说明

| 包名 | 用途 | 版本要求 |
|------|------|---------|
| `akshare` | A 股行情数据源 | ≥1.10.0 |
| `torch` | LSTM 模型训练与推理 | ≥2.0.0 |
| `pandas` | 数据处理 | ≥2.0.0 |
| `scikit-learn` | MinMaxScaler 特征归一化 | ≥1.3.0 |
| `flask` | 本地看板 Web 服务 | ≥3.0.0 |
| `schedule` | 模拟盘定时调度 | ≥1.2.0 |
| `scipy` | 线性回归趋势分析 | ≥1.11.0 |
| `wcwidth` | 终端中文字符精确对齐 | ≥0.2.0 |
| `matplotlib` | 回测净值曲线图 | ≥3.7.0 |

### 2.4 项目克隆与初始化

```bash
git clone <repository_url>
cd AlphaQuant

# 目录结构会在首次运行时自动创建：
# data/cache/、models/saved/、logs/、reports/
```

---

## 3. 项目结构说明

```
AlphaQuant/
├── config.py                  # 全局配置（股票池、路径、模型超参、交易参数）
├── main.py                    # 命令行主入口
├── requirements.txt           # Python 依赖列表
│
├── data/
│   ├── loader.py              # AKShare 历史数据获取 + 本地缓存
│   ├── realtime.py            # 每日收盘后实时行情拉取
│   └── cache/                 # 本地 CSV 缓存（自动创建）
│
├── features/
│   └── builder.py             # 原始特征构建（8个特征，无人工指标）
│
├── models/
│   ├── lstm_model.py          # LSTM 模型定义（两层 LSTM + FC）
│   ├── trainer.py             # 训练器（时序分割、AUC 早停、版本管理）
│   └── saved/                 # 模型权重（lstm_best.pt + 历史版本）
│
├── backtest/
│   ├── engine.py              # 历史回测引擎（预计算信号、涨跌停处理）
│   └── metrics.py             # 绩效指标（夏普、Calmar、Sortino、胜率）
│
├── paper_trading/
│   ├── account.py             # 虚拟账户（资金、持仓、T+1、JSON 持久化）
│   ├── executor.py            # 交易执行（批量推理、信号缓存）
│   ├── scheduler.py           # 每日 15:30 自动调度
│   ├── risk.py                # 风控（单股 -7% / 组合峰值回撤 -12%）
│   └── logger.py              # 交易日志（CSV + 防重复快照）
│
├── dashboard/
│   └── app.py                 # Flask 本地看板（5个页面）
│
├── diagnose/
│   ├── analyzer.py            # 单股诊断（趋势+量能+AI概率+风控价位）
│   └── report.py              # 终端格式化报告输出
│
├── logs/                      # 运行日志（自动创建）
│   ├── account_state.json     # 账户状态持久化
│   ├── account_daily.csv      # 每日账户快照
│   ├── trades_YYYYMMDD.csv    # 每日交易记录
│   ├── signals_cache.json     # 今日信号缓存
│   └── peak_assets.json       # 历史最高净值记录
│
└── reports/                   # 回测报告（自动创建）
    └── backtest_nav.png       # 净值曲线图
```

---

## 4. 全局配置参数

所有参数集中在 `config.py`，按需修改后重新运行即可生效。

### 4.1 股票池配置

```python
# config.py
STOCK_POOL = [
    "sh600519",  # 贵州茅台  消费
    "sh600036",  # 招商银行  金融
    # ... 共 20 只，覆盖金融/消费/科技/医药/新能源
]
```

**股票代码格式：** `sh` 前缀 = 上交所，`sz` 前缀 = 深交所  
例：`sh600519`（贵州茅台）、`sz000858`（五粮液）、`sz300750`（宁德时代）

**修改股票池后需要：**
1. 重新运行 `--mode train --refresh` 下载新股票数据并重训模型
2. 重新运行 `--mode backtest` 更新回测结果

### 4.2 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `START_DATE` | `"20190101"` | 历史数据起始日期 |
| `WINDOW_SIZE` | `20` | LSTM 输入时间窗口（天） |
| `FEATURE_DIM` | `8` | 特征维度数量 |
| `LABEL_HORIZON` | `5` | 标签：未来 N 日 |
| `LABEL_THRESHOLD` | `0.05` | 正样本阈值（未来5日内涨>5%） |

### 4.3 模型超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LSTM_HIDDEN1` | `128` | 第一层 LSTM 隐藏层大小 |
| `LSTM_HIDDEN2` | `64` | 第二层 LSTM 隐藏层大小 |
| `DROPOUT` | `0.3` | Dropout 比例 |
| `LEARNING_RATE` | `1e-3` | Adam 初始学习率 |
| `BATCH_SIZE` | `64` | 训练批次大小 |
| `MAX_EPOCHS` | `100` | 最大训练轮数 |
| `EARLY_STOP_PATIENCE` | `15` | 早停：验证 AUC 连续15轮无提升则停止 |
| `TRAIN_RATIO` | `0.8` | 训练/验证集比例（时序分割） |

### 4.4 交易参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INIT_CAPITAL` | `1_000_000` | 初始资金（100万） |
| `COMMISSION_BUY` | `0.0003` | 买入手续费（0.03%） |
| `COMMISSION_SELL` | `0.0013` | 卖出手续费（0.13%，含印花税） |
| `BUY_THRESHOLD` | `0.65` | 买入概率阈值（>65%触发买入） |
| `SELL_THRESHOLD` | `0.35` | 卖出概率阈值（<35%触发卖出） |
| `MAX_POSITION_RATIO` | `0.20` | 单股最大仓位（总资产20%） |
| `MAX_HOLDINGS` | `5` | 最大同时持仓只数 |
| `STOP_LOSS_RATIO` | `-0.07` | 单股止损线（-7%） |
| `PORTFOLIO_STOP` | `-0.12` | 组合止损线（从峰值回撤-12%） |
| `SUSPEND_DAYS` | `3` | 触发组合止损后暂停交易天数 |

---

## 5. 数据层

### 5.1 历史数据获取

**文件：** `data/loader.py`

系统使用 AKShare 获取前复权日 K 线数据，字段包括：

| 字段 | 说明 |
|------|------|
| `date` | 交易日期 |
| `open` | 开盘价（前复权） |
| `close` | 收盘价（前复权） |
| `high` | 最高价（前复权） |
| `low` | 最低价（前复权） |
| `volume` | 成交量（手） |
| `amount` | 成交额（元） |
| `pct_change` | 涨跌幅（%） |
| `turnover` | 换手率（%） |
| `volume_ratio` | 量比（默认1.0） |
| `main_net_inflow` | 主力净流入（元，尽力而为） |

**缓存机制：**
- 首次运行：全量下载 → 保存到 `data/cache/{code}.csv`
- 后续运行：检查缓存日期 → 只增量下载新数据
- 强制刷新：`python main.py --mode train --refresh`

### 5.2 实时行情更新

**文件：** `data/realtime.py`

每日 15:30 触发（由模拟盘调度器调用）：
1. 判断今天是否为交易日（调用交易日历 API）
2. 从东方财富实时接口拉取当日收盘行情
3. 将今日数据追加到各股票的历史缓存 CSV

**手动触发缓存更新：**
```python
from data.realtime import update_all_caches
update_all_caches()
```

---

## 6. 特征构建

### 6.1 8个原始特征

**文件：** `features/builder.py`

系统不使用任何人工设计的技术指标，全部特征从原始 OHLCV 数据直接计算：

| 特征名 | 计算公式 | 含义 |
|--------|---------|------|
| `pct_change` | 直接来自行情 | 当日涨跌幅（日收益率）|
| `turnover` | 直接来自行情 | 换手率 |
| `volume_ratio` | 直接来自行情 | 量比 |
| `volume_norm` | `volume / 20日均量` | 成交量相对均量比值 |
| `main_inflow_ratio` | `主力净流入 / 总成交额` | 主力参与度 |
| `amplitude` | `(high - low) / 昨收` | 当日振幅 |
| `open_change` | `open / 昨收 - 1` | 开盘跳空幅度 |
| `close_strength` | `(close - low) / (high - low)` | 收盘强弱（0=收于最低，1=收于最高）|

### 6.2 时间窗口与标签

**输入形状：** `(batch_size, 20, 8)` — 过去20个交易日的8维特征

**标签定义：**
```
正样本（买入信号 = 1）：未来5个交易日内最高价相对今日收盘涨幅 > 5%
负样本（不买入 = 0）：否则
```

### 6.3 归一化方式

- 使用 `MinMaxScaler` 将每个特征归一化到 `[0, 1]`
- **关键：Scaler 仅在训练集（前80%数据）上 fit**，验证集用相同 scaler transform
- Scaler 参数保存到 `models/saved/scaler.joblib`，推理时自动加载

### 6.4 正负样本比例

通常正样本（大涨）约占 15-25%，系统通过**加权 BCE Loss** 自动处理不平衡问题：
```python
pos_weight = neg_count / pos_count  # 自动计算权重
```

---

## 7. 模型训练

### 7.1 LSTM 模型架构

**文件：** `models/lstm_model.py`

```
输入 (batch, 20, 8)
    ↓
LSTM层1: hidden=128, Dropout=0.3
    ↓
LSTM层2: hidden=64,  Dropout=0.3
    ↓  取最后一个时间步
全连接: 64 → 32 → 1
    ↓
Sigmoid → 买入概率 [0, 1]
```

### 7.2 训练流程

```bash
# 首次训练（下载数据 + 训练模型）
python main.py --mode train

# 强制重新下载所有数据后训练
python main.py --mode train --refresh
```

**训练输出示例：**
```
Epoch   1 | Loss: 0.6823 | Train AUC: 0.5234 | Val AUC: 0.5189
Epoch   2 | Loss: 0.6541 | Train AUC: 0.5678 | Val AUC: 0.5421
...
Epoch  43 | Loss: 0.4821 | Train AUC: 0.7234 | Val AUC: 0.6987
早停：15 轮无提升，最佳 Val AUC: 0.6987
模型已保存: models/saved/lstm_best.pt
版本副本已保存: models/saved/lstm_20260604_153022_auc0.6987.pt
```

**关键训练策略：**

| 策略 | 实现 |
|------|------|
| 时序分割 | 前80%训练，后20%验证，禁止随机打乱（防止未来数据泄露） |
| 早停 | Val AUC 连续15轮不提升时停止 |
| 学习率衰减 | `ReduceLROnPlateau`，patience=5，factor=0.5 |
| 梯度裁剪 | `clip_grad_norm_(max_norm=1.0)` 防止梯度爆炸 |
| 模型版本 | 每次训练保存带时间戳的副本（`lstm_YYYYMMDD_HHMMSS_auc0.xxxx.pt`）|

### 7.3 理解 AUC 指标

- **AUC < 0.55**：模型基本无效，考虑增加数据量或调整超参数
- **AUC 0.55~0.65**：有一定预测能力，可以进行回测验证
- **AUC 0.65~0.75**：良好，回测通常有正收益
- **AUC > 0.75**：优秀（但需警惕数据泄露）

### 7.4 模型版本管理

```
models/saved/
├── lstm_best.pt                          # 当前最佳模型（最新训练结果）
├── scaler.joblib                         # 特征归一化参数
├── lstm_20260601_143022_auc0.6823.pt    # 历史版本（可手动回滚）
└── lstm_20260604_153022_auc0.6987.pt    # 历史版本
```

**手动回滚到历史版本：**
```bash
cp models/saved/lstm_20260601_143022_auc0.6823.pt models/saved/lstm_best.pt
```

---

## 8. 历史回测

### 8.1 运行回测

```bash
python main.py --mode backtest
```

**前置条件：** 必须先运行 `--mode train` 生成模型文件。

### 8.2 回测规则

| 规则 | 说明 |
|------|------|
| 初始资金 | 100万（`INIT_CAPITAL`） |
| T+1 限制 | 今日买入的股票，明日才可卖出 |
| 买入手续费 | 0.03%（`COMMISSION_BUY`） |
| 卖出手续费 | 0.13%（含印花税，`COMMISSION_SELL`） |
| 涨停处理 | 涨停当日无法买入（已处理） |
| 跌停处理 | 跌停当日无法卖出（已处理） |
| 单股止损 | 浮亏超 -7% 强制卖出 |
| 仓位控制 | 单股不超过总资产20%，整手（100股倍数）买入 |
| 最大持仓 | 同时持仓不超过5只 |

### 8.3 绩效指标

回测完成后输出以下指标：

| 指标 | 说明 |
|------|------|
| **总收益率** | `(最终净值 / 初始资金) - 1` |
| **年化收益率** | 复利年化 |
| **夏普比率** | `√252 × 超额日收益均值 / 标准差`（无风险利率3%）|
| **Sortino 比率** | 只惩罚下行波动的夏普变体 |
| **Calmar 比率** | `年化收益率 / 最大回撤绝对值` |
| **最大回撤** | 从历史最高净值的最大跌幅 |
| **胜率** | 盈利交易次数 / 总交易次数（FIFO配对）|
| **超额收益** | 相对沪深300的年化超额（需提供基准数据）|

**输出示例：**
```
====================================================
  AlphaQuant 回测绩效报告
====================================================
  总收益率          42.38%
  年化收益率        9.27%
  夏普比率          1.234
  Sortino比率       1.687
  Calmar比率        0.923
  最大回撤          -10.04%
  胜率              58.3%
  交易次数          120
  超额收益          3.45%
  最终资产          ¥1,423,800
====================================================
```

### 8.4 回测报告

回测完成后自动保存净值曲线图至 `reports/backtest_nav.png`，包含：
- AlphaQuant 策略净值曲线（蓝色）
- 沪深300基准曲线（橙色虚线，如有数据）
- 基准线1.0

---

## 9. 模拟盘（实盘演练）

### 9.1 启动模拟盘

```bash
# 启动后阻塞运行，每日 15:30 自动执行
python main.py --mode paper
```

**输出：**
```
[AlphaQuant] 模拟盘调度器运行中，每日 15:30 自动执行（Ctrl+C 停止）
```

**Windows 后台运行（推荐）：**
```batch
:: 创建 start_paper.bat
@echo off
conda activate alphaquant
pythonw -c "from paper_trading.scheduler import start_scheduler; start_scheduler()"
```

**Linux 后台运行：**
```bash
nohup python main.py --mode paper > logs/paper.log 2>&1 &
echo $! > logs/paper.pid
```

### 9.2 每日执行流程

每个交易日 15:30 自动按以下顺序执行：

```
1. 判断今天是否为交易日 → 非交易日直接跳过
2. 加载模型和 Scaler
3. 更新行情缓存（追加今日数据到 CSV）
4. 获取当日实时收盘行情
5. 更新持仓市值
6. 风控检查：
   ├── 组合止损检查（峰值回撤 > -12%?）→ 触发则全部清仓 + 暂停3天
   └── 是否在暂停期内 → 是则跳过今日交易
7. 批量推理所有股票买入概率（一次性加载所有数据）
8. 保存信号缓存（供看板使用）
9. 单股止损检查（浮亏 > -7% 且非今日买入）
10. 执行卖出信号（概率 < 0.35）
11. 执行买入信号（概率 > 0.65，按概率从高到低）
12. 更新账户市值
13. 写入当日交易日志 + 账户快照
```

### 9.3 虚拟账户管理

**账户状态持久化文件：** `logs/account_state.json`

```json
{
  "cash": 850000.00,
  "holdings": {
    "sh600519": {
      "shares": 100,
      "cost": 1650.00,
      "buy_date": "2026-06-01",
      "market_value": 168800.00,
      "current_price": 1688.00
    }
  },
  "today_bought": [],
  "updated_at": "2026-06-04T15:32:18"
}
```

**T+1 规则实现：**
- 每日开始时清空 `today_bought` 集合
- 买入时将股票代码加入 `today_bought`
- 卖出时检查是否在 `today_bought` 中，在则拒绝（T+1限制）

### 9.4 风控规则详解

**单股止损（-7%）：**
```
触发条件：(当前价 - 持仓成本) / 持仓成本 < -7%
执行动作：强制市价卖出全部持仓
日志原因：stop_loss
```

**组合止损（峰值回撤 -12%）：**
```
触发条件：(当前总资产 - 历史最高总资产) / 历史最高总资产 < -12%
执行动作：全部清仓 + 暂停交易 3 个交易日
暂停状态：写入 logs/suspend_state.json
注意：使用历史最高净值（非初始资金）计算回撤
```

**仓位控制：**
```
单股最大仓位 = 总资产 × 20%（可调 MAX_POSITION_RATIO）
最大同时持仓 = 5 只（可调 MAX_HOLDINGS）
买入数量取整至100的整数倍（整手交易）
```

### 9.5 手动触发当日执行

无需等待 15:30，手动执行当日流程：

```python
from paper_trading.executor import run_daily_execution
run_daily_execution()
```

### 9.6 重置账户

```bash
python main.py --mode reset
```

执行后会提示确认，输入 `yes` 后：
- 现金恢复为初始资金（100万）
- 清空所有持仓
- 清除暂停状态和峰值记录

### 9.7 交易日志

**每日交易记录：** `logs/trades_YYYYMMDD.csv`

```csv
time,code,action,price,shares,commission,reason
15:32:05,sh600519,buy,1688.00,100,0.51,signal_0.78
15:32:07,sz000858,sell,205.30,500,1.33,signal_0.28
```

**每日账户快照：** `logs/account_daily.csv`

```csv
date,total_assets,cash,holding_value,daily_pnl,cum_pnl
2026-06-01,1000000.00,1000000.00,0.00,0.00,0.00
2026-06-02,1023800.00,835148.00,188652.00,23800.00,23800.00
2026-06-03,1031200.00,835148.00,196052.00,7400.00,31200.00
```

---

## 10. 本地看板

### 10.1 启动看板

```bash
python main.py --mode dashboard
```

浏览器访问：**http://127.0.0.1:5000**

### 10.2 五个页面说明

#### 页面一：账户总览（`/`）
- 虚拟总资产、现金余额、持仓市值（卡片展示）
- 今日盈亏、累计盈亏（金额+百分比，绿色/红色区分）
- 资产净值曲线（近90日，Chart.js 交互图表）
- 每5分钟自动刷新

#### 页面二：当前持仓（`/holdings`）
- 表格：代码、持仓数量、成本价、当前价、浮动盈亏、买入日期
- 浮盈绿色高亮，浮亏红色高亮
- 每2分钟自动刷新

#### 页面三：今日信号（`/signals`）
- 读取 `logs/signals_cache.json`（由模拟盘每日更新，非实时计算）
- 显示所有股票买入概率排行，概率进度条可视化
- 显示缓存更新时间，提示是否为当日数据
- 三种建议徽章：★买入 / ▼卖出 / —观望

#### 页面四：历史交易（`/trades`）
- 支持按**股票代码**和**日期**筛选（GET 参数过滤）
- 每笔交易：时间、代码、操作、价格、数量、手续费、原因
- 买入绿色徽章、卖出红色徽章
- 显示最近 500 条，按时间倒序

**筛选示例：**
```
http://127.0.0.1:5000/trades?code=sh600519
http://127.0.0.1:5000/trades?date=2026-06-01
http://127.0.0.1:5000/trades?code=sh600519&date=2026-06-01
```

#### 页面五：模型表现（`/performance`）
- 信号胜率（FIFO配对，正确处理同一股票多次交易）
- 已完成交易次数、盈利/亏损次数
- 累计实现盈亏（含手续费成本）
- 最大连胜/连败次数
- 每日盈亏状态柱状图（近30日）

### 10.3 同时运行看板和模拟盘

建议开两个终端窗口分别运行：

```bash
# 终端1：启动模拟盘
conda activate alphaquant
python main.py --mode paper

# 终端2：启动看板
conda activate alphaquant
python main.py --mode dashboard
```

---

## 11. 单股诊断

### 11.1 基本用法

```bash
# 诊断单只股票
python main.py --mode diagnose --stock sh600519

# 批量诊断（逗号分隔）
python main.py --mode diagnose --stock sh600519,sz000858,sz300750

# 带持仓成本（判断止盈止损）
python main.py --mode diagnose --stock sh600519 --cost 1650.00
```

### 11.2 报告示例

```
╔══════════════════════════════════════════╗
║  AlphaQuant 单股诊断报告                 ║
╠══════════════════════════════════════════╣
║  股票：贵州茅台 (600519)                 ║
║  当前价：1688.00  涨跌：+2.34%           ║
╠══════════════════════════════════════════╣
║  AI买入概率：78%  ████████░░  高         ║
║  趋势判断：上升趋势（斜率 2.341，R²=0.82）║
║  量能状态：放量上涨（量比 1.82x）        ║
║  主力资金：净流入 +1.23亿               ║
╠══════════════════════════════════════════╣
║  操作建议：★ 建议买入                    ║
║  建议仓位：不超过总资产 19%              ║
║  止损价格：1,569.84（-7%）               ║
║  止盈目标：1,772.40（+5%）               ║
╚══════════════════════════════════════════╝
```

### 11.3 诊断维度说明

**趋势判断（近20日线性回归）：**

| 判断结果 | 条件 |
|---------|------|
| 上升趋势 | 斜率 > 0 且 R² > 0.6 |
| 下降趋势 | 斜率 < 0 且 R² > 0.6 |
| 震荡趋势 | R² ≤ 0.6（趋势不明显）|

**量能判断：**

| 状态 | 条件 |
|------|------|
| 放量上涨 | 今日量 / 20日均量 > 1.5 且当日上涨 |
| 放量下跌 | 今日量 / 20日均量 > 1.5 且当日下跌 |
| 缩量 | 今日量 / 20日均量 < 0.7 |
| 量能正常 | 其他 |

**操作建议逻辑：**

| AI概率 | 趋势 | 建议 |
|--------|------|------|
| > 0.65 | 上升 | ★ 建议买入 |
| > 0.65 | 震荡/下降 | ◎ 观察买入 |
| 0.35~0.65 | 任意 | — 持仓观望 |
| < 0.35 | 任意 | ▼ 建议卖出/观望 |
| 任意 | 下降 | ▼ 建议卖出/观望 |

**持仓成本传入时的额外判断：**

| 浮动盈亏 | 建议 |
|---------|------|
| > +5% | 止盈卖出 |
| < -7% | 止损卖出 |
| 其他 | 按上表逻辑判断 |

**建议仓位计算：**
```python
position_pct = min(int(prob * 25), 20)  # 最高20%
# 概率0.65 → 建议16%仓位
# 概率0.80 → 建议20%仓位
```

---

## 12. 完整工作流

### 12.1 首次使用完整步骤

```bash
# 步骤1：激活环境
conda activate alphaquant
cd AlphaQuant

# 步骤2：下载数据并训练模型（约1-2小时，含数据下载）
python main.py --mode train

# 步骤3：查看训练完成后的回测效果
python main.py --mode backtest

# 步骤4：查看当前所有股票的今日信号
python main.py --mode signal

# 步骤5：诊断感兴趣的股票
python main.py --mode diagnose --stock sh600519

# 步骤6：启动模拟盘（开两个终端）
# 终端A：
python main.py --mode paper

# 终端B：
python main.py --mode dashboard
# 打开浏览器：http://127.0.0.1:5000
```

### 12.2 日常运维流程

```bash
# 每周一次：检查看板，查看模型表现和账户状态
# 浏览器访问 http://127.0.0.1:5000

# 每月一次：增量更新数据并重新训练
python main.py --mode train

# 重新回测验证策略是否有效
python main.py --mode backtest

# 有需要时诊断具体股票
python main.py --mode diagnose --stock sh600519 --cost 1650.00
```

### 12.3 模拟盘长期运行建议

**Windows 开机自启（任务计划程序）：**
1. 打开"任务计划程序" → 创建任务
2. 触发器：登录时 / 系统启动时
3. 操作：`conda run -n alphaquant python C:\path\to\AlphaQuant\main.py --mode paper`

**日志监控：**
```bash
# 查看今日交易日志
type logs\trades_20260604.csv

# 查看账户历史
type logs\account_daily.csv
```

---

## 13. 命令行参数速查

```
python main.py --mode <模式> [选项]
```

| 模式 | 说明 | 常用选项 |
|------|------|---------|
| `train` | 下载数据并训练模型 | `--refresh`（强制重新下载）|
| `backtest` | 运行历史回测 | — |
| `paper` | 启动模拟盘调度器（阻塞）| — |
| `dashboard` | 启动网页看板 | — |
| `signal` | 打印今日信号排行 | — |
| `diagnose` | 单股诊断 | `--stock`（代码）`--cost`（成本）|
| `reset` | 重置虚拟账户 | — |

**完整示例：**
```bash
python main.py --mode train
python main.py --mode train --refresh
python main.py --mode backtest
python main.py --mode paper
python main.py --mode dashboard
python main.py --mode signal
python main.py --mode diagnose --stock sh600519
python main.py --mode diagnose --stock sh600519,sz000858,sz300750
python main.py --mode diagnose --stock sh600519 --cost 1650.00
python main.py --mode reset
```

---

## 14. 常见问题与排错

### Q1：`AKShare 拉取失败` 或数据为空

**原因：** AKShare 接口版本变更、网络限制或请求频率过高。

**解决：**
```bash
# 升级 akshare 到最新版本
pip install akshare --upgrade

# 检查接口是否可用
python -c "import akshare as ak; print(ak.stock_zh_a_hist(symbol='600519', period='daily', start_date='20260101', end_date='20260104', adjust='qfq'))"
```

如果特定接口失效，可在 `data/loader.py` 的 `_fetch_from_akshare` 函数中换用其他 AKShare 接口（如 `stock_zh_a_hist_163`）。

---

### Q2：`模型未找到，请先训练`

**原因：** 未运行训练步骤，`models/saved/lstm_best.pt` 不存在。

**解决：**
```bash
python main.py --mode train
```

---

### Q3：训练时 `特征构建失败，数据可能不足`

**原因：** 某些股票历史数据行数不足 `WINDOW_SIZE + LABEL_HORIZON + 10 = 35` 行。

**解决：** 检查 `data/cache/` 目录下对应股票的 CSV 文件行数：
```bash
# 查看缓存文件行数（Linux/Mac）
wc -l data/cache/*.csv

# 若特定股票数据为空，手动重新拉取
python -c "
from data.loader import load_stock_data
df = load_stock_data('sh600519', force_refresh=True)
print(len(df))
"
```

---

### Q4：回测运行非常慢（超过30分钟）

**原因：** 可能是使用了旧版本的 `engine.py`（未包含信号预计算优化）。

**验证：** 运行时应看到以下输出：
```
正在预计算所有信号（可能需要1-2分钟）...
信号预计算完成：XXXXX 条（20 只股票）
```

如果没有看到"预计算"字样，说明是旧版本，请重新拉取最新代码。

---

### Q5：看板 `/signals` 页面显示"暂无信号缓存"

**原因：** 信号缓存文件 `logs/signals_cache.json` 不存在，尚未运行过模拟盘或手动触发信号。

**解决：**
```bash
# 手动生成信号缓存
python main.py --mode signal

# 或者手动触发
python -c "
from paper_trading.executor import get_current_signals
sigs = get_current_signals()
print(f'已计算 {len(sigs)} 只股票信号')
"
```

---

### Q6：看板图表不显示（空白区域）

**原因：** Chart.js 从 CDN 加载失败（网络限制）。

**解决：** 下载 Chart.js 到本地：
```bash
# 下载 Chart.js 到静态目录（需联网执行一次）
mkdir -p dashboard/static
curl -o dashboard/static/chart.min.js https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js
```

然后修改 `dashboard/app.py` 中的 `_CHARTJS` 变量：
```python
_CHARTJS = '<script src="/static/chart.min.js"></script>'
```

并在 `start_dashboard()` 调用前添加静态文件路由（Flask 会自动处理 `static/` 目录）。

---

### Q7：Windows 中文路径或编码错误

**解决：**
1. 确保项目路径不含中文或特殊字符（如 `C:\AlphaQuant` 而非 `C:\我的项目\AlphaQuant`）
2. 在 `main.py` 顶部添加：
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```
3. 使用 `chcp 65001` 切换 Windows 终端到 UTF-8 编码

---

### Q8：`schedule` 不触发或时间不对

**原因：** 系统时间与北京时间不一致，或进程被挂起。

**验证：**
```python
import datetime
print(datetime.datetime.now())  # 确认是北京时间（UTC+8）
```

**Windows 设置时区：** 控制面板 → 日期和时间 → 更改时区为 `(UTC+08:00) 北京/重庆/香港/乌鲁木齐`

---

### Q9：模型预测概率全部集中在 0.45-0.55 附近

**原因：** 模型未收敛，训练数据量不足或正负样本极度不平衡。

**解决：**
1. 增加训练数据：修改 `config.py` 中 `START_DATE = "20160101"` 扩大历史范围
2. 增加股票池数量（更多样本）
3. 调整超参：降低 `DROPOUT` 至 0.2，增加 `MAX_EPOCHS` 至 200

---

### Q10：止损单无法执行（日志中有"跌停，无法卖出"）

**说明：** 这是正常行为。A 股跌停日不允许卖出，系统会在下一个交易日继续检查止损条件并执行卖出。这是对真实市场规则的正确模拟。

---

## 15. 性能与资源说明

### 时间估算

| 操作 | 耗时估算 | 说明 |
|------|---------|------|
| 首次数据下载（20只股票）| 5-15 分钟 | 取决于网络和 AKShare 速度 |
| 模型训练 | 10-30 分钟 | CPU，约 30-80 轮早停 |
| 历史回测 | 1-3 分钟 | 信号预计算后约30秒/只股票 |
| 每日模型推理（20只）| 3-10 秒 | CPU 推理 |
| 单股诊断 | 5-15 秒 | 含数据加载和推理 |

### 磁盘占用

| 内容 | 大小估算 |
|------|---------|
| 每只股票 CSV 缓存（7年日线）| ~200 KB |
| 20只股票缓存合计 | ~4 MB |
| 模型文件（lstm_best.pt）| ~1.5 MB |
| 每个历史版本模型 | ~1.5 MB |

### 内存占用

| 阶段 | 内存估算 |
|------|---------|
| 训练中（加载全量数据）| ~500 MB |
| 推理/回测 | ~200 MB |
| 看板运行 | ~100 MB |

---

## 16. 研究扩展方向

以下是五个可以进一步深入研究的方向，难度由低到高：

### 16.1 因子有效性分析（入门）

在 `features/` 下新增 `factor_analysis.py`，使用 IC/ICIR 评估每个特征的预测价值：

```python
# IC = 特征与未来收益的截面相关系数
# ICIR = IC均值 / IC标准差（稳定性指标）
# IC > 0.05 被认为有一定预测价值
```

这可以帮助识别哪些特征真正有用，为特征工程提供数据依据。

### 16.2 多时间尺度融合（中级）

同时使用日线（20天）、周线（12周）、月线（6月）构建三路 LSTM 分支，最终拼接输出。不同时间尺度捕捉不同周期的市场规律。

### 16.3 Transformer 替代 LSTM（中高级）

用 Temporal Fusion Transformer 或 PatchTST 替代当前的双层 LSTM。Transformer 的自注意力机制可以直接学习序列中任意两个时间点之间的关系，对长序列效果更好。

### 16.4 市场状态识别（高级）

增加一个市场状态分类器（HMM 或基于沪深300特征的 MLP），自动识别当前处于牛市/震荡/熊市状态，根据不同市场状态动态调整买卖阈值：

```python
# 牛市：降低买入门槛（0.60），提高止盈（+8%）
# 熊市：提高买入门槛（0.75），降低止损（-5%）
```

### 16.5 图神经网络建模股票关联（研究级）

用 GNN 建模行业内股票之间的价格联动关系，让模型在判断某只股票时能"看到"同行业股票的涨跌信息。例如判断宁德时代时，同时输入比亚迪、赣锋锂业等相关股票的特征作为邻居节点信息。

---

## 附录：关键文件速查

| 想要修改... | 对应文件 | 关键位置 |
|------------|---------|---------|
| 股票池 | `config.py` | `STOCK_POOL` 列表 |
| 买卖阈值 | `config.py` | `BUY_THRESHOLD`, `SELL_THRESHOLD` |
| 止损比例 | `config.py` | `STOP_LOSS_RATIO`, `PORTFOLIO_STOP` |
| 模型结构 | `models/lstm_model.py` | `LSTMModel.__init__` |
| 特征列表 | `features/builder.py` | `FEATURE_NAMES` |
| 标签定义 | `features/builder.py` | `build_sequences` 内标签计算 |
| 每日运行时间 | `config.py` | `DAILY_RUN_TIME` |
| 初始资金 | `config.py` | `INIT_CAPITAL` |
| 看板端口 | `config.py` | `DASHBOARD_PORT` |

---

*AlphaQuant 仅用于量化研究和学习目的，不构成投资建议。*
