# AlphaQuant — A股量化 AI 系统

基于 GRU 神经网络的 A 股端到端量化交易系统，涵盖数据获取、特征工程、模型训练、历史回测、模拟盘和可视化看板。

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **数据层** | 搜狐/腾讯/新浪三源自动降级，并行拉取，本地 CSV 缓存 |
| **特征工程** | 16个技术特征，逐窗口 Z-Score 归一化，无前视偏差 |
| **AI 模型** | 双层 GRU + Attention + 全连接，预测未来5日涨幅超5%的概率 |
| **回测引擎** | 严格 T+1、手续费、涨跌停、止损，对比5大基准指数 |
| **模拟盘** | 每日15:30自动执行，虚拟账户持久化 |
| **可视化** | 训练曲线、回测净值、单股买卖点图、网页看板 |

---

## 快速开始

### 1. 安装（首次）

**Windows：** 双击 `install.bat`

**Mac / Linux：**
```bash
bash install.sh
```

需要提前安装 [Python 3.10+](https://www.python.org/downloads/)，不需要 conda。

### 2. 启动

**Windows：** 双击 `start.bat`

**Mac / Linux：**
```bash
source venv/bin/activate
python main.py
```

### 3. 使用菜单

```
╔════════════════════════════════════════════════╗
║         AlphaQuant — A股量化 AI 系统           ║
╠════════════════════════════════════════════════╣
║  【训练 & 回测】                               ║
║  1  训练模型（完整模式，全A股随机100只）         ║
║  2  训练模型（快速模式，随机20只/近3年）         ║
║  3  继续训练（在已有模型基础上增量学习）         ║
║  4  历史回测                                   ║
╠════════════════════════════════════════════════╣
║  【工具】                                      ║
║  9  单股买卖点图（K线 + 模型信号）              ║
║  e  导出今日信号到 Excel                       ║
╚════════════════════════════════════════════════╝
```

---

## 项目结构

```
AlphaQuant/
├── main.py              # 主入口（交互菜单 + CLI）
├── config.py            # 全局配置
├── install.bat          # Windows 一键安装
├── start.bat            # Windows 一键启动
├── install.sh           # Mac/Linux 安装脚本
├── requirements.txt
│
├── data/
│   ├── loader.py        # 历史K线（搜狐→腾讯→新浪三源）
│   ├── realtime.py      # 实时行情（新浪）
│   └── index_fetcher.py # 基准指数数据
│
├── features/
│   └── builder.py       # 16个技术特征 + 逐窗口归一化
│
├── models/
│   ├── lstm_model.py    # GRU + Attention 模型
│   └── trainer.py       # 训练器（AdamW + 余弦退火）
│
├── backtest/
│   ├── engine.py        # 回测引擎
│   └── metrics.py       # 绩效指标
│
├── paper_trading/       # 模拟盘
├── dashboard/           # Flask 网页看板
├── diagnose/            # 单股诊断
└── utils/
    └── viz.py           # 共用绘图工具
```

---

## 模型说明

**输入：** 过去30个交易日的16维技术特征（动量/均线偏离/量能/震荡指标）

**架构：**
```
输入 (batch, 30, 16)
  → BatchNorm
  → GRU层1 (hidden=128) + LayerNorm
  → GRU层2 (hidden=64)  + LayerNorm
  → Scaled Dot-Product Attention
  → FC (128 → 64 → 1)
  → Sigmoid → 买入概率
```

**标签：** 未来5个交易日内最高价相对今日收盘涨幅 > 5% 为正样本

**交易策略：** 相对排名模式——每日按概率从高到低买入前N只，无需模型完全收敛也能产生交易

---

## 配置说明

修改 `config.py` 调整行为，无需改代码：

```python
FULL_STOCK_COUNT  = 100    # 完整模式从全A股随机抽取数量
QUICK_STOCK_COUNT = 20     # 快速模式抽取数量
START_DATE        = "20150101"  # 训练数据起始年份
MAX_EPOCHS        = 300    # 最大训练轮数
CPU_THREAD_RATIO  = 0.5    # 训练使用CPU核心比例（0=全部）
BUY_THRESHOLD     = 0.65   # 绝对模式买入阈值
RELATIVE_RANK_MODE = True  # True=排名模式，False=绝对阈值模式
```

---

## 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10 | 3.11+ |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 5 GB | SSD 10 GB |
| GPU | 不需要 | NVIDIA（训练加速） |
| 网络 | 需要国内网络访问金融数据 | — |

---

## 详细文档

完整使用说明见 [DOCS.md](DOCS.md)，包含：
- 所有配置参数说明
- 16个特征详细解释
- 回测绩效指标计算方法
- 模拟盘风控规则
- 网页看板使用说明
- 常见问题排查

---

## 免责声明

本项目仅用于量化研究和学习目的，不构成任何投资建议。股市有风险，入市需谨慎。
