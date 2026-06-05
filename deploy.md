# AlphaQuant 服务器部署教程（宝塔面板）

> 目标：本地训练好模型 → 部署到服务器 → 服务器每个交易日**自动跑模拟盘**（自动买卖、记盈亏）→ 你有空上网页看盈亏。
> 注意：这是**纸面（模拟）交易**。真金白银需照信号手动下单，或自行对接券商 API（本系统不含）。

---

## 〇、前提
1. **国内服务器**（阿里云/腾讯云等，北京/上海区最佳）——必须能访问国内财经站（腾讯/搜狐/新浪）。境外服务器很可能拉不到数据。
2. 服务器已装**宝塔面板**。
3. 本地已经训练好模型（`models/saved/lstm_best.pt` 存在）。
4. 服务器时区设为 **Asia/Shanghai**（宝塔→计划任务依赖正确时区）。

---

## 一、本地准备要上传的数据/模型（git 不带这两样）
在本地 AlphaQuant 目录，把这两个文件夹各打成 zip：
- `data/cache/`   —— 历史行情缓存（模拟盘要用）
- `models/saved/` —— 训练好的模型

（代码用 git 拉，**数据和模型必须手动传**，因为它们被 .gitignore 排除了。）

---

## 二、宝塔上装 Python 环境
1. 宝塔 →「软件商店」→ 搜索安装 **Python项目管理器**（或「PM2管理器」也行）。
2. 在 Python项目管理器里**安装 Python 3.10（或更高）版本**。
   （或者你也可以走第四步的 SSH 终端自建 venv，二选一。）

---

## 三、上传代码 + 数据 + 模型
1. 代码：宝塔「终端」里
   ```bash
   cd /www/wwwroot
   git clone <你的仓库地址> AlphaQuant
   cd AlphaQuant
   git checkout claude/modest-goodall-v5Egd
   ```
   （或本地打包整个项目，用宝塔「文件」上传到 `/www/wwwroot/AlphaQuant` 解压。）
2. 数据/模型：宝塔「文件」进入 `/www/wwwroot/AlphaQuant/data/`，上传 `cache.zip` 解压到 `data/cache/`；
   进入 `models/`，上传 `saved.zip` 解压到 `models/saved/`。

---

## 四、装依赖（宝塔「终端」）
```bash
cd /www/wwwroot/AlphaQuant
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU版，省空间
```

---

## 五、部署前自检（重要）
```bash
cd /www/wwwroot/AlphaQuant
# 1) 测网络能不能拉数据（出数字=通；报错/0=服务器访问不了财经站，要换国内服务器）
./venv/bin/python -c "from data.loader import _fetch_kline; print('拉到', len(_fetch_kline('sh600519','20260101','20260605')), '条')"
# 2) 手动跑一次模拟盘，确认能正常买卖
./venv/bin/python daily_paper.py
```
跑完看 `logs/daily_paper.log`、`logs/account_state.json`（持仓）确认正常。

---

## 六、配每日自动运行（宝塔「计划任务」）
宝塔 →「计划任务」→ 添加任务：
- **任务类型**：Shell脚本
- **任务名称**：AlphaQuant每日模拟盘
- **执行周期**：每天，时间 **15:05**（A股收盘后）
- **脚本内容**：
  ```bash
  cd /www/wwwroot/AlphaQuant && ./venv/bin/python daily_paper.py >> logs/cron.log 2>&1
  ```
保存即可。它每天会自动：抓新票生长候选池 → 刷新缓存 → 跑模拟盘自动买卖（非交易日自动跳过）。

---

## 七、随时上网页看盈亏（看板）
**方式 A（最简单、最安全）**：看文件
- 宝塔「文件」看 `logs/account_daily.csv`（每日总资产/盈亏）、`logs/daily_paper.log`。

**方式 B（网页看板，能看净值曲线/持仓/今日信号）**：
1. 改 `config.py`：把 `DASHBOARD_HOST = "127.0.0.1"` 改成 `"0.0.0.0"`。
2. 用宝塔 Python项目管理器「添加项目」，启动命令：
   ```
   /www/wwwroot/AlphaQuant/venv/bin/python main.py --mode dashboard
   ```
   设为常驻（开机自启）。
3. 宝塔「安全」放行端口 **5000**。
4. ⚠️ **安全（务必做）**：看板的「操作中心」能触发训练/回测/重置账户，**公网裸奔很危险**。三选一：
   - **最稳**：不放行公网，用 SSH 隧道访问 `ssh -L 5000:127.0.0.1:5000 root@服务器IP`，本地浏览器开 `http://localhost:5000`；
   - 或宝塔「网站」建反向代理到 127.0.0.1:5000 并**加 Basic Auth 密码**；
   - 或宝塔「安全」里把 5000 端口**只放行你自己的 IP**。

---

## 八、日常维护
- **更新模型**：本地重新训练后，把新的 `models/saved/lstm_best.pt` 重新传到服务器覆盖即可（无需在服务器上训练，省 CPU）。
- **重置账户**：`./venv/bin/python -c "import main; main.reset_account()"`（换策略/重新开始时）。
- **看日志**：`tail -f logs/daily_paper.log`。

---

## 九、再次提醒（别跳过）
- 这是**纸面前向验证**，先连续跑 **3–6 个月**看真盘前向表现，再谈真钱。
- 诚实样本外超额仅 +0.78%（≈跟随大盘），别期待跑赢大盘；它的价值是「有纪律、可控回撤」。
- 持仓上限 5 只更集中、波动更大；大盘跌破 20 日均线会自动空仓避险。
