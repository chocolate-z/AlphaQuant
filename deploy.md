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
   git clone https://github.com/chocolate-z/AlphaQuant.git AlphaQuant
   cd AlphaQuant
   # 默认分支就是 claude/modest-goodall-v5Egd，clone 完即最新，无需再 checkout
   ```
   - 私有仓库会要账号密码：用户名填 `chocolate-z`，密码填 **GitHub Personal Access Token**（不是登录密码，去 GitHub→Settings→Developer settings→Tokens 生成，勾 repo 权限）。
   - Ubuntu 若提示 venv 创建失败：`apt install -y python3.10-venv` 后重建。
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

## 六、配每日自动买卖（宝塔「计划任务」）
A股 15:00 收盘，收盘后跑一次出信号、自动买卖。你要「三点半后」，就设 **15:35**。

宝塔 →「计划任务」→ 添加任务：
- **任务类型**：Shell脚本
- **任务名称**：AlphaQuant每日模拟盘
- **执行周期**：每天，时间 **15:35**
- **脚本内容**：
  ```bash
  cd /www/wwwroot/AlphaQuant && ./venv/bin/python daily_paper.py >> logs/cron.log 2>&1
  ```
保存即可。它每天会自动：抓新票生长候选池 → 刷新缓存 → 跑模拟盘自动买卖。
**周末/节假日会自动跳过**（`daily_paper.py` 开头 `is_trade_day()` 判断，非交易日直接退出），所以设“每天”没问题。

> 不想用宝塔界面、想直接 crontab 的话（效果一样）：
> ```bash
> crontab -e
> # 末尾加一行（周一到周五 15:35）：
> 35 15 * * 1-5 cd /www/wwwroot/AlphaQuant && /www/wwwroot/AlphaQuant/venv/bin/python daily_paper.py >> /www/wwwroot/AlphaQuant/logs/cron.log 2>&1
> ```
> 跑完看 `logs/cron.log`、`logs/daily_paper.log` 确认。

---

## 七、随时上网页看盈亏（看板）
**方式 A（最简单、最安全）**：看文件
- 宝塔「文件」看 `logs/account_daily.csv`（每日总资产/盈亏）、`logs/daily_paper.log`。

**方式 B（网页看板，能看净值曲线/持仓/今日信号，且开机自启、崩了自动重启）**：

不用改代码——看板地址改由**环境变量** `DASHBOARD_HOST` 控制（默认 127.0.0.1 只本机可见）。下面用 **systemd** 让它常驻+开机自启（最稳）。

**1) 建服务文件**（宝塔「终端」执行 `nano /etc/systemd/system/alphaquant-dash.service`，粘贴）：
```ini
[Unit]
Description=AlphaQuant Dashboard
After=network.target

[Service]
Type=simple
WorkingDirectory=/www/wwwroot/AlphaQuant
Environment=KMP_DUPLICATE_LIB_OK=TRUE
Environment=DASHBOARD_HOST=0.0.0.0
ExecStart=/www/wwwroot/AlphaQuant/venv/bin/python main.py --mode dashboard
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
```
（`Ctrl+O` 回车保存，`Ctrl+X` 退出）

**2) 启用并启动**：
```bash
systemctl daemon-reload
systemctl enable --now alphaquant-dash   # 开机自启 + 立即启动
systemctl status alphaquant-dash         # 看到 active (running) 即成功
```
（看实时日志：`journalctl -u alphaquant-dash -f`；改完代码重启：`systemctl restart alphaquant-dash`）

**3) 放行端口 5000（两层都要开）**：
- 宝塔「安全」→ 放行 5000；
- **云服务器安全组**（腾讯云/阿里云控制台）→ 入站规则也放行 5000。两层只要漏一层就连不上。

**4) 浏览器访问**：`http://你的服务器公网IP:5000`

**5) ⚠️ 安全（务必做）**：看板「操作中心」能触发训练/重置账户，**公网裸奔很危险**。三选一：
- **最稳**：第3步**不放行公网**，改用 SSH 隧道——本地电脑跑 `ssh -L 5000:127.0.0.1:5000 root@服务器IP`，再开 `http://localhost:5000`（此时服务文件里 `DASHBOARD_HOST` 设回 `127.0.0.1` 即可）；
- 或宝塔「网站」建反向代理到 `127.0.0.1:5000` 并**加 Basic Auth 密码**（同样把服务设回 127.0.0.1）；
- 或安全组里 5000 端口**只放行你自己的公网 IP**（最省事，IP 变了要改）。

> 也可以用宝塔「Python项目管理器」添加项目（启动命令 `main.py --mode dashboard`、勾选开机自启）替代 systemd——但要在它的环境变量里加 `DASHBOARD_HOST=0.0.0.0` 和 `KMP_DUPLICATE_LIB_OK=TRUE`。systemd 更稳、崩溃自动拉起，推荐。

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
