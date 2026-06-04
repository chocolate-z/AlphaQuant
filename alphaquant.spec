# PyInstaller 打包配置
# 用法：pyinstaller alphaquant.spec

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# 收集 torch/numpy/sklearn 的数据文件和动态库
datas = []
datas += collect_data_files('torch')
datas += collect_data_files('sklearn')
datas += collect_data_files('matplotlib')

# 需要随 exe 一起打包的项目文件夹
datas += [
    ('config.py',       '.'),
    ('data',            'data'),
    ('models',          'models'),
    ('features',        'features'),
    ('backtest',        'backtest'),
    ('paper_trading',   'paper_trading'),
    ('dashboard',       'dashboard'),
    ('diagnose',        'diagnose'),
    ('utils',           'utils'),
]

binaries = []
binaries += collect_dynamic_libs('torch')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'torch', 'torch.nn', 'torch.optim',
        'sklearn.metrics', 'sklearn.preprocessing',
        'pandas', 'numpy', 'matplotlib',
        'matplotlib.backends.backend_agg',
        'requests', 'flask', 'flask_cors',
        'openpyxl', 'joblib',
        'data.loader', 'data.realtime', 'data.index_fetcher',
        'features.builder',
        'models.lstm_model', 'models.trainer',
        'backtest.engine', 'backtest.metrics',
        'paper_trading.executor', 'paper_trading.account',
        'paper_trading.scheduler', 'paper_trading.risk',
        'paper_trading.logger',
        'dashboard.app',
        'diagnose.analyzer', 'diagnose.report',
        'utils.viz',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'tkinter', 'test', 'unittest',
        'IPython', 'jupyter', 'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AlphaQuant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,          # 压缩可执行文件（需安装 upx）
    console=True,      # 保留控制台窗口（菜单交互需要）
    icon=None,         # 如有图标文件改为 'icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AlphaQuant',
)
