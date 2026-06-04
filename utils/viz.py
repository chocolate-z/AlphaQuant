import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun",
    "Heiti SC", "PingFang SC", "STHeiti",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC",
]

def setup_chinese_font():
    """Auto-detect and set a Chinese font for matplotlib. Call once before plotting."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
