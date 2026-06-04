# 训练器：时序分割、BCELoss + pos_weight、早停、AUC监控

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR_PATIENCE,
    LEARNING_RATE, TRAIN_RATIO, MODEL_SAVE_DIR, REPORTS_DIR,
)
from models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)


def train_model(X: np.ndarray, y: np.ndarray, resume: bool = False) -> LSTMModel:
    """
    训练 LSTM 模型（时序分割，禁止随机打乱）。

    Args:
        X: (N, WINDOW_SIZE, FEATURE_DIM) 特征序列
        y: (N,) 二分类标签
        resume: True = 加载已有模型权重后继续训练（增量训练）

    Returns:
        训练好的 LSTMModel（已保存到 models/saved/）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    n = len(X)
    split = int(n * TRAIN_RATIO)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info(f"训练集: {len(X_train)}，验证集: {len(X_val)}")
    logger.info(f"正样本比例 - 训练: {y_train.mean():.3f}，验证: {y_val.mean():.3f}")

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight_val = neg_count / max(pos_count, 1)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl_np = y_val.copy()

    train_ds     = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel().to(device)
    pretrained_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if resume and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        msg = f"已加载已有模型权重（{pretrained_path}），在此基础上继续训练"
        logger.info(msg)
        print(f"\n  ✔ {msg}\n")
    elif resume:
        logger.warning("未找到已有模型文件，将从头开始训练")
        print("\n  ⚠ 未找到已有模型文件，将从头开始训练\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 余弦退火：学习率从 LEARNING_RATE 平滑降到 1e-6，避免震荡不收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    best_val_auc = 0.0
    best_state   = None
    best_epoch   = 1
    no_improve   = 0

    history = {"loss": [], "train_auc": [], "val_auc": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss   = 0.0
        train_preds  = []
        train_labels = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            # 手动加权 BCE（处理正负样本不平衡）
            loss = -(
                pos_weight_val * yb * torch.log(pred + 1e-9)
                + (1 - yb) * torch.log(1 - pred + 1e-9)
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(xb)
            train_preds.extend(pred.detach().cpu().numpy().flatten())
            train_labels.extend(yb.detach().cpu().numpy().flatten())

        avg_loss = total_loss / len(X_train)

        try:
            train_auc = roc_auc_score(train_labels, train_preds)
        except Exception:
            train_auc = 0.5

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl).cpu().numpy().flatten()
        try:
            val_auc = roc_auc_score(y_vl_np, val_pred)
        except Exception:
            val_auc = 0.5

        scheduler.step()
        history["loss"].append(avg_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        # 用中文解释进度，方便小白理解
        trend = "↑ 提升" if val_auc > best_val_auc else ("→ 持平" if no_improve < 3 else "↓ 停滞")
        msg = (
            f"第{epoch:3d}轮 | "
            f"损失(越低越好): {avg_loss:.4f} | "
            f"训练识别率: {train_auc:.4f} | "
            f"验证识别率: {val_auc:.4f} {trend}"
        )
        logger.info(msg)
        print(msg)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                msg = f"早停：{EARLY_STOP_PATIENCE} 轮无提升，最佳验证识别率: {best_val_auc:.4f}"
                logger.info(msg)
                print(f"\n  ⏹ {msg}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"模型已保存: {save_path}，最佳 Val AUC: {best_val_auc:.4f}")

    # 同时保存带时间戳的版本，避免覆盖历史最优模型
    import shutil
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_SAVE_DIR, f"lstm_{ts}_auc{best_val_auc:.4f}.pt")
    shutil.copy2(save_path, versioned_path)
    logger.info(f"版本副本已保存: {versioned_path}")

    # 生成训练可视化报告
    chart_path = _save_training_chart(history, best_epoch, best_val_auc)
    if chart_path:
        logger.info(f"训练报告已保存: {chart_path}")
        print(f"\n  📊 训练报告已保存至: {chart_path}")

    return model


def _save_training_chart(history: dict, best_epoch: int, best_val_auc: float) -> str:
    """
    保存训练曲线图（Loss曲线 + AUC曲线），帮助直观评估模型学习情况。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 自动检测中文字体
        candidates = [
            "Microsoft YaHei", "SimHei", "SimSun",
            "Heiti SC", "PingFang SC", "STHeiti",
            "WenQuanYi Micro Hei", "Noto Sans CJK SC",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.family"] = name
                break
        plt.rcParams["axes.unicode_minus"] = False

        epochs     = list(range(1, len(history["loss"]) + 1))
        losses     = history["loss"]
        train_aucs = history["train_auc"]
        val_aucs   = history["val_auc"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ── 损失曲线 ──
        ax1.plot(epochs, losses, color="#2196F3", linewidth=1.8, label="训练损失")
        ax1.axvline(best_epoch, color="#F44336", linestyle="--", alpha=0.8,
                    label=f"最佳轮次 {best_epoch}")
        ax1.set_title("训练损失曲线（越低模型越收敛）", fontsize=12)
        ax1.set_xlabel("训练轮次 (Epoch)")
        ax1.set_ylabel("损失值 (Loss)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ── AUC曲线 ──
        ax2.plot(epochs, train_aucs, color="#2196F3", linewidth=1.8, label="训练识别率")
        ax2.plot(epochs, val_aucs,   color="#4CAF50", linewidth=1.8, label="验证识别率")
        ax2.axvline(best_epoch, color="#F44336", linestyle="--", alpha=0.8,
                    label=f"最佳轮次 {best_epoch}")
        ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="随机猜测基准 0.5")
        ax2.fill_between(epochs, 0.5, val_aucs,
                         where=[v > 0.5 for v in val_aucs],
                         alpha=0.12, color="#4CAF50", label="优于随机区域")
        ax2.set_ylim(0.3, 1.0)
        ax2.set_title("识别率曲线 AUC（越高模型越准确）", fontsize=12)
        ax2.set_xlabel("训练轮次 (Epoch)")
        ax2.set_ylabel("AUC（0.5=瞎猜，1.0=完美）")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        fig.suptitle(
            f"AlphaQuant 模型训练报告 — 最佳验证识别率: {best_val_auc:.4f}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        path = os.path.join(REPORTS_DIR, "training_report.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"训练图表生成失败: {e}")
        return ""


def predict_proba(model: LSTMModel, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    批量推理，返回买入概率数组。

    Args:
        model: LSTMModel
        X: (N, 20, 8)

    Returns:
        proba: (N,) float32
    """
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    X_t = torch.tensor(X, dtype=torch.float32).to(dev)
    with torch.no_grad():
        out = model(X_t).cpu().numpy().flatten()
    return out
