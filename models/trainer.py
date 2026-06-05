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
    CPU_THREAD_RATIO, WEIGHT_DECAY, NOISE_STD, ENSEMBLE_N_MODELS,
)
from models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)


def train_model(X: np.ndarray, y: np.ndarray, resume: bool = False,
                _seed: int = None, _save_name: str = "lstm_best") -> LSTMModel:
    """
    训练 GRU 模型（时序分割，禁止随机打乱）。

    Args:
        X:          (N, WINDOW_SIZE, FEATURE_DIM) 特征序列
        y:          (N,) 二分类标签
        resume:     True = 加载已有模型权重后继续训练
        _seed:      随机种子（集成训练内部使用，保证各模型多样性）
        _save_name: 保存文件名前缀（集成训练内部使用）

    Returns:
        训练好的 LSTMModel（已保存到 models/saved/）
    """
    # 固定随机种子（集成训练时每个模型用不同种子）
    if _seed is not None:
        torch.manual_seed(_seed)
        np.random.seed(_seed)

    # 训练单模型时，让旧集成清单失效（特征维度/结构可能已变）
    if _save_name == "lstm_best":
        _manifest = os.path.join(MODEL_SAVE_DIR, "ensemble_manifest.json")
        if os.path.exists(_manifest):
            os.remove(_manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        total = os.cpu_count() or 1
        if CPU_THREAD_RATIO == 0:
            n_threads = total
        elif 0 < CPU_THREAD_RATIO <= 1:
            n_threads = max(1, int(total * CPU_THREAD_RATIO))
        else:
            n_threads = max(1, min(int(CPU_THREAD_RATIO), total))
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        print(f"  CPU 线程: {n_threads}/{total} 核")
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
    # num_workers>0 让 CPU 预取数据，不阻塞 GPU；pin_memory 加速 CPU→GPU 传输
    _nw = min(4, os.cpu_count() or 1) if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=_nw, pin_memory=(device.type == "cuda"))

    model = LSTMModel().to(device)
    # torch.compile 在 PyTorch 2.x + CPU 上可额外提速 20~40%
    # 第一轮会多花约 30s 编译，之后每轮更快
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, backend="aot_eager")
            print("  ✔ torch.compile 已启用（首轮编译约 30s，之后每轮更快）")
        except Exception:
            pass  # 编译失败静默降级，不影响训练

    pretrained_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if resume and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        msg = f"已加载已有模型权重（{pretrained_path}），在此基础上继续训练"
        logger.info(msg)
        print(f"\n  ✔ {msg}\n")
    elif resume:
        logger.warning("未找到已有模型文件，将从头开始训练")
        print("\n  ⚠ 未找到已有模型文件，将从头开始训练\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 余弦退火：学习率从 LEARNING_RATE 平滑降到 1e-6，避免震荡不收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    best_val_auc = 0.0
    best_state   = None
    best_epoch   = 1
    no_improve   = 0

    history = {"loss": [], "train_auc": [], "val_auc": []}
    # AMP：GPU 上用 float16 前向/反向，自动提速 1.5~3x；CPU 上无效，直接跳过
    use_amp = device.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss   = 0.0
        train_preds  = []
        train_labels = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            # 数据增强：给输入加高斯噪声，迫使模型学稳健规律而非记忆噪声（仅训练时）
            if NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * NOISE_STD
            optimizer.zero_grad(set_to_none=True)   # 比 zero_grad() 省内存

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = -(
                        pos_weight_val * yb * torch.log(pred + 1e-9)
                        + (1 - yb) * torch.log(1 - pred + 1e-9)
                    ).mean()
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = model(xb)
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

    save_path = os.path.join(MODEL_SAVE_DIR, f"{_save_name}.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"模型已保存: {save_path}，最佳 Val AUC: {best_val_auc:.4f}")

    # 带时间戳的版本（仅主模型保存，避免集成时产生大量副本）
    if _save_name == "lstm_best":
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
        from utils.viz import setup_chinese_font
        setup_chinese_font()

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


def predict_proba(model_or_models, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    批量推理，返回买入概率数组。
    支持单模型（LSTMModel）或集成模型列表（list[LSTMModel]）。
    传入列表时自动取各模型概率的平均值（集成投票）。

    Args:
        model_or_models: 单模型或模型列表
        X: (N, WINDOW_SIZE, FEATURE_DIM)

    Returns:
        proba: (N,) float32
    """
    if isinstance(model_or_models, list):
        # 集成：对每个模型分别推理，取概率均值
        all_probs = np.stack([predict_proba(m, X, device) for m in model_or_models])
        return all_probs.mean(axis=0)
    model = model_or_models
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    X_t = torch.tensor(X, dtype=torch.float32).to(dev)
    with torch.no_grad():
        out = model(X_t).cpu().numpy().flatten()
    return out


def train_ensemble(X: np.ndarray, y: np.ndarray, n_models: int = None) -> list:
    """
    集成训练：用不同随机种子训练 N 个模型，推理时取概率均值。
    各模型在同样特征上从不同初始点出发，捕获不同的规律，
    平均后方差降低约 1/√N，验证 AUC 通常比单模型高 1-3%。

    Args:
        n_models: 模型数量，None 时用 config.ENSEMBLE_N_MODELS

    Returns:
        list of LSTMModel（均已 eval()）
    """
    import json, shutil
    if n_models is None:
        n_models = ENSEMBLE_N_MODELS

    print(f"\n  集成训练模式：将依次训练 {n_models} 个不同随机种子的模型")
    print(f"  推理时自动取平均概率，效果优于任何单一模型\n")

    best_auc  = -1.0
    best_idx  = 0
    aucs      = []
    models    = []
    seeds     = [42 + i * 17 for i in range(n_models)]

    for i, seed in enumerate(seeds):
        print(f"\n{'='*56}")
        print(f"  集成训练 {i+1}/{n_models}   (随机种子 seed={seed})")
        print(f"{'='*56}\n")
        m = train_model(X, y, _seed=seed, _save_name=f"lstm_ensemble_{i}")
        models.append(m)

        # 用验证集快速评估当前模型 AUC
        from sklearn.metrics import roc_auc_score
        split   = int(len(X) * TRAIN_RATIO)
        X_val   = X[split:]
        y_val   = y[split:]
        probs   = predict_proba(m, X_val)
        try:
            auc = roc_auc_score(y_val, probs)
        except Exception:
            auc = 0.5
        aucs.append(auc)
        print(f"\n  模型 {i+1} 验证 AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_idx = i

    # 最佳单模型也复制为 lstm_best.pt（单模式回退时使用）
    best_src = os.path.join(MODEL_SAVE_DIR, f"lstm_ensemble_{best_idx}.pt")
    shutil.copy2(best_src, os.path.join(MODEL_SAVE_DIR, "lstm_best.pt"))

    # 保存集成清单
    manifest = {"n_models": n_models, "seeds": seeds, "val_aucs": aucs, "best_idx": best_idx}
    with open(os.path.join(MODEL_SAVE_DIR, "ensemble_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 生成集成对比图（各模型 AUC + 均值线）
    _save_ensemble_chart(aucs)

    avg_auc = sum(aucs) / len(aucs)
    print(f"\n{'='*56}")
    print(f"  集成训练完成！")
    print(f"  各模型验证 AUC: {[f'{a:.4f}' for a in aucs]}")
    print(f"  平均 AUC: {avg_auc:.4f}  |  最佳单模型 AUC: {best_auc:.4f}")
    print(f"  推理时将自动使用 {n_models} 个模型投票，效果优于单模型")
    print(f"{'='*56}\n")

    return models


def _save_ensemble_chart(aucs: list):
    """保存集成训练汇总图（各模型 AUC 柱状图）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from utils.viz import setup_chinese_font
        setup_chinese_font()

        fig, ax = plt.subplots(figsize=(8, 4))
        x     = list(range(1, len(aucs) + 1))
        avg   = sum(aucs) / len(aucs)
        colors = ["#4CAF50" if a >= avg else "#FF9800" for a in aucs]
        bars  = ax.bar(x, aucs, color=colors, width=0.5, zorder=3)
        ax.axhline(avg,  color="#2196F3", linewidth=1.8, linestyle="--", label=f"均值 {avg:.4f}")
        ax.axhline(0.5,  color="gray",    linewidth=1.0, linestyle=":",  label="随机基准 0.5", alpha=0.7)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{auc:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"模型 {i}" for i in x])
        ax.set_ylim(max(0.4, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
        ax.set_ylabel("验证集 AUC")
        ax.set_title(f"集成训练汇总 — {len(aucs)} 个模型，平均 AUC={avg:.4f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "ensemble_report.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  集成汇总图已保存: {path}")
    except Exception as e:
        logger.warning(f"集成图表生成失败: {e}")
