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
    LEARNING_RATE, TRAIN_RATIO, MODEL_SAVE_DIR,
)
from models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)


def train_model(X: np.ndarray, y: np.ndarray) -> LSTMModel:
    """
    训练 LSTM 模型（时序分割，禁止随机打乱）。

    Args:
        X: (N, 20, 8) 特征序列
        y: (N,) 二分类标签

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

    model     = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=LR_PATIENCE, factor=0.5
    )

    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0

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

        scheduler.step(val_auc)

        msg = (f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
               f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        logger.info(msg)
        print(msg)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f"早停：{EARLY_STOP_PATIENCE} 轮无提升，最佳 Val AUC: {best_val_auc:.4f}")
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

    return model


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
