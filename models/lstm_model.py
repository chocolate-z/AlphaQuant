# 序列模型：两层 GRU + Attention + FC，预测未来5日内涨幅超5%的概率
# GRU 比 LSTM 参数少 25%，CPU 上速度快 25~30%，精度相当

import json
import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_HIDDEN1, LSTM_HIDDEN2, FC_HIDDEN, DROPOUT, FEATURE_DIM, MODEL_SAVE_DIR


class LSTMModel(nn.Module):
    """
    两层 GRU + Scaled-Dot-Product Attention + 3层FC 分类模型。
    类名保持 LSTMModel 避免改动加载/保存代码。

    Input:  (batch, seq_len=30, features=16)
    Output: (batch, 1) — sigmoid 买入概率
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden1: int = LSTM_HIDDEN1,
        hidden2: int = LSTM_HIDDEN2,
        fc_hidden: int = FC_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden2 = hidden2

        self.input_norm = nn.BatchNorm1d(input_size)

        # GRU：比 LSTM 少 1/3 参数，CPU 更快
        self.gru1     = nn.GRU(input_size=input_size, hidden_size=hidden1,
                               num_layers=1, batch_first=True)
        self.norm1    = nn.LayerNorm(hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.gru2     = nn.GRU(input_size=hidden1, hidden_size=hidden2,
                               num_layers=1, batch_first=True)
        self.norm2    = nn.LayerNorm(hidden2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1     = nn.Linear(hidden2, fc_hidden * 2)
        self.bn_fc1  = nn.BatchNorm1d(fc_hidden * 2)
        self.relu1   = nn.ReLU()
        self.drop_fc = nn.Dropout(dropout * 0.5)
        self.fc2     = nn.Linear(fc_hidden * 2, fc_hidden)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(fc_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        xn = self.input_norm(x.reshape(-1, f)).reshape(b, t, f)

        out, _ = self.gru1(xn)
        out = self.norm1(out)
        out = self.dropout1(out)

        out, _ = self.gru2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        # Scaled dot-product attention：让模型自动找关键时间步
        query   = out[:, -1:, :]
        scores  = torch.bmm(query, out.transpose(1, 2)) / (self.hidden2 ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out     = torch.bmm(weights, out).squeeze(1)

        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.relu1(out)
        out = self.drop_fc(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return self.sigmoid(out)


def strip_compile_prefix(state: dict) -> dict:
    """
    去掉 torch.compile 包装产生的 '_orig_mod.' 键前缀。

    train_model 里用 torch.compile 包装过模型，其 state_dict 的每个键都会带
    '_orig_mod.' 前缀；普通 LSTMModel 的键没有该前缀，直接 load 会因键名不匹配
    而报错。保存/加载时统一剥掉前缀即可兼容两种来源的权重文件。
    """
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state.items()}
    return state


def load_model(model_path: str, device: str = "cpu") -> LSTMModel:
    model = LSTMModel()
    state = torch.load(model_path, map_location=device, weights_only=True)
    state = strip_compile_prefix(state)   # 兼容 torch.compile 保存的旧权重
    model.load_state_dict(state)
    model.eval()
    return model


def load_ensemble(device: str = "cpu") -> list:
    """
    加载集成模型列表。如果集成文件不存在则返回空列表。
    推理时对所有模型的概率取平均，显著降低单模型方差。
    """
    manifest_path = os.path.join(MODEL_SAVE_DIR, "ensemble_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path) as f:
            info = json.load(f)
        models = []
        for i in range(info.get("n_models", 0)):
            p = os.path.join(MODEL_SAVE_DIR, f"lstm_ensemble_{i}.pt")
            if os.path.exists(p):
                models.append(load_model(p, device))
        if models:
            return models
    except Exception:
        pass
    return []


def load_best_available(device: str = "cpu"):
    """
    统一加载入口：优先返回集成模型列表，否则返回单模型，否则返回 None。
    predict_proba() 已支持接收列表，所有调用方无需关心具体形式。
    """
    ens = load_ensemble(device)
    if ens:
        return ens
    p = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if os.path.exists(p):
        return load_model(p, device)
    return None
