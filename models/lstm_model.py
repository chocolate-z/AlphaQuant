# 序列模型：两层 GRU + Attention + FC，预测未来5日内涨幅超5%的概率
# GRU 比 LSTM 参数少 25%，CPU 上速度快 25~30%，精度相当

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_HIDDEN1, LSTM_HIDDEN2, FC_HIDDEN, DROPOUT, FEATURE_DIM


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


def load_model(model_path: str, device: str = "cpu") -> LSTMModel:
    model = LSTMModel()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
