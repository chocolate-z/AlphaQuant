# LSTM模型定义：两层LSTM + 全连接层，输出买入概率

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_HIDDEN1, LSTM_HIDDEN2, FC_HIDDEN, DROPOUT, FEATURE_DIM


class LSTMModel(nn.Module):
    """
    两层 LSTM + FC 分类模型，预测未来5日内涨幅超5%的概率。

    Input:  (batch_size, seq_len=20, input_size=8)
    Output: (batch_size, 1) — sigmoid 买入概率
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

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc1     = nn.Linear(hidden2, fc_hidden)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(fc_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, 1) in [0, 1]
        """
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]   # 取最后一步
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def load_model(model_path: str, device: str = "cpu") -> LSTMModel:
    """
    从 .pt 文件加载模型权重。

    Args:
        model_path: 权重文件路径
        device: 'cpu' 或 'cuda'

    Returns:
        处于 eval 模式的 LSTMModel
    """
    model = LSTMModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
