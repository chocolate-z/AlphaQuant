# LSTM模型定义：两层LSTM + 全连接层，输出买入概率

import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_HIDDEN1, LSTM_HIDDEN2, FC_HIDDEN, DROPOUT, FEATURE_DIM


class LSTMModel(nn.Module):
    """
    两层 LSTM + LayerNorm + FC 分类模型，预测未来5日内涨幅超5%的概率。

    Input:  (batch_size, seq_len=30, input_size=16)
    Output: (batch_size, 1) — sigmoid 买入概率

    改进：
    - 输入 BatchNorm：稳定训练初期梯度
    - 每层 LSTM 后加 LayerNorm：缓解梯度消失
    - FC 层加一层（128→64→32→1），增强非线性表达
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

        self.input_norm = nn.BatchNorm1d(input_size)

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1,
                             num_layers=1, batch_first=True)
        self.norm1    = nn.LayerNorm(hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2,
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
        # x: (batch, seq_len, input_size)
        b, t, f = x.shape
        # BatchNorm 作用在特征维：reshape → (b*t, f) → norm → reshape 回
        xn = self.input_norm(x.reshape(-1, f)).reshape(b, t, f)

        out, _ = self.lstm1(xn)
        out = self.norm1(out)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        out = out[:, -1, :]     # 取最后时间步
        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.relu1(out)
        out = self.drop_fc(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return self.sigmoid(out)


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
