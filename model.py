"""
model.py - 模型定义模块
=====================

包含LSTM性别识别模型
"""

import torch
import torch.nn as nn


class LSTMGenderDetector(nn.Module):
    """基于LSTM的性别识别模型"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # 双向LSTM，所以是2倍hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, features):
        # 为LSTM添加序列维度
        features = features.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM前向传播
        out, _ = self.lstm(features)
        
        # 使用最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        output = self.fc(out)
        return output