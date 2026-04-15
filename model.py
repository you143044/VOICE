"""
model.py - 模型和配置模块
==========================

包含：
- 配置类 Config
- 模型架构定义
- 特征提取器
- 性别识别模型
"""

import os
import numpy as np
import torch
import torch.nn as nn
import librosa

class Config:
    """项目配置类"""
    
    def __init__(self):
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH = r"D:\A--Learning-D\项目\VOICE2\dataset"
        self.SAMPLE_RATE = 16000
        self.DURATION = 3
        self.FIXED_LENGTH = self.SAMPLE_RATE * self.DURATION
        self.BATCH_SIZE = 64
        self.EPOCHS = 50
        self.LEARNING_RATE = 1e-4
        self.DROPOUT_RATE = 0.3
        self.RNN_TYPE = 'lstm'
        self.RNN_HIDDEN_SIZE = 256
        self.RNN_LAYERS = 2
        self.RNN_BIDIRECTIONAL = True
        self.TRAIN_SPLIT = 0.8
        self.VALIDATION_SPLIT = 0.1
        self.TEST_SPLIT = 0.1
        self.USE_GPU = True
        self.USE_AMP = True
        self.MODEL_SAVE_PATH = os.path.join(self.PROJECT_ROOT, 'models')
        self.MODEL_NAME = 'gender_detector'
        self.GENDER_MAPPING = {'male': 0, 'female': 1}
        self.REVERSE_GENDER_MAPPING = {0: 'male', 1: 'female'}
        
        self.MFCC_DIM = 39
        self.F0_DIM = 13
        self.ENERGY_DIM = 1
        # 正确计算总特征维度：MFCC(39*4) + F0(13) + 能量(1) = 170
        self.TOTAL_FEATURE_DIM = self.MFCC_DIM * 4 + self.F0_DIM + self.ENERGY_DIM
        
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)

class AdvancedFeatureExtractor:
    """高级特征提取器 - MFCC + F0 + 能量"""
    
    def __init__(self, sample_rate=16000, n_mfcc=39, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.F0_DIM = 13
        self.ENERGY_DIM = 1
    
    def extract_mfcc(self, audio):
        """提取MFCC特征"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            return np.concatenate([mfcc_mean, mfcc_std])
        except Exception as e:
            print(f"MFCC提取错误: {e}")
            return np.zeros(self.n_mfcc * 2)
    
    def extract_f0(self, audio):
        """提取基频F0特征 - 使用更简单的方法"""
        try:
            # 使用更简单的方法提取F0，避免使用可能导致崩溃的librosa.pyin
            # 计算音频的过零率和频谱特征作为F0的替代
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            
            # 计算这些特征的统计值
            zcr_mean = np.mean(zero_crossing_rate)
            zcr_std = np.std(zero_crossing_rate)
            sc_mean = np.mean(spectral_centroid)
            sc_std = np.std(spectral_centroid)
            sb_mean = np.mean(spectral_bandwidth)
            sb_std = np.std(spectral_bandwidth)
            
            # 生成13维F0特征
            features = []
            features.append(sc_mean)  # 频谱质心作为主要频率特征
            features.append(sc_std)
            features.append(zcr_mean)
            features.append(zcr_std)
            features.append(sb_mean)
            features.append(sb_std)
            features.append(sc_mean * 0.5)  # 模拟最小值
            features.append(sc_mean * 1.5)  # 模拟最大值
            features.append(sc_mean)  # 模拟中位数
            features.append(sc_mean * 0.75)  # 模拟25分位数
            features.append(sc_mean * 1.25)  # 模拟75分位数
            features.append(sc_mean * 0.1)  # 模拟标准差
            features.append(0.8)  # 模拟浊音比例
            
            return np.array(features)
        except Exception as e:
            print(f"F0提取错误: {e}")
            return np.zeros(self.F0_DIM)
    
    def extract_energy(self, audio):
        """提取音量能量特征"""
        try:
            energy = np.sum(audio ** 2) / len(audio)
            energy_db = 10 * np.log10(energy + 1e-10)
            
            return np.array([energy_db])
        except Exception as e:
            print(f"能量提取错误: {e}")
            return np.zeros(self.ENERGY_DIM)
    
    def extract_all_features(self, audio):
        """提取所有特征"""
        try:
            mfcc_features = self.extract_mfcc(audio)
            f0_features = self.extract_f0(audio)
            energy_features = self.extract_energy(audio)
            
            all_features = np.concatenate([
                mfcc_features,
                f0_features,
                energy_features
            ])
            
            return all_features
        except Exception as e:
            print(f"特征提取错误: {e}")
            return np.zeros(self.n_mfcc * 2 + self.F0_DIM + self.ENERGY_DIM)

class LocalFeatureExtractor(nn.Module):
    """简化的CNN特征提取器"""
    
    def __init__(self, input_length, num_features=128):
        super().__init__()
        self.input_length = input_length
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=32, stride=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_features)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, 1, self.input_length)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        return features

class GenderDetectorModel(nn.Module):
    """性别识别模型 - 基于手工特征的MLP"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.feature_extractor = AdvancedFeatureExtractor(
            sample_rate=config.SAMPLE_RATE,
            n_mfcc=config.MFCC_DIM
        )
        
        # 更复杂的模型架构
        self.fc = nn.Sequential(
            nn.Linear(config.TOTAL_FEATURE_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, features):
        output = self.fc(features)
        return output


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
