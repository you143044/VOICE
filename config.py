"""
config.py - 项目配置文件
==================================

包含项目所有配置参数
"""

import os

class Config:
    """项目配置类"""
    
    def __init__(self):
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH = os.path.join(self.PROJECT_ROOT, 'dataset')
        self.DATASET_PROCESSED_PATH = os.path.join(self.PROJECT_ROOT, 'dataset_processed')
        self.FEATURES_PATH = os.path.join(self.PROJECT_ROOT, 'features')
        self.MODELS_PATH = os.path.join(self.PROJECT_ROOT, 'models')
        
        self.SAMPLE_RATE = 16000
        self.DURATION = 3
        self.FIXED_LENGTH = self.SAMPLE_RATE * self.DURATION
        
        self.BATCH_SIZE = 64
        self.EPOCHS = 30
        self.LEARNING_RATE = 1e-4
        self.DROPOUT_RATE = 0.3
        
        self.RNN_HIDDEN_SIZE = 128
        self.RNN_LAYERS = 2
        self.RNN_BIDIRECTIONAL = True
        
        self.TRAIN_SPLIT = 0.8
        self.VALIDATION_SPLIT = 0.1
        self.TEST_SPLIT = 0.1
        
        self.USE_GPU = True
        
        self.GENDER_MAPPING = {'male': 0, 'female': 1}
        self.REVERSE_GENDER_MAPPING = {0: 'male', 1: 'female'}
        
        # 特征维度
        self.FEATURE_DIM = 13
        
        # 创建必要的目录
        os.makedirs(self.DATASET_PROCESSED_PATH, exist_ok=True)
        os.makedirs(self.FEATURES_PATH, exist_ok=True)
        os.makedirs(self.MODELS_PATH, exist_ok=True)