# 声音性别识别系统

## 项目简介

这是一个基于深度学习的声音性别识别系统，使用 LSTM 模型对音频进行性别分类。

## 核心功能

- 音频数据集格式统一（16kHz 单声道）
- 特征提取（13维音频特征）
- LSTM 模型训练和评估
- 图形化界面应用

## 项目结构

```
VOICE/
├── app.py                # 主应用程序（图形界面）
├── model.py              # 模型定义（包含LSTM模型）
├── train.py              # 原始训练脚本
├── train_lstm.py         # LSTM模型训练脚本
├── preprocess_final.py   # 音频预处理脚本
├── extract_features_light.py  # 轻量版特征提取脚本
├── dataset/              # 原始数据集
│   └── data_thchs30/     # THCHS-30 数据集
├── dataset_processed/    # 处理后的数据集
├── features/             # 提取的特征
└── models/               # 训练好的模型
```

## 快速开始

### 1. 预处理音频数据集

将音频统一为 16kHz 单声道格式：

```bash
python preprocess_final.py
```

### 2. 提取特征

提取音频特征用于模型训练：

```bash
python extract_features_light.py
```

### 3. 训练 LSTM 模型

使用提取的特征训练 LSTM 模型：

```bash
python train_lstm.py
```

### 4. 运行应用程序

启动图形化界面：

```bash
python app.py
```

## 技术说明

### 特征提取

- 能量特征（能量值、能量分贝）
- 过零率
- 频谱特征（FFT 均值、标准差）
- 时域统计特征（均值、标准差、最大值、最小值）

### 模型架构

- **LSTM 模型**：双向 LSTM，2 层，128 隐藏单元
- **输入维度**：13 维特征
- **输出**：二分类（男性/女性）

### 性能指标

- 验证准确率：58.81%
- 测试准确率：58.78%
- F1 分数：0.7404

## 数据说明

- **数据集**：THCHS-30（清华大学语音语料库）
- **样本数量**：26,776 个音频文件
- **性别分布**：男性 11,034 个，女性 15,742 个
- **音频长度**：3秒固定长度

## 依赖项

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- tqdm
- PyQt5（图形界面）
- pyaudio（录音功能）

## 改进方向

1. **特征提取**：使用更丰富的特征（如 MFCC、F0 等）
2. **模型优化**：调整 LSTM 架构和超参数
3. **数据处理**：处理类别不平衡问题
4. **实时处理**：优化实时录音和识别性能

## 许可证

MIT License
