"""
extract_features_light.py - 轻量版特征提取脚本
========================================

使用更稳定的方式提取特征
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import pickle

from model import Config


def get_gender_from_filename(filename):
    """从文件名获取性别标签"""
    first_char = filename[0].upper()
    if first_char in ['A', 'C']:
        return 'female'
    elif first_char in ['B', 'D']:
        return 'male'
    else:
        return 'unknown'


def load_audio(audio_path, fixed_length):
    """加载并预处理音频"""
    try:
        sr, audio = wavfile.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        if len(audio) > fixed_length:
            audio = audio[:fixed_length]
        elif len(audio) < fixed_length:
            pad_left = (fixed_length - len(audio)) // 2
            pad_right = fixed_length - len(audio) - pad_left
            audio = np.pad(audio, (pad_left, pad_right), mode='constant')
        
        return audio
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def extract_simple_features(audio, sample_rate=16000):
    """提取简单的音频特征"""
    try:
        features = []
        
        # 能量特征
        energy = np.sum(audio ** 2) / len(audio)
        energy_db = 10 * np.log10(energy + 1e-10)
        features.append(energy_db)
        
        # 过零率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features.append(zero_crossings)
        
        # 频谱特征（使用简单的FFT）
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)[:len(fft)//2]
        features.extend(np.mean(magnitude) / 10000)
        features.extend(np.std(magnitude) / 10000)
        
        # 时域统计特征
        features.extend(np.mean(audio))
        features.extend(np.std(audio))
        features.extend(np.max(audio))
        features.extend(np.min(audio))
        
        # 增加到固定维度
        while len(features) < 13:
            features.append(0.0)
        
        return np.array(features)[:13]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(13)


def main():
    print("="*60)
    print("轻量版特征提取")
    print("="*60)
    
    config = Config()
    dataset_dir = r"D:\A--Learning-D\项目\VOICE\dataset_processed"
    
    audio_files = []
    genders = []
    
    print(f"\n扫描目录: {dataset_dir}")
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                gender = get_gender_from_filename(file)
                if gender != 'unknown':
                    audio_files.append(os.path.join(root, file))
                    genders.append(gender)
    
    print(f"找到 {len(audio_files)} 个有效音频文件")
    print(f"  男性: {len([g for g in genders if g == 'male'])}")
    print(f"  女性: {len([g for g in genders if g == 'female'])}")
    
    features_list = []
    labels_list = []
    failed_files = []
    
    print("\n开始提取特征...")
    for idx, (audio_path, gender) in enumerate(tqdm(zip(audio_files, genders), total=len(audio_files), desc="提取特征")):
        audio = load_audio(audio_path, 16000*3)
        
        if audio is not None:
            features = extract_simple_features(audio)
            features_list.append(features)
            labels_list.append(0 if gender == 'male' else 1)
        else:
            failed_files.append(audio_path)
    
    print(f"\n特征提取完成！")
    print(f"  成功: {len(features_list)}")
    print(f"  失败: {len(failed_files)}")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    save_dir = os.path.join(os.path.dirname(__file__), 'features')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'features.npy')
    labels_path = os.path.join(save_dir, 'labels.npy')
    
    np.save(save_path, features_array)
    np.save(labels_path, labels_array)
    
    print(f"\n特征已保存到:")
    print(f"  特征: {save_path}")
    print(f"  标签: {labels_path}")
    print(f"  特征维度: {features_array.shape[1]}")
    print(f"  样本数量: {features_array.shape[0]}")


if __name__ == "__main__":
    main()
