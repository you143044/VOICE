"""
feature_extractor.py - 整合版特征提取模块
=======================================

包含音频加载、预处理和特征提取功能
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from config import Config


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


def extract_features(audio, sample_rate=16000):
    """提取音频特征"""
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
        features.append(np.mean(magnitude) / 10000)
        features.append(np.std(magnitude) / 10000)
        
        # 时域统计特征
        features.append(np.mean(audio))
        features.append(np.std(audio))
        features.append(np.max(audio))
        features.append(np.min(audio))
        
        # 增加到固定维度
        while len(features) < 13:
            features.append(0.0)
        
        return np.array(features)[:13]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(13)


def process_audio_files(dataset_dir, output_dir, target_sr=16000):
    """处理音频文件，统一格式"""
    print("扫描音频文件...")
    audio_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个 WAV 文件")
    print(f"输出目录: {output_dir}")
    
    success = 0
    failed = 0
    failed_list = []
    
    print("\n开始处理...")
    for i, filepath in enumerate(tqdm(audio_files, desc="处理音频")):
        rel_path = os.path.relpath(filepath, dataset_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        try:
            sr, audio = wavfile.read(filepath)
            
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if sr != target_sr:
                from scipy import signal
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)
            
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            audio_int16 = (audio * 32767).astype(np.int16)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, target_sr, audio_int16)
            success += 1
        except Exception as e:
            print(f"  错误: {os.path.basename(filepath)} - {e}")
            failed += 1
            failed_list.append(filepath)
        
        if (i + 1) % 500 == 0:
            print(f"\n进度: {i+1}/{len(audio_files)} | 成功: {success} | 失败: {failed}")
    
    print("\n" + "="*50)
    print(f"处理完成！")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"输出目录: {output_dir}")
    
    return success, failed


def extract_all_features(dataset_dir, config):
    """提取所有音频文件的特征"""
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
        audio = load_audio(audio_path, config.FIXED_LENGTH)
        
        if audio is not None:
            features = extract_features(audio)
            features_list.append(features)
            labels_list.append(config.GENDER_MAPPING[gender])
        else:
            failed_files.append(audio_path)
    
    print(f"\n特征提取完成！")
    print(f"  成功: {len(features_list)}")
    print(f"  失败: {len(failed_files)}")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    save_path = os.path.join(config.FEATURES_PATH, 'features.npy')
    labels_path = os.path.join(config.FEATURES_PATH, 'labels.npy')
    
    np.save(save_path, features_array)
    np.save(labels_path, labels_array)
    
    print(f"\n特征已保存到:")
    print(f"  特征: {save_path}")
    print(f"  标签: {labels_path}")
    print(f"  特征维度: {features_array.shape[1]}")
    print(f"  样本数量: {features_array.shape[0]}")
    
    return features_array, labels_array