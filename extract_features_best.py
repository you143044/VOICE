"""
extract_features_best.py - 最佳特征提取脚本
============================================

严格按照推荐表格使用库：
- 基频 F0: parselmouth（Praat内核，秒出）
- 共振峰: parselmouth（一行代码）
- 频谱能量 / 过零率: librosa（最快）
- 谐噪比 HNR: parselmouth（最准）
- 谐波结构: pyworld（专业）
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
import parselmouth
import pyworld as pw
from tqdm import tqdm


def extract_features(audio_path):
    """提取高级音频特征"""
    try:
        sr, audio = wav.read(audio_path)
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float64) / 2147483648.0
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        sound = parselmouth.Sound(audio, sr)
        
        # 1. 基频 F0（使用parselmouth）
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        
        f0_features = []
        if len(f0_values) > 0:
            f0_features = [
                np.mean(f0_values),
                np.std(f0_values),
                np.min(f0_values),
                np.max(f0_values),
                np.median(f0_values),
                np.percentile(f0_values, 25),
                np.percentile(f0_values, 75),
                np.max(f0_values) - np.min(f0_values),
                np.percentile(f0_values, 75) - np.percentile(f0_values, 25),
                len(f0_values) / len(pitch.selected_array['frequency'])
            ]
        else:
            f0_features = [0] * 10
        
        # 2. 共振峰（使用parselmouth）
        formants = sound.to_formant_burg(time_step=0.01, maximum_formant=5500.0)
        f1_list, f2_list, f3_list = [], [], []
        
        num_frames = formants.get_number_of_frames()
        for i in range(num_frames):
            time = formants.get_time_from_frame_number(i + 1)
            f1 = formants.get_value_at_time(1, time)
            f2 = formants.get_value_at_time(2, time)
            f3 = formants.get_value_at_time(3, time)
            if f1: f1_list.append(f1)
            if f2: f2_list.append(f2)
            if f3: f3_list.append(f3)
        
        formant_features = []
        for values in [f1_list, f2_list, f3_list]:
            if len(values) > 0:
                formant_features.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])
            else:
                formant_features.extend([0, 0, 0, 0])
        
        # 3. 频谱能量 / 过零率（使用librosa）
        spectral_features = []
        try:
            mel_spec = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            spectral_features = [
                np.mean(mel_spec_db),
                np.std(mel_spec_db),
                np.min(mel_spec_db),
                np.max(mel_spec_db),
                np.mean(librosa.feature.spectral_centroid(y=audio.astype(np.float32), sr=sr)[0]),
                np.mean(librosa.feature.spectral_bandwidth(y=audio.astype(np.float32), sr=sr)[0]),
                np.mean(librosa.feature.spectral_rolloff(y=audio.astype(np.float32), sr=sr)[0]),
                np.mean(librosa.feature.zero_crossing_rate(y=audio.astype(np.float32))[0])
            ]
        except:
            spectral_features = [0] * 8
        
        # 4. 谐噪比 HNR（使用parselmouth）
        hnr_features = []
        try:
            harmonicity = sound.to_harmonicity_cc(time_step=0.01)
            hnr_values = harmonicity.values.flatten()
            hnr_values = hnr_values[hnr_values > -200]
            if len(hnr_values) > 0:
                hnr_features = [np.mean(hnr_values), np.std(hnr_values)]
            else:
                hnr_features = [0, 0]
        except:
            hnr_features = [0, 0]
        
        # 5. 谐波结构（使用pyworld）
        harmonic_features = []
        try:
            f0_pw, sp, ap = pw.wav2world(audio, sr, frame_period=5.0)
            sp_mean = np.mean(sp, axis=0)
            harmonic_features = [
                np.mean(sp_mean),
                np.std(sp_mean),
                np.max(sp_mean),
                np.min(sp_mean),
                np.mean(f0_pw[f0_pw > 0]) if np.any(f0_pw > 0) else 0
            ]
        except:
            harmonic_features = [0] * 5
        
        # 6. 能量特征
        energy = np.sum(audio**2)
        energy_db = 10 * np.log10(energy + 1e-10)
        energy_features = [energy, energy_db]
        
        # 合并所有特征
        all_features = (
            f0_features +
            formant_features +
            spectral_features +
            hnr_features +
            harmonic_features +
            energy_features
        )
        
        all_features = [0 if np.isnan(f) or np.isinf(f) else f for f in all_features]
        
        return np.array(all_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error: {os.path.basename(audio_path)} - {e}")
        return np.zeros(10 + 12 + 8 + 2 + 5 + 2, dtype=np.float32)


def process_dataset(input_dir, output_dir):
    """处理整个数据集"""
    features = []
    labels = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    for audio_path in tqdm(audio_files, desc="提取特征"):
        feature = extract_features(audio_path)
        label = 1 if 'female' in audio_path.lower() else 0
        features.append(feature)
        labels.append(label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    np.save(os.path.join(output_dir, 'features_best.npy'), features)
    np.save(os.path.join(output_dir, 'labels_best.npy'), labels)
    
    print(f"\n完成！")
    print(f"  样本: {len(features)}")
    print(f"  维度: {features.shape[1]}")
    print(f"  男性: {np.sum(labels == 0)}")
    print(f"  女性: {np.sum(labels == 1)}")


def main():
    print("=" * 60)
    print("最佳特征提取（使用推荐库）")
    print("=" * 60)
    
    input_dir = os.path.join('dataset_processed')
    output_dir = os.path.join('features')
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    print(f"\n输入: {input_dir}")
    print(f"输出: {output_dir}")
    
    process_dataset(input_dir, output_dir)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
