"""
extract_features_final.py - 最终稳定版特征提取
============================================

只使用：
- parselmouth: F0, 共振峰, HNR
- scipy/numpy: 频谱能量, 过零率
跳过 pyworld 避免崩溃
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import parselmouth
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
        
        # 3. 频谱能量和过零率（使用scipy/numpy）
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=512, noverlap=256)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        spectral_features = [
            np.mean(Sxx_db),
            np.std(Sxx_db),
            np.min(Sxx_db),
            np.max(Sxx_db),
            np.mean(np.sum(f[:, np.newaxis] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)),
            np.mean(np.sqrt(np.sum((f[:, np.newaxis] - np.mean(f))**2 * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)))
        ]
        
        # 过零率
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        zcr = len(zero_crossings) / len(audio)
        spectral_features.append(zcr)
        
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
        
        # 5. 能量特征
        energy = np.sum(audio**2)
        energy_db = 10 * np.log10(energy + 1e-10)
        energy_features = [energy, energy_db]
        
        # 6. 时域统计特征
        time_features = [np.mean(audio), np.std(audio), np.max(audio), np.min(audio)]
        
        # 合并所有特征
        all_features = (
            f0_features +
            formant_features +
            spectral_features +
            hnr_features +
            energy_features +
            time_features
        )
        
        all_features = [0 if np.isnan(f) or np.isinf(f) else f for f in all_features]
        
        return np.array(all_features, dtype=np.float32)
        
    except Exception as e:
        return np.zeros(10 + 12 + 7 + 2 + 2 + 4, dtype=np.float32)


def get_gender_from_filename(filename):
    """从文件名获取性别标签"""
    first_char = filename[0].upper()
    if first_char in ['A', 'C']:
        return 1  # 女性
    elif first_char in ['B', 'D']:
        return 0  # 男性
    else:
        return None


def process_dataset(input_dir, output_dir):
    """处理整个数据集"""
    features = []
    labels = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                gender = get_gender_from_filename(file)
                if gender is not None:
                    audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个有效音频文件")
    
    for i, audio_path in enumerate(tqdm(audio_files, desc="提取特征")):
        feature = extract_features(audio_path)
        filename = os.path.basename(audio_path)
        label = get_gender_from_filename(filename)
        features.append(feature)
        labels.append(label)
        
        if (i + 1) % 1000 == 0:
            temp_features = np.array(features)
            temp_labels = np.array(labels)
            np.save(os.path.join(output_dir, 'features_temp.npy'), temp_features)
            np.save(os.path.join(output_dir, 'labels_temp.npy'), temp_labels)
            print(f"  已保存临时文件 (第{i+1}个)")
    
    features = np.array(features)
    labels = np.array(labels)
    
    np.save(os.path.join(output_dir, 'features_final.npy'), features)
    np.save(os.path.join(output_dir, 'labels_final.npy'), labels)
    
    print(f"\n完成！")
    print(f"  样本: {len(features)}")
    print(f"  维度: {features.shape[1]}")
    print(f"  男性: {np.sum(labels == 0)}")
    print(f"  女性: {np.sum(labels == 1)}")


def main():
    print("=" * 60)
    print("最终稳定版特征提取")
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
