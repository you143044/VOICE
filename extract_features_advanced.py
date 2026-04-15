"""
extract_features_advanced.py - 高级特征提取脚本
============================================

提取基频 F0、共振峰、频谱能量分布、谐波结构、谐噪比、过零率等特征
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import parselmouth
import pyworld as pw
from tqdm import tqdm


def extract_features(audio_path):
    """提取高级音频特征"""
    try:
        # 读取音频文件
        sr, audio = wav.read(audio_path)
        
        # 转换为 float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # 确保音频是单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 使用 parselmouth 提取 F0 和共振峰
        sound = parselmouth.Sound(audio, sr)
        
        # 1. 基频 F0 特征
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # 只保留有效F0
        
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
                len(f0_values) / len(pitch.selected_array['frequency']),  # 浊音比例
                np.std(f0_values) / np.mean(f0_values) if np.mean(f0_values) > 0 else 0,  # 变异系数
                np.mean((f0_values - np.mean(f0_values))**3) / (np.std(f0_values)**3) if np.std(f0_values) > 0 else 0,  # 偏度
                np.mean((f0_values - np.mean(f0_values))**4) / (np.std(f0_values)**4) if np.std(f0_values) > 0 else 0   # 峰度
            ]
        else:
            f0_features = [0] * 13
        
        # 2. 共振峰特征
        formants = sound.to_formant_burg()
        f1_values = []
        f2_values = []
        f3_values = []
        
        for t in range(0, sound.n_samples, 1000):
            f1 = formants.get_value_at_time(1, t / sound.sampling_frequency)
            f2 = formants.get_value_at_time(2, t / sound.sampling_frequency)
            f3 = formants.get_value_at_time(3, t / sound.sampling_frequency)
            if f1 > 0:
                f1_values.append(f1)
            if f2 > 0:
                f2_values.append(f2)
            if f3 > 0:
                f3_values.append(f3)
        
        formant_features = []
        for values in [f1_values, f2_values, f3_values]:
            if len(values) > 0:
                formant_features.extend([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values)
                ])
            else:
                formant_features.extend([0, 0, 0, 0])
        
        # 3. 频谱能量分布
        import scipy.signal as signal
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=256, noverlap=128)
        
        # 频谱能量特征
        spectral_features = []
        if Sxx.size > 0:
            # 频谱能量的统计特征
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            spectral_features = [
                np.mean(Sxx_db),
                np.std(Sxx_db),
                np.min(Sxx_db),
                np.max(Sxx_db),
                # 频谱质心
                np.mean(np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0) + 1e-10),
                # 频谱带宽
                np.mean(np.sqrt(np.sum((f[:, np.newaxis] - np.mean(f))**2 * Sxx, axis=0) / np.sum(Sxx, axis=0) + 1e-10))
            ]
        else:
            spectral_features = [0] * 6
        
        # 4. 谐波结构和谐噪比 (使用 pyworld)
        f0, sp, ap = pw.wav2world(audio.astype(np.float64), sr, frame_period=5.0)
        
        # 谐波结构特征
        harmonic_features = []
        if sp.shape[0] > 0:
            # 频谱包络的统计特征
            sp_mean = np.mean(sp, axis=0)
            harmonic_features = [
                np.mean(sp_mean),
                np.std(sp_mean),
                np.max(sp_mean),
                np.min(sp_mean)
            ]
        else:
            harmonic_features = [0] * 4
        
        # 5. 谐噪比 HNR
        hnr_values = []
        for i in range(len(f0)):
            if f0[i] > 0:
                try:
                    hnr = parselmouth.praat.call(
                        parselmouth.Sound(audio[i*int(sr*0.005):(i+1)*int(sr*0.005)], sr),
                        "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
                    )
                    hnr_val = hnr.get_value()
                    if hnr_val > 0:
                        hnr_values.append(hnr_val)
                except:
                    pass
        
        hnr_features = [np.mean(hnr_values) if hnr_values else 0]
        
        # 6. 过零率
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        zcr = len(zero_crossings) / len(audio)
        zcr_features = [zcr]
        
        # 7. 能量特征
        energy = np.sum(audio**2)
        energy_db = 10 * np.log10(energy + 1e-10)
        energy_features = [energy, energy_db]
        
        # 合并所有特征
        all_features = (
            f0_features +
            formant_features +
            spectral_features +
            harmonic_features +
            hnr_features +
            zcr_features +
            energy_features
        )
        
        # 处理可能的NaN值
        all_features = [0 if np.isnan(f) else f for f in all_features]
        
        return np.array(all_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # 返回全零特征
        return np.zeros(13 + 12 + 6 + 4 + 1 + 1 + 2, dtype=np.float32)


def process_dataset(input_dir, output_dir):
    """处理整个数据集"""
    features = []
    labels = []
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历目录
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                
                # 提取特征
                feature = extract_features(audio_path)
                
                # 确定标签（根据目录结构）
                # 假设目录结构为：dataset/males/ 和 dataset/females/
                label = 1 if 'female' in root.lower() else 0
                
                features.append(feature)
                labels.append(label)
    
    # 保存特征和标签
    features = np.array(features)
    labels = np.array(labels)
    
    np.save(os.path.join(output_dir, 'features_advanced.npy'), features)
    np.save(os.path.join(output_dir, 'labels_advanced.npy'), labels)
    
    print(f"\n特征提取完成！")
    print(f"  成功: {len(features)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  样本数量: {features.shape[0]}")
    print(f"  男性样本: {np.sum(labels == 0)}")
    print(f"  女性样本: {np.sum(labels == 1)}")
    print(f"\n特征已保存到:")
    print(f"  特征: {os.path.join(output_dir, 'features_advanced.npy')}")
    print(f"  标签: {os.path.join(output_dir, 'labels_advanced.npy')}")


def main():
    print("=" * 60)
    print("高级音频特征提取")
    print("=" * 60)
    
    input_dir = os.path.join('dataset_processed')
    output_dir = os.path.join('features')
    
    if not os.path.exists(input_dir):
        print(f"\n错误: 输入目录不存在！")
        print(f"  输入目录: {input_dir}")
        print("请先运行 preprocess_final.py 处理音频文件！")
        return
    
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    process_dataset(input_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("特征提取完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
