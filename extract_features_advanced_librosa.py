"""
extract_features_advanced_librosa.py - 高级特征提取脚本（使用librosa）
============================================================

提取基频 F0、共振峰、频谱能量分布、谐波结构、谐噪比、过零率等特征
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
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
        
        # 1. 基频 F0 特征（使用librosa的yin算法）
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        f0_values = f0[voiced_flag]
        
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
                np.mean(voiced_flag),  # 浊音比例
                np.std(f0_values) / np.mean(f0_values) if np.mean(f0_values) > 0 else 0,  # 变异系数
                np.mean((f0_values - np.mean(f0_values))**3) / (np.std(f0_values)**3) if np.std(f0_values) > 0 else 0,  # 偏度
                np.mean((f0_values - np.mean(f0_values))**4) / (np.std(f0_values)**4) if np.std(f0_values) > 0 else 0   # 峰度
            ]
        else:
            f0_features = [0] * 13
        
        # 2. 共振峰特征（使用LPC）
        formant_features = []
        try:
            # 计算LPC系数
            lpc_order = 12
            lpc_coeffs = librosa.lpc(audio, order=lpc_order)
            
            # 找到极点
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) > 0]
            
            # 计算共振峰频率
            formant_freqs = np.sort(np.angle(roots) * (sr / (2 * np.pi)))
            
            # 提取前3个共振峰
            for i in range(3):
                if i < len(formant_freqs):
                    formant_features.extend([formant_freqs[i]])
                else:
                    formant_features.extend([0])
        except:
            formant_features = [0] * 3
        
        # 3. 频谱能量分布
        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        spectral_features = []
        if mel_spec_db.size > 0:
            spectral_features = [
                np.mean(mel_spec_db),
                np.std(mel_spec_db),
                np.min(mel_spec_db),
                np.max(mel_spec_db),
                # 频谱质心
                np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0]),
                # 频谱带宽
                np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]),
                # 频谱平坦度
                np.mean(librosa.feature.spectral_flatness(y=audio)[0])
            ]
        else:
            spectral_features = [0] * 7
        
        # 4. 谐波结构（使用谐波与噪声比）
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_features = [
            np.mean(harmonic),
            np.std(harmonic),
            np.max(harmonic),
            np.min(harmonic)
        ]
        
        # 5. 谐噪比 (HNR)
        hnr = np.mean(librosa.feature.spectral_flatness(y=harmonic)[0]) / np.mean(librosa.feature.spectral_flatness(y=percussive)[0]) if np.mean(librosa.feature.spectral_flatness(y=percussive)[0]) > 0 else 0
        hnr_features = [hnr]
        
        # 6. 过零率
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        zcr_features = [zcr]
        
        # 7. 能量特征
        energy = np.sum(audio**2)
        energy_db = 10 * np.log10(energy + 1e-10)
        energy_features = [energy, energy_db]
        
        # 8. MFCC特征（额外添加，提高识别能力）
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_features = []
        if mfcc.size > 0:
            mfcc_features = np.mean(mfcc, axis=1).tolist()
        else:
            mfcc_features = [0] * 13
        
        # 合并所有特征
        all_features = (
            f0_features +
            formant_features +
            spectral_features +
            harmonic_features +
            hnr_features +
            zcr_features +
            energy_features +
            mfcc_features
        )
        
        # 处理可能的NaN值
        all_features = [0 if np.isnan(f) else f for f in all_features]
        
        return np.array(all_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # 返回全零特征
        return np.zeros(13 + 3 + 7 + 4 + 1 + 1 + 2 + 13, dtype=np.float32)


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
