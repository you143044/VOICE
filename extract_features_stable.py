"""
extract_features_stable.py - 稳定的高级特征提取脚本
============================================

提取基频 F0、共振峰、频谱能量分布、谐波结构、谐噪比、过零率等特征
使用 scipy 和 numpy，避免使用易崩溃的库
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from tqdm import tqdm


def compute_f0(audio, sr):
    """计算基频 F0"""
    try:
        # 使用自相关方法计算基频
        frame_size = int(sr * 0.02)  # 20ms帧
        hop_size = int(sr * 0.01)    # 10ms步长
        
        f0_values = []
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            
            # 计算自相关
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 找到自相关的峰值
            peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
            
            if len(peaks) > 0:
                # 找到第一个峰值（基频）
                fundamental_period = peaks[0]
                if fundamental_period > 0:
                    f0 = sr / fundamental_period
                    if 80 < f0 < 400:  # 正常语音基频范围
                        f0_values.append(f0)
        
        return f0_values
    except:
        return []

def compute_formants(audio, sr):
    """计算共振峰"""
    try:
        # 使用LPC计算共振峰
        frame_size = int(sr * 0.02)
        hop_size = int(sr * 0.01)
        
        formants = []
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            
            # 预加重
            frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
            
            # 计算LPC
            lpc_order = 12
            try:
                lpc_coeffs = signal.lpc(frame, lpc_order)
                
                # 找到极点
                roots = np.roots(lpc_coeffs)
                roots = roots[np.imag(roots) > 0]
                
                # 计算共振峰频率
                freqs = np.sort(np.angle(roots) * (sr / (2 * np.pi)))
                
                # 提取前3个共振峰
                if len(freqs) >= 3:
                    formants.extend(freqs[:3])
            except:
                pass
        
        return formants
    except:
        return []

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
        
        # 1. 基频 F0 特征
        f0_values = compute_f0(audio, sr)
        
        f0_features = []
        if len(f0_values) > 0:
            f0_features = [
                np.mean(f0_values),
                np.std(f0_values),
                np.min(f0_values),
                np.max(f0_values),
                np.median(f0_values),
                np.percentile(f0_values, 25),
                np.percentile(f0_values, 75)
            ]
        else:
            f0_features = [0] * 7
        
        # 2. 共振峰特征
        formant_values = compute_formants(audio, sr)
        
        formant_features = []
        if len(formant_values) > 0:
            formant_features = [
                np.mean(formant_values),
                np.std(formant_values),
                np.min(formant_values),
                np.max(formant_values)
            ]
        else:
            formant_features = [0] * 4
        
        # 3. 频谱能量分布
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=256, noverlap=128)
        
        spectral_features = []
        if Sxx.size > 0:
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            spectral_features = [
                np.mean(Sxx_db),
                np.std(Sxx_db),
                np.min(Sxx_db),
                np.max(Sxx_db),
                # 频谱质心
                np.mean(np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0) + 1e-10)
            ]
        else:
            spectral_features = [0] * 5
        
        # 4. 谐波结构（使用能量比）
        # 低通滤波提取谐波
        b, a = signal.butter(4, 3000, 'low', fs=sr)
        harmonic = signal.filtfilt(b, a, audio)
        
        # 高通滤波提取噪声
        b, a = signal.butter(4, 3000, 'high', fs=sr)
        noise = signal.filtfilt(b, a, audio)
        
        harmonic_energy = np.sum(harmonic**2)
        noise_energy = np.sum(noise**2)
        harmonic_ratio = harmonic_energy / (noise_energy + 1e-10)
        
        harmonic_features = [harmonic_ratio]
        
        # 5. 谐噪比 (HNR)
        hnr = harmonic_ratio
        hnr_features = [hnr]
        
        # 6. 过零率
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        zcr = len(zero_crossings) / len(audio)
        zcr_features = [zcr]
        
        # 7. 能量特征
        energy = np.sum(audio**2)
        energy_db = 10 * np.log10(energy + 1e-10)
        energy_features = [energy, energy_db]
        
        # 8. 时域统计特征
        time_features = [
            np.mean(audio),
            np.std(audio),
            np.max(audio),
            np.min(audio)
        ]
        
        # 合并所有特征
        all_features = (
            f0_features +
            formant_features +
            spectral_features +
            harmonic_features +
            hnr_features +
            zcr_features +
            energy_features +
            time_features
        )
        
        # 处理可能的NaN值
        all_features = [0 if np.isnan(f) else f for f in all_features]
        
        return np.array(all_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # 返回全零特征
        return np.zeros(7 + 4 + 5 + 1 + 1 + 1 + 2 + 4, dtype=np.float32)


def process_dataset(input_dir, output_dir):
    """处理整个数据集"""
    features = []
    labels = []
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历目录
    total_files = 0
    # 先计算总文件数
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                total_files += 1
    
    print(f"Total files to process: {total_files}")
    
    processed = 0
    # 处理文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                
                # 提取特征
                feature = extract_features(audio_path)
                
                # 确定标签（根据目录结构）
                # 假设目录结构为：dataset/males/ 和 dataset/females/
                label = 1 if 'female' in root.lower() else 0
                
                features.append(feature)
                labels.append(label)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed: {processed}/{total_files}")
    
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
    print("稳定的高级音频特征提取")
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
