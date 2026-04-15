"""
preprocess_final.py - 最终版音频格式统一脚本
===========================================

处理后保存到新目录，避免权限问题
"""

import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wavfile
from scipy import signal


def process_single_file(input_path, output_path, target_sr=16000):
    """处理单个音频文件"""
    try:
        sr, audio = wavfile.read(input_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        audio_int16 = (audio * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wavfile.write(output_path, target_sr, audio_int16)
        return True
    except Exception as e:
        print(f"  错误: {os.path.basename(input_path)} - {e}")
        return False


def main():
    dataset_dir = r"D:\A--Learning-D\项目\VOICE\dataset"
    output_dir = r"D:\A--Learning-D\项目\VOICE\dataset_processed"
    
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
    for i, filepath in enumerate(tqdm(audio_files, desc="处理")):
        rel_path = os.path.relpath(filepath, dataset_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        if process_single_file(filepath, output_path):
            success += 1
        else:
            failed += 1
            failed_list.append(filepath)
        
        if (i + 1) % 500 == 0:
            print(f"\n进度: {i+1}/{len(audio_files)} | 成功: {success} | 失败: {failed}")
    
    print("\n" + "="*50)
    print(f"处理完成！")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"输出目录: {output_dir}")
    
    if failed_list:
        print(f"\n失败文件（前20个）:")
        for f in failed_list[:20]:
            print(f"  - {os.path.basename(f)}")


if __name__ == "__main__":
    main()
