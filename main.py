"""
main.py - 主脚本
=================

整合预处理、特征提取和训练功能
"""

import os
from config import Config
from feature_extractor import process_audio_files, extract_all_features


def main():
    print("=" * 60)
    print("声音性别识别系统 - 主脚本")
    print("=" * 60)
    
    config = Config()
    
    print("\n1. 音频预处理")
    print("-" * 40)
    if os.path.exists(config.DATASET_PATH):
        success, failed = process_audio_files(
            config.DATASET_PATH,
            config.DATASET_PROCESSED_PATH,
            config.SAMPLE_RATE
        )
        if success > 0:
            print("\n2. 特征提取")
            print("-" * 40)
            extract_all_features(config.DATASET_PROCESSED_PATH, config)
            
            print("\n3. 模型训练")
            print("-" * 40)
            print("请运行: python train.py")
        else:
            print("\n错误: 没有成功处理任何音频文件！")
    else:
        print(f"错误: 数据集目录不存在: {config.DATASET_PATH}")
        print("请先创建数据集目录并添加音频文件。")
    
    print("\n" + "=" * 60)
    print("操作完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()