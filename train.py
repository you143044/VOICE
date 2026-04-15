"""
train.py - 训练模块
===================

包含：
- 数据加载和预处理
- 训练和验证
- 模型保存
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

from model import Config, GenderDetectorModel, AdvancedFeatureExtractor

class AudioDataset(torch.utils.data.Dataset):
    """音频数据集类 - 使用手工特征"""
    
    def __init__(self, audio_paths, genders, config):
        self.audio_paths = audio_paths
        self.genders = genders
        self.config = config
        self.feature_extractor = AdvancedFeatureExtractor(
            sample_rate=config.SAMPLE_RATE,
            n_mfcc=config.MFCC_DIM
        )
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        gender = self.genders[idx]
        
        if idx % 10 == 0:
            print(f"Processing file {idx}/{len(self.audio_paths)}: {audio_path}")
        
        try:
            audio = self._load_audio(audio_path)
            if idx % 10 == 0:
                print(f"Audio loaded successfully, length: {len(audio)}")
            
            features = self.feature_extractor.extract_all_features(audio)
            if idx % 10 == 0:
                print(f"Features extracted successfully, shape: {features.shape}")
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            gender_label = torch.tensor(self.config.GENDER_MAPPING[gender], dtype=torch.long)
            
            return features_tensor, gender_label
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            # 返回零特征和默认标签
            return torch.zeros(self.config.TOTAL_FEATURE_DIM), torch.tensor(0, dtype=torch.long)
    
    def _load_audio(self, audio_path):
        try:
            # 使用scipy.io.wavfile加载WAV文件
            sr, audio = wavfile.read(audio_path)
            
            # 转换为float32
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # 归一化
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # 确保音频长度
            if len(audio) > self.config.FIXED_LENGTH:
                audio = audio[:self.config.FIXED_LENGTH]
            elif len(audio) < self.config.FIXED_LENGTH:
                pad_left = (self.config.FIXED_LENGTH - len(audio)) // 2
                pad_right = self.config.FIXED_LENGTH - len(audio) - pad_left
                audio = np.pad(audio, (pad_left, pad_right), mode='constant')
            
            return audio
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(self.config.FIXED_LENGTH)

class DataManager:
    """数据管理类"""
    
    def __init__(self, config):
        self.config = config
    
    def load_dataset(self):
        audio_paths = []
        genders = []
        
        males_path = os.path.join(self.config.DATASET_PATH, 'males')
        females_path = os.path.join(self.config.DATASET_PATH, 'females')
        
        if os.path.exists(males_path):
            for filename in os.listdir(males_path):
                if filename.endswith('.wav'):
                    audio_paths.append(os.path.join(males_path, filename))
                    genders.append('male')
        
        if os.path.exists(females_path):
            for filename in os.listdir(females_path):
                if filename.endswith('.wav'):
                    audio_paths.append(os.path.join(females_path, filename))
                    genders.append('female')
        
        print(f"Loaded {len(audio_paths)} audio files")
        print(f"  Male: {len([g for g in genders if g == 'male'])}")
        print(f"  Female: {len([g for g in genders if g == 'female'])}")
        
        return audio_paths, genders
    
    def create_datasets(self, audio_paths, genders):
        X = np.array(audio_paths)
        y = np.array(genders)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1 - self.config.TRAIN_SPLIT, random_state=42, stratify=y
        )
        
        val_size = self.config.VALIDATION_SPLIT / (self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        datasets = {
            'train': AudioDataset(X_train, y_train, self.config),
            'val': AudioDataset(X_val, y_val, self.config),
            'test': AudioDataset(X_test, y_test, self.config)
        }
        
        print(f"\nDatasets created:")
        print(f"  Train: {len(datasets['train'])}")
        print(f"  Val: {len(datasets['val'])}")
        print(f"  Test: {len(datasets['test'])}")
        
        return datasets
    
    def create_dataloaders(self, datasets):
        dataloaders = {}
        for split, dataset in datasets.items():
            dataloaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=(split == 'train'),
                num_workers=4,
                pin_memory=True
            )
        return dataloaders

class Trainer:
    """训练器类"""
    
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        print(f"开始训练一个epoch，数据加载器长度: {len(dataloader)}")
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            if batch_idx % 10 == 0:
                print(f"处理批次 {batch_idx}/{len(dataloader)}, 输入形状: {inputs.shape}, 标签形状: {labels.shape}")
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            if self.config.USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.float().unsqueeze(1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def train(self, dataloaders):
        print(f"\nStarting training for {self.config.EPOCHS} epochs...")
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc = self.train_one_epoch(dataloaders['train'])
            val_loss, val_acc = self.validate(dataloaders['val'])
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                best_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.config.MODEL_NAME}_best.pth')
                os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch + 1,
                    'val_acc': best_val_acc,
                    'history': self.history
                }, best_path)
                print(f"  -> 保存最佳模型 (val_acc={best_val_acc:.4f})")
            
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS} - {epoch_time:.1f}s")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, dataloader):
        print("\nEvaluating model...")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
                
                all_preds.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Male', 'Female']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.config.MODEL_NAME}.pth')
        
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
        return path

def main():
    print("=" * 60)
    print("声音性别识别 - 训练模式")
    print("=" * 60)
    
    config = Config()
    
    data_manager = DataManager(config)
    audio_paths, genders = data_manager.load_dataset()
    datasets = data_manager.create_datasets(audio_paths, genders)
    dataloaders = data_manager.create_dataloaders(datasets)
    
    device = torch.device('cuda' if config.USE_GPU and torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = GenderDetectorModel(config).to(device)
    
    trainer = Trainer(config, model, device)
    try:
        trainer.train(dataloaders)
        try:
            trainer.evaluate(dataloaders['test'])
        except Exception as e:
            print(f"\n评估过程出错: {e}")
            print("但模型训练已完成，继续保存模型...")
    except Exception as e:
        print(f"\n训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        final_path = os.path.join(config.MODEL_SAVE_PATH, f'{config.MODEL_NAME}_final.pth')
        trainer.save_model(final_path)
        print(f"\n最终模型已保存到: {final_path}")
        
        best_path = os.path.join(config.MODEL_SAVE_PATH, f'{config.MODEL_NAME}_best.pth')
        if os.path.exists(best_path):
            print(f"最佳模型已保存到: {best_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
