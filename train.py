"""
train.py - 整合版训练脚本
=====================

使用LSTM模型进行训练和评估
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

from config import Config
from model import LSTMGenderDetector


class FeatureDataset(torch.utils.data.Dataset):
    """预提取特征数据集"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


class Trainer:
    """训练器类"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.float().unsqueeze(1)
            
            self.optimizer.zero_grad()
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
    
    def train(self, dataloaders, epochs=30):
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
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
                best_path = os.path.join('models', 'lstm_gender_detector_best.pth')
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch + 1,
                    'val_acc': best_val_acc,
                    'history': self.history
                }, best_path)
                print(f"  -> 保存最佳模型 (val_acc={best_val_acc:.4f})")
            
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time:.1f}s")
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
            path = os.path.join('models', 'lstm_gender_detector_final.pth')
        
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
        return path


def main():
    print("=" * 60)
    print("声音性别识别 - LSTM 模型训练")
    print("=" * 60)
    
    config = Config()
    features_path = os.path.join('features', 'features.npy')
    labels_path = os.path.join('features', 'labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"\n错误: 特征文件不存在！")
        print(f"  特征文件: {features_path}")
        print(f"  标签文件: {labels_path}")
        print("请先运行 feature_extractor.py 提取特征！")
        return
    
    print(f"\n加载特征文件...")
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"  特征维度: {features.shape[1]}")
    print(f"  样本数量: {features.shape[0]}")
    print(f"  男性样本: {np.sum(labels == 0)}")
    print(f"  女性样本: {np.sum(labels == 1)}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    datasets = {
        'train': FeatureDataset(X_train, y_train),
        'val': FeatureDataset(X_val, y_val),
        'test': FeatureDataset(X_test, y_test)
    }
    
    print(f"\n数据集划分:")
    print(f"  Train: {len(datasets['train'])}")
    print(f"  Val:   {len(datasets['val'])}")
    print(f"  Test:  {len(datasets['test'])}")
    
    dataloaders = {}
    for split, dataset in datasets.items():
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=0
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    input_dim = features.shape[1]
    model = LSTMGenderDetector(input_dim=input_dim).to(device)
    
    trainer = Trainer(model, device)
    try:
        trainer.train(dataloaders, epochs=200)
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
        trainer.save_model()
        best_path = os.path.join('models', 'lstm_gender_detector_best.pth')
        if os.path.exists(best_path):
            print(f"最佳模型已保存到: {best_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()