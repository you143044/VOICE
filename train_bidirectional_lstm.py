"""
train_bidirectional_lstm.py - 双向LSTM模型训练脚本
============================================

使用双向LSTM进行性别识别训练
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class BidirectionalLSTM(nn.Module):
    """双向LSTM性别识别模型（优化版）"""
    
    def __init__(self, input_dim, hidden_dim=512, num_layers=5, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, features):
        features = features.unsqueeze(1)
        out, _ = self.lstm(features)
        out = out[:, -1, :]
        output = self.fc(out)
        return output


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
    
    def __init__(self, model, device, class_weight=None, patience=15):
        self.model = model
        self.device = device
        self.patience = patience
        if class_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight], device=device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
    
    def train(self, dataloaders, epochs=50):
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_acc = 0.0
        best_model_state = None
        early_stopping_patience = self.patience
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_one_epoch(dataloaders['train'])
            val_loss, val_acc = self.validate(dataloaders['val'])
            
            self.scheduler.step(epoch)
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                best_path = os.path.join('models', 'bidirectional_lstm_best.pth')
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch + 1,
                    'val_acc': best_val_acc,
                    'history': self.history
                }, best_path)
                print(f"  -> 保存最佳模型 (val_acc={best_val_acc:.4f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time:.1f}s - LR: {lr:.6f}")
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
            path = os.path.join('models', 'bidirectional_lstm_final.pth')
        
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
        return path


def main():
    print("=" * 60)
    print("双向LSTM模型训练")
    print("=" * 60)
    
    features_path = os.path.join('features', 'features_final.npy')
    labels_path = os.path.join('features', 'labels_final.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"\n错误: 特征文件不存在！")
        print(f"  特征文件: {features_path}")
        print(f"  标签文件: {labels_path}")
        print("请先运行 extract_features_final.py 提取特征！")
        return
    
    print(f"\n加载特征文件...")
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"  特征维度: {features.shape[1]}")
    print(f"  样本数量: {features.shape[0]}")
    print(f"  男性样本: {np.sum(labels == 0)}")
    print(f"  女性样本: {np.sum(labels == 1)}")
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
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
            batch_size=128,
            shuffle=(split == 'train'),
            num_workers=0
        )
    
    # 尝试使用最佳可用设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    input_dim = features.shape[1]
    model = BidirectionalLSTM(input_dim=input_dim, hidden_dim=256, num_layers=3).to(device)
    
    num_males = np.sum(labels == 0)
    num_females = np.sum(labels == 1)
    class_weight = num_males / num_females if num_females > num_males else num_females / num_males
    print(f"\nClass weight: {class_weight:.4f}")
    
    # 不使用早停，训练1000轮
    trainer = Trainer(model, device, class_weight=class_weight, patience=1000)
    try:
        trainer.train(dataloaders, epochs=1000)
        try:
            trainer.evaluate(dataloaders['test'])
        except Exception as e:
            print(f"\n评估过程出错: {e}")
    except Exception as e:
        print(f"\n训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.save_model()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
