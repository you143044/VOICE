"""
app.py - 声音性别识别系统前端界面
===================================

包含实时识别、音频文件加载和模型选择功能
"""

import os
import sys
import numpy as np
import torch
import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from config import Config
from model import LSTMGenderDetector
from feature_extractor import extract_features, load_audio


class RecognitionThread(QThread):
    """实时识别线程"""
    update_signal = pyqtSignal(str)
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.running = False
    
    def run(self):
        self.running = True
        
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024
        )
        
        try:
            while self.running:
                # 读取音频数据
                data = stream.read(1024 * 48)  # 3秒音频
                audio = np.frombuffer(data, dtype=np.float32)
                
                # 提取特征
                features = extract_features(audio)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                
                # 模型预测
                with torch.no_grad():
                    output = self.model(features_tensor)
                    prediction = (torch.sigmoid(output) > 0.5).item()
                    gender = '女性' if prediction else '男性'
                
                self.update_signal.emit(f"识别结果: {gender}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def stop(self):
        self.running = False
        self.wait()


class GenderRecognitionApp(QMainWindow):
    """主应用窗口"""
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.current_model = None
        self.recognition_thread = None
        self.init_ui()
        self.load_models()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("声音性别识别系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 中央 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel("声音性别识别系统")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        model_label.setFont(QFont("Arial", 12))
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Arial", 12))
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.setFont(QFont("Arial", 12))
        self.load_model_button.clicked.connect(self.load_selected_model)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_model_button)
        main_layout.addLayout(model_layout)
        
        # 状态显示
        self.status_label = QLabel("请加载模型")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: blue")
        main_layout.addWidget(self.status_label)
        
        # 识别结果
        self.result_label = QLabel("识别结果: 未识别")
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 实时识别按钮
        self.start_button = QPushButton("开始实时识别")
        self.start_button.setFont(QFont("Arial", 12))
        self.start_button.clicked.connect(self.start_recognition)
        
        self.stop_button = QPushButton("停止实时识别")
        self.stop_button.setFont(QFont("Arial", 12))
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)
        
        # 加载音频文件按钮
        self.load_audio_button = QPushButton("加载音频文件")
        self.load_audio_button.setFont(QFont("Arial", 12))
        self.load_audio_button.clicked.connect(self.load_audio_file)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.load_audio_button)
        main_layout.addLayout(button_layout)
        
        # 日志输出
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Courier", 10))
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("日志输出...")
        main_layout.addWidget(self.log_text)
    
    def load_models(self):
        """加载模型列表"""
        models_dir = self.config.MODELS_PATH
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            for model_file in model_files:
                self.model_combo.addItem(model_file)
        else:
            self.log("模型目录不存在")
    
    def load_selected_model(self):
        """加载选中的模型"""
        model_file = self.model_combo.currentText()
        if not model_file:
            self.log("请选择模型")
            return
        
        model_path = os.path.join(self.config.MODELS_PATH, model_file)
        try:
            # 加载模型
            checkpoint = torch.load(model_path)
            model = LSTMGenderDetector(input_dim=13)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.current_model = model
            self.status_label.setText(f"模型已加载: {model_file}")
            self.log(f"成功加载模型: {model_file}")
        except Exception as e:
            self.log(f"加载模型失败: {e}")
            self.status_label.setText("加载模型失败")
    
    def start_recognition(self):
        """开始实时识别"""
        if not self.current_model:
            self.log("请先加载模型")
            return
        
        self.recognition_thread = RecognitionThread(self.current_model, self.config)
        self.recognition_thread.update_signal.connect(self.update_result)
        self.recognition_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log("开始实时识别...")
    
    def stop_recognition(self):
        """停止实时识别"""
        if self.recognition_thread:
            self.recognition_thread.stop()
            self.recognition_thread = None
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.log("停止实时识别")
    
    def load_audio_file(self):
        """加载音频文件并识别"""
        if not self.current_model:
            self.log("请先加载模型")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "WAV files (*.wav)"
        )
        
        if file_path:
            try:
                # 加载音频
                audio = load_audio(file_path, self.config.FIXED_LENGTH)
                if audio is not None:
                    # 提取特征
                    features = extract_features(audio)
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    
                    # 模型预测
                    with torch.no_grad():
                        output = self.current_model(features_tensor)
                        prediction = (torch.sigmoid(output) > 0.5).item()
                        gender = '女性' if prediction else '男性'
                    
                    self.result_label.setText(f"识别结果: {gender}")
                    self.log(f"识别文件: {os.path.basename(file_path)} -> {gender}")
                else:
                    self.log("加载音频文件失败")
            except Exception as e:
                self.log(f"识别失败: {e}")
    
    def update_result(self, result):
        """更新识别结果"""
        self.result_label.setText(result)
    
    def log(self, message):
        """添加日志"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.recognition_thread:
            self.recognition_thread.stop()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = GenderRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()