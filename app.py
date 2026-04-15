"""
app.py - 应用程序模块
======================

包含：
- 音频处理
- 用户界面
- 实时识别
$env:QT_QPA_PLATFORM_PLUGIN_PATH="$PWD/VOICE/Lib/site-packages/PyQt5/Qt5/plugins/platforms"
python.exe app.py
"""

import sys
import os
import time
import numpy as np
import torch
import pyaudio
from scipy.io import wavfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QStatusBar, QMenuBar, QAction, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPainter, QPen, QColor

from model import Config, GenderDetectorModel, AdvancedFeatureExtractor

class AudioProcessor:
    """音频处理器类"""
    
    def __init__(self, config):
        self.config = config
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.feature_extractor = AdvancedFeatureExtractor(
            sample_rate=config.SAMPLE_RATE,
            n_mfcc=config.MFCC_DIM
        )
    
    def predict(self, audio, model, device):
        """预测音频的性别"""
        try:
            # 自定义归一化函数，替代librosa.util.normalize
            def normalize(audio):
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    return audio / max_val
                return audio
            
            audio = normalize(audio)
            if len(audio) > self.config.FIXED_LENGTH:
                audio = audio[:self.config.FIXED_LENGTH]
            elif len(audio) < self.config.FIXED_LENGTH:
                pad_left = (self.config.FIXED_LENGTH - len(audio)) // 2
                pad_right = self.config.FIXED_LENGTH - len(audio) - pad_left
                audio = np.pad(audio, (pad_left, pad_right), mode='constant')
            
            features = self.feature_extractor.extract_all_features(audio)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(features_tensor)
                probability = torch.sigmoid(output).item()
            
            gender = 'female' if probability > 0.5 else 'male'
            confidence = probability if probability > 0.5 else 1 - probability
            
            return gender, confidence
        except Exception as e:
            print(f"预测错误: {e}")
            import traceback
            traceback.print_exc()
            return 'male', 0.0
    
    def __del__(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

class AudioThread(QThread):
    """音频处理线程"""
    update_signal = pyqtSignal(str, float)
    error_signal = pyqtSignal(str)
    audio_data_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, config, processor, model, device):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.device = device
        self.is_running = False
        self.audio_buffer = []
        self.stream = None
    
    def run(self):
        self.is_running = True
        self.audio_buffer = []
        
        try:
            def audio_callback(in_data, frame_count, time_info, status):
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.audio_buffer.append(audio_data)
                
                self.audio_data_signal.emit(audio_data)
                
                if len(self.audio_buffer) >= int(self.config.SAMPLE_RATE / 1024 * self.config.DURATION):
                    full_audio = np.concatenate(self.audio_buffer[:int(self.config.SAMPLE_RATE / 1024 * self.config.DURATION)])
                    self.audio_buffer = self.audio_buffer[int(self.config.SAMPLE_RATE / 1024 * self.config.DURATION):]
                    
                    gender, confidence = self.processor.predict(full_audio, self.model, self.device)
                    self.update_signal.emit(gender, confidence)
                
                return (in_data, pyaudio.paContinue)
            
            self.stream = self.processor.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024,
                stream_callback=audio_callback
            )
            
            self.stream.start_stream()
            
            while self.is_running and self.stream.is_active():
                time.sleep(0.1)
        
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except:
                    pass
                try:
                    self.stream.close()
                except:
                    pass
            self.stream = None
    
    def stop(self):
        self.is_running = False
        self.wait()

class WaveformWidget(QFrame):
    """声波图显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumHeight(100)
        self.setMaximumHeight(150)
        self.setStyleSheet('background-color: #f0f0f0; border: 1px solid #ccc;')
        
        self.audio_data = np.zeros(1024)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)
    
    def set_audio_data(self, data):
        self.audio_data = np.roll(self.audio_data, -len(data))
        self.audio_data[-len(data):] = data
    
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        center_y = height // 2
        
        painter.fillRect(rect, QColor('#f0f0f0'))
        
        painter.setPen(QPen(QColor('#2196F3'), 2))
        
        if len(self.audio_data) > 0:
            step = width / len(self.audio_data)
            points = []
            
            for i, value in enumerate(self.audio_data):
                x = int(i * step)
                y = int(center_y - value * center_y * 0.8)
                points.append((x, y))
            
            if points:
                from PyQt5.QtCore import QPoint
                qpoints = [QPoint(x, y) for x, y in points]
                painter.drawPolyline(*qpoints)
        
        painter.setPen(QPen(QColor('#ccc'), 1))
        painter.drawLine(0, center_y, width, center_y)

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.device = None
        self.audio_processor = None
        self.audio_thread = None
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        self.setWindowTitle('声音性别识别')
        self.setGeometry(100, 100, 600, 450)
        
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('文件')
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tool_menu = menubar.addMenu('工具')
        settings_action = QAction('设置与分析', self)
        settings_action.triggered.connect(self.open_settings)
        tool_menu.addAction(settings_action)
        
        help_menu = menubar.addMenu('帮助')
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        title_label = QLabel('声音性别识别')
        title_label.setFont(QFont('Arial', 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.status_label = QLabel('准备就绪')
        self.status_label.setFont(QFont('Arial', 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.waveform_widget = WaveformWidget()
        layout.addWidget(self.waveform_widget)
        
        self.result_label = QLabel('等待识别...')
        self.result_label.setFont(QFont('Arial', 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet('color: blue;')
        layout.addWidget(self.result_label)
        
        self.confidence_label = QLabel('')
        self.confidence_label.setFont(QFont('Arial', 12))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
        self.start_button = QPushButton('开始实时识别')
        self.start_button.setStyleSheet('background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;')
        self.start_button.clicked.connect(self.toggle_recognition)
        layout.addWidget(self.start_button)
        
        self.load_button = QPushButton('加载音频文件')
        self.load_button.setStyleSheet('background-color: #2196F3; color: white; padding: 10px; font-size: 14px;')
        self.load_button.clicked.connect(self.load_audio_file)
        layout.addWidget(self.load_button)
        
        self.settings_button = QPushButton('设置与分析')
        self.settings_button.setStyleSheet('background-color: #ff9800; color: white; padding: 10px; font-size: 14px;')
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def load_model(self):
        self.status_label.setText('正在加载模型...')
        self.status_bar.showMessage('加载模型中...')
        
        try:
            self.device = torch.device('cuda' if self.config.USE_GPU and torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            self.model = GenderDetectorModel(self.config)
            
            model_paths = [
                os.path.join(self.config.MODEL_SAVE_PATH, f'{self.config.MODEL_NAME}_best.pth'),
                os.path.join(self.config.MODEL_SAVE_PATH, f'{self.config.MODEL_NAME}_final.pth'),
                os.path.join(self.config.MODEL_SAVE_PATH, f'{self.config.MODEL_NAME}.pth')
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Model loaded from {model_path}")
                    if 'val_acc' in checkpoint:
                        print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("No model found. Using untrained model.")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.audio_processor = AudioProcessor(self.config)
            
            self.status_label.setText('模型加载成功！')
            self.status_bar.showMessage('就绪')
            
        except Exception as e:
            self.status_label.setText(f'模型加载失败: {str(e)}')
            self.status_bar.showMessage('错误')
    
    def open_settings(self):
        QMessageBox.information(self, '设置', '设置功能正在开发中...')
    
    def show_about(self):
        about_text = "声音性别识别 v1.0\n"
        about_text += "基于深度学习的实时性别识别系统\n"
        about_text += "支持实时识别和音频文件识别\n"
        about_text += "提供特征可视化和模型分析功能\n"
        about_text += "© 2026 Voice Gender Detector"
        
        QMessageBox.information(self, '关于', about_text)
    
    def toggle_recognition(self):
        if self.audio_thread is None or not self.audio_thread.is_running:
            self.start_recognition()
        else:
            self.stop_recognition()
    
    def start_recognition(self):
        if self.model is None:
            QMessageBox.warning(self, '错误', '请先加载模型！')
            return
        
        self.stop_recognition()
        
        self.audio_thread = AudioThread(self.config, self.audio_processor, self.model, self.device)
        self.audio_thread.update_signal.connect(self.update_result)
        self.audio_thread.error_signal.connect(self.show_error)
        self.audio_thread.audio_data_signal.connect(self.update_waveform)
        self.audio_thread.start()
        
        self.start_button.setText('停止实时识别')
        self.start_button.setStyleSheet('background-color: #f44336; color: white; padding: 10px; font-size: 14px;')
        self.status_bar.showMessage('正在录音...')
    
    def stop_recognition(self):
        if self.audio_thread:
            try:
                self.audio_thread.update_signal.disconnect(self.update_result)
            except:
                pass
            try:
                self.audio_thread.error_signal.disconnect(self.show_error)
            except:
                pass
            
            self.audio_thread.stop()
            self.audio_thread = None
        
        self.start_button.setText('开始实时识别')
        self.start_button.setStyleSheet('background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;')
        self.status_bar.showMessage('已停止')
    
    def update_result(self, gender, confidence):
        print(f"更新UI: gender={gender}, confidence={confidence}")
        gender_text = '男声' if gender == 'male' else '女声'
        self.result_label.setText(gender_text)
        self.confidence_label.setText(f'置信度: {confidence:.1%}')
        
        color = '#2196F3' if gender == 'male' else '#E91E63'
        self.result_label.setStyleSheet(f'color: {color}; font-weight: bold;')
        QApplication.processEvents()
    
    def update_waveform(self, audio_data):
        self.waveform_widget.set_audio_data(audio_data)
    
    def show_error(self, error_msg):
        QMessageBox.critical(self, '错误', error_msg)
        self.stop_recognition()
    
    def load_audio_file(self):
        if self.model is None:
            QMessageBox.warning(self, '错误', '请先加载模型！')
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择音频文件', '', '音频文件 (*.wav)'  # 只支持WAV文件
        )
        
        if file_path:
            try:
                print(f"加载文件: {file_path}")
                # 使用scipy.io.wavfile加载WAV文件
                sr, audio = wavfile.read(file_path)
                
                # 转换为float32
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                
                # 归一化
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
                
                gender, confidence = self.audio_processor.predict(audio, self.model, self.device)
                
                print(f"预测结果: gender={gender}, confidence={confidence}")
                self.update_result(gender, confidence)
                self.status_bar.showMessage(f'已加载: {os.path.basename(file_path)}')
                
            except Exception as e:
                print(f"加载音频错误: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, '错误', f'加载音频失败: {str(e)}')

def main():
    print("=" * 60)
    print("声音性别识别 - 应用模式")
    print("=" * 60)
    
    config = Config()
    
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
