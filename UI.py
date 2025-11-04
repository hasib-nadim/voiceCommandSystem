import numpy as np
import sounddevice as sd
import soundfile as sf
import sys
from pathlib import Path 
import json

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from resemblyzer import VoiceEncoder, preprocess_wav
import dsp


class VoiceCommandCard(QtWidgets.QWidget):
    """Compact card for voice command."""
    record_toggled = pyqtSignal(bool, int)
    
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #252525);
                border-radius: 10px;
                border: 1px solid #3a3a3a;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header with status
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"Voice {self.index + 1}")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #fff;")
        self.status_dot = QtWidgets.QLabel("○")
        self.status_dot.setStyleSheet("color: #666; font-size: 18px;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_dot)
        layout.addLayout(header)
        
        # Command combo
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(['open_calculator', 'open_notepad', 'open_browser', 
                            'on_light', 'off_light', 'custom'])
        self.combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QComboBox:hover { border-color: #2196F3; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { 
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: #fff;
                selection-background-color: #2196F3;
                border: 1px solid #444;
            }
        """)
        layout.addWidget(self.combo)
        
        # Record button
        self.record_btn = QtWidgets.QPushButton("🎤 Record")
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self._on_record)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f44336, stop:1 #d32f2f);
            }
        """)
        layout.addWidget(self.record_btn)
    
    def _on_record(self, checked):
        if checked:
            self.record_btn.setText("⏹ Stop")
            self.set_status("recording", "#ff9800")
        else:
            self.record_btn.setText("🎤 Record")
        self.record_toggled.emit(checked, self.index)
    
    def set_status(self, status, color="#4CAF50"):
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 18px;")
        if status == "saved":
            self.status_dot.setText("✓")
        elif status == "recording":
            self.status_dot.setText("●")
        else:
            self.status_dot.setText("○")


class AudioBuffer(QtCore.QThread):
    def __init__(self, duration=5.0, sr=16000):
        super().__init__()
        self.sr = sr
        self.buffer_size = int(sr * duration)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.pos = 0
        self.lock = QtCore.QMutex()
        self._running = False

    def run(self):
        self._running = True
        with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', blocksize=1024) as stream:
            while self._running:
                data, _ = stream.read(1024)
                arr = data.reshape(-1)
                
                self.lock.lock()
                n = len(arr)
                if self.pos + n <= self.buffer_size:
                    self.buffer[self.pos:self.pos + n] = arr
                else:
                    first = self.buffer_size - self.pos
                    self.buffer[self.pos:] = arr[:first]
                    self.buffer[:n - first] = arr[first:]
                self.pos = (self.pos + n) % self.buffer_size
                self.lock.unlock()

    def get_audio(self, duration=3.0):
        samples = int(self.sr * duration)
        samples = min(samples, self.buffer_size)
        
        self.lock.lock()
        if self.pos >= samples:
            result = self.buffer[self.pos - samples:self.pos].copy()
        else:
            result = np.concatenate([
                self.buffer[-(samples - self.pos):],
                self.buffer[:self.pos]
            ])
        self.lock.unlock()
        return result

    def stop(self):
        self._running = False
        self.wait(2000)


class MainWindow(QtWidgets.QMainWindow):
    detection = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Voice Command System')
        
        # Data
        self.encoder = VoiceEncoder()
        self.embeddings = [None, None, None]
        self.audio_buffer = None
        self.last_detection = {}
        
        self._setup_ui()
        self._load_saved_commands()
        
        # Timers
        self.process_timer = QtCore.QTimer()
        self.process_timer.timeout.connect(self._process)
        self.viz_timer = QtCore.QTimer()
        self.viz_timer.timeout.connect(self._update_viz)
        self.detection.connect(self._on_detection)

    def _setup_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QLabel { color: #fff; }
        """)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("🎙️ Voice Commands")
        title.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            color: #2196F3;
            padding: 5px;
        """)
        header.addWidget(title)
        header.addStretch()
        main_layout.addLayout(header)
        
        # Command cards in row
        cards_layout = QtWidgets.QHBoxLayout()
        cards_layout.setSpacing(10)
        self.cards = []
        for i in range(3):
            card = VoiceCommandCard(i)
            card.record_toggled.connect(self._on_record)
            card.combo.currentIndexChanged.connect(self._save_bindings)
            self.cards.append(card)
            cards_layout.addWidget(card)
        main_layout.addLayout(cards_layout)
        
        # Control panel
        control_panel = QtWidgets.QWidget()
        control_panel.setStyleSheet("""
            QWidget {
                background: #222;
                border-radius: 8px;
                border: 1px solid #333;
            }
        """)
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        control_layout.setContentsMargins(12, 10, 12, 10)
        control_layout.setSpacing(15)
        
        # Threshold slider
        thresh_label = QtWidgets.QLabel("Threshold:")
        thresh_label.setStyleSheet("font-size: 12px; color: #aaa;")
        control_layout.addWidget(thresh_label)
        
        self.threshold_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(40)
        self.threshold_slider.setMaximum(75)
        self.threshold_slider.setValue(55)
        self.threshold_slider.setFixedWidth(150)
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
                border: 2px solid #0d47a1;
            }
            QSlider::handle:horizontal:hover {
                background: #42A5F5;
            }
        """)
        control_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QtWidgets.QLabel("0.55")
        self.threshold_label.setStyleSheet("font-size: 13px; color: #2196F3; font-weight: bold; min-width: 40px;")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f'{v/100:.2f}')
        )
        control_layout.addWidget(self.threshold_label)
        
        control_layout.addStretch()
        
        # Listen button
        self.listen_btn = QtWidgets.QPushButton("🎧 Start Listening")
        self.listen_btn.setCheckable(True)
        self.listen_btn.toggled.connect(self._toggle_listen)
        self.listen_btn.setFixedHeight(36)
        self.listen_btn.setFixedWidth(160)
        self.listen_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388E3C);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f44336, stop:1 #d32f2f);
            }
        """)
        control_layout.addWidget(self.listen_btn)
        
        main_layout.addWidget(control_panel)
        
        # Graph
        self.fig = Figure(figsize=(8, 2.5), facecolor='#1a1a1a')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        self.ax = self.fig.add_subplot(111, facecolor='#222')
        self.ax.set_title('Detection Scores', color='#fff', fontsize=13, pad=8, fontweight='bold')
        self.ax.set_xlabel('Commands', color='#888', fontsize=10)
        self.ax.set_ylabel('Similarity', color='#888', fontsize=10)
        self.ax.tick_params(colors='#888', labelsize=9)
        self.ax.grid(True, alpha=0.1, color='#444')
        for spine in self.ax.spines.values():
            spine.set_color('#333')
        self.ax.set_ylim(0, 1)
        self.fig.tight_layout(pad=2)
        main_layout.addWidget(self.canvas)
        self._update_detection_graph([0, 0, 0], None)
        
        # Log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        self.log.setStyleSheet("""
            QTextEdit {
                background-color: #222;
                color: #ccc;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        main_layout.addWidget(self.log)

    def _load_saved_commands(self):
        Path('sample').mkdir(exist_ok=True)
        Path('chunk').mkdir(exist_ok=True)
        
        for i in range(3):
            fname = f'sample/voice_{i+1}.wav'
            if Path(fname).exists():
                try:
                    audio = preprocess_wav(fname)
                    clean = dsp.clearNoise(audio)
                    self.embeddings[i] = self.encoder.embed_utterance(clean)
                    self.cards[i].set_status("saved", "#4CAF50")
                    self.log_msg(f'✓ Loaded voice {i+1}')
                except Exception as e:
                    self.log_msg(f'✗ Error loading voice {i+1}: {e}')
        
        bindings_file = Path('sample/voice_bindings.json')
        if bindings_file.exists():
            try:
                data = json.loads(bindings_file.read_text())
                for i, card in enumerate(self.cards):
                    key = f'voice_{i+1}'
                    if key in data:
                        idx = card.combo.findText(data[key])
                        if idx != -1:
                            card.combo.setCurrentIndex(idx)
            except:
                pass

    def _on_record(self, checked, idx):
        if checked:
            fname = f'sample/voice_{idx+1}.wav'
            Path(fname).parent.mkdir(exist_ok=True)
            
            self._rec_file = sf.SoundFile(fname, mode='w', samplerate=16000, 
                                          channels=1, subtype='PCM_16')
            
            def callback(indata, frames, time, status):
                self._rec_file.write(indata.copy())
            
            self._rec_stream = sd.InputStream(samplerate=16000, channels=1, callback=callback)
            self._rec_stream.start()
            self.log_msg(f'🎤 Recording voice {idx+1}...')
        else:
            if hasattr(self, '_rec_stream'):
                self._rec_stream.stop()
                self._rec_stream.close()
            if hasattr(self, '_rec_file'):
                fname = self._rec_file.name
                self._rec_file.close()
                
                try:
                    audio = preprocess_wav(fname)
                    clean = dsp.clearNoise(audio)
                    self.embeddings[idx] = self.encoder.embed_utterance(clean)
                    self.cards[idx].set_status("saved", "#4CAF50")
                    self.log_msg(f'✓ Saved voice {idx+1}')
                except Exception as e:
                    self.log_msg(f'✗ Error: {e}')

    def _toggle_listen(self, checked):
        if checked:
            self.audio_buffer = AudioBuffer(duration=5.0, sr=16000)
            self.audio_buffer.start()
            self.process_timer.start(250)
            self.viz_timer.start(100)
            self.listen_btn.setText("⏹ Stop")
            self.log_msg('🎧 Listening started')
        else:
            self.process_timer.stop()
            self.viz_timer.stop()
            if self.audio_buffer:
                self.audio_buffer.stop()
                self.audio_buffer = None
            self.listen_btn.setText("🎧 Start Listening")
            self.log_msg('⏹ Stopped')

    def _process(self):
        if not self.audio_buffer:
            return
        
        audio = self.audio_buffer.get_audio(2.5)
        if len(audio) < 8000:
            return
        
        energy = np.sqrt(np.mean(audio ** 2))
        if energy < 0.005:
            return
        
        active_commands = [i for i, emb in enumerate(self.embeddings) if emb is not None]
        if not active_commands:
            return
        
        try:
            clean = dsp.clearNoise(audio)
            current_embed = self.encoder.embed_utterance(preprocess_wav(clean))
            
            similarities = []
            for idx in range(3):
                if self.embeddings[idx] is not None:
                    sim = np.dot(current_embed, self.embeddings[idx]) / (
                        np.linalg.norm(current_embed) * np.linalg.norm(self.embeddings[idx]) + 1e-8
                    )
                    similarities.append(sim)
                else:
                    similarities.append(0.0)
            
            self._update_detection_graph(similarities, None)
            
            min_threshold = self.threshold_slider.value() / 100.0
            max_sim = max(similarities)
            
            if max_sim >= min_threshold:
                best_idx = similarities.index(max_sim)
                self.detection.emit((best_idx, max_sim, similarities))
                
        except Exception as e:
            self.log_msg(f'⚠ Process error: {e}')

    def _on_detection(self, data):
        cmd_idx, similarity, all_scores = data
        
        import time
        current_time = time.time()
        
        if cmd_idx in self.last_detection:
            if current_time - self.last_detection[cmd_idx] < 2.0:
                return
        
        self.last_detection[cmd_idx] = current_time
        cmd = self.cards[cmd_idx].combo.currentText()
        
        self.cards[cmd_idx].set_status("saved", "#00ff00")
        QtCore.QTimer.singleShot(500, lambda: self.cards[cmd_idx].set_status("saved", "#4CAF50"))
        
        self._update_detection_graph(all_scores, cmd_idx)
        
        scores_str = " | ".join([f"V{i+1}:{s:.2f}" for i, s in enumerate(all_scores)])
        self.log_msg(f'✓ {cmd} (Voice {cmd_idx+1}) [{scores_str}]')

    def _update_detection_graph(self, similarities, winner_idx):
        self.ax.clear()
        
        colors = ['#2196F3', '#FFC107', '#9C27B0']
        if winner_idx is not None:
            colors[winner_idx] = '#4CAF50'
        
        bars = self.ax.bar(
            ['V1', 'V2', 'V3'],
            similarities,
            color=colors,
            edgecolor='#fff',
            linewidth=1.2,
            alpha=0.9
        )
        
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            height = bar.get_height()
            label = f'{sim:.3f}'
            if i == winner_idx:
                label = f'✓ {label}'
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        label, ha='center', va='bottom', color='#fff', 
                        fontweight='bold', fontsize=10)
        
        threshold = self.threshold_slider.value() / 100.0
        self.ax.axhline(y=threshold, color='#f44336', linestyle='--', linewidth=1.5, 
                       alpha=0.8, label=f'Threshold ({threshold:.2f})')
        
        self.ax.set_ylim(0, 1.0)
        self.ax.set_ylabel('Similarity', color='#888', fontsize=10)
        self.ax.set_xlabel('Commands', color='#888', fontsize=10)
        self.ax.set_title('Detection Scores', color='#fff', fontsize=13, pad=8, fontweight='bold')
        self.ax.tick_params(colors='#888', labelsize=9)
        self.ax.grid(True, alpha=0.1, color='#444', linestyle='-', linewidth=0.5)
        self.ax.legend(loc='upper right', facecolor='#222', edgecolor='#333', 
                      labelcolor='#fff', fontsize=8, framealpha=0.9)
        
        for spine in self.ax.spines.values():
            spine.set_color('#333')
        
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    def _update_viz(self):
        pass

    def _save_bindings(self):
        try:
            data = {f'voice_{i+1}': card.combo.currentText() 
                    for i, card in enumerate(self.cards)}
            Path('sample/voice_bindings.json').write_text(json.dumps(data, indent=2))
        except:
            pass

    def log_msg(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def closeEvent(self, event):
        if self.listen_btn.isChecked():
            self.listen_btn.setChecked(False)
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(26, 26, 26))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(34, 34, 34))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(26, 26, 26))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(34, 34, 34))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(33, 150, 243))
    app.setPalette(palette)
    
    window = MainWindow()
    window.resize(750, 650)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()