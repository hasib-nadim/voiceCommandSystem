import numpy as np
import sounddevice as sd
import soundfile as sf
import sys
from pathlib import Path
import librosa
# PyQt5 UI
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal

# Matplotlib for plotting
import matplotlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import sys
from pathlib import Path
import json
import concurrent.futures

# PyQt5 UI
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal

# Matplotlib for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import dsp
import sys, subprocess, webbrowser

class Recorder(QtCore.QObject):
    """Non-blocking recorder using sounddevice callback and soundfile.
    Emits signals for status updates so UI can stay responsive.
    """
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, samplerate=16000, channels=1, parent=None):
        super().__init__(parent)
        self.samplerate = samplerate
        self.channels = channels
        self._file = None
        self._stream = None

    def start(self, filename: str):
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self._file = sf.SoundFile(filename, mode='w', samplerate=self.samplerate, channels=self.channels, subtype='PCM_16')

            def callback(indata, frames, time, status):
                if status:
                    # forward status as error text
                    self.error.emit(str(status))
                # write frames (convert to 2D if needed)
                self._file.write(indata.copy())

            self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=callback)
            self._stream.start()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            if self._file is not None:
                filename = self._file.name
                self._file.close()
                self._file = None
                self.finished.emit(filename)
        except Exception as e:
            self.error.emit(str(e))


class ChunkRecorder(QtCore.QThread):
    """Continuous recorder thread that reads small frames from an InputStream and emits them.
    Emits numpy 1-D float32 arrays via chunk_ready signal.
    """
    chunk_ready = pyqtSignal(object)  # emits numpy array or None on error

    def __init__(self, frames_per_buffer=64, samplerate=16000, channels=1, parent=None):
        super().__init__(parent)
        self.frames_per_buffer = int(frames_per_buffer)
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self._running = False
        self._stream = None

    def run(self):
        self._running = True
        try:
            with sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype='float32', blocksize=self.frames_per_buffer) as stream:
                self._stream = stream
                while self._running:
                    try:
                        data, overflow = stream.read(self.frames_per_buffer)
                        arr = np.asarray(data, dtype=np.float32).reshape(-1)
                        self.chunk_ready.emit(arr)
                    except Exception:
                        self.chunk_ready.emit(None)
                        break
        except Exception:
            self.chunk_ready.emit(None)
        finally:
            self._stream = None

    def stop(self):
        self._running = False
        try:
            if self._stream is not None:
                try:
                    self._stream.abort()
                except Exception:
                    pass
        finally:
            self.wait(2000)


class MainWindow(QtWidgets.QMainWindow):
    process_result = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Voice Command Recorder')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # three record rows
        self.record_buttons = []
        self.combo_boxes = []
        self.status_labels = []
        self.recorders = []
        self.voiceComands = []

        Path('sample').mkdir(parents=True, exist_ok=True)

        for i in range(3):
            row = QtWidgets.QHBoxLayout()
            btn = QtWidgets.QPushButton(f'Record voice {i+1}')
            combo = QtWidgets.QComboBox()
            combo.addItems(['open_calculator', 'open_notepad', 'open_browser', 'on_light', 'off_light', 'custom'])
            fname = f'sample/voice_{i+1}.wav'
            status_text = 'saved' if Path(fname).exists() else 'idle'
            if(status_text == 'saved'):
                data, _ = librosa.load(fname, sr=16000, mono=True)
                self.voiceComands.append(dsp.clearNoise(data))
            status = QtWidgets.QLabel(status_text)
            row.addWidget(btn)
            row.addWidget(combo)
            row.addWidget(status)
            layout.addLayout(row)

            self.record_buttons.append(btn)
            self.combo_boxes.append(combo)
            self.status_labels.append(status)

            rec = Recorder()
            rec.finished.connect(self._on_finished)
            rec.error.connect(self._on_error)
            self.recorders.append(rec)

            btn.setCheckable(True)
            btn.toggled.connect(lambda checked, idx=i: self.toggle_record(checked, idx))

        # bindings file
        self.bindings_file = Path('sample/voice_bindings.json')
        self._load_bindings()
        for i, combo in enumerate(self.combo_boxes):
            combo.currentIndexChanged.connect(lambda _idx, j=i: self._save_bindings())

        # Start Listening button
        self.listen_btn = QtWidgets.QPushButton('Start Listening')
        layout.addWidget(self.listen_btn)
        self.listen_btn.setCheckable(True)

        # Matplotlib canvas for waveform (initially empty)
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Live waveform')
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')
        layout.addWidget(self.canvas)

        # chunk recorder thread placeholder
        self.chunk_thread = None
        self.listen_btn.toggled.connect(self._toggle_listen)

        # Log area (compact, scrollable)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.log.setFixedHeight(220)
        self.log.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        layout.addWidget(QtWidgets.QLabel('Log'))
        layout.addWidget(self.log)

        # background processing executor
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.process_result.connect(self._on_process_result)

        # plotting buffer (5s)
        self._plot_buffer_len = 5 * 16000
        self._plot_buffer = np.zeros(self._plot_buffer_len, dtype=np.float32)
        self._plot_head = 0


    def _toggle_listen(self, checked: bool):
        if checked:
            if self.chunk_thread is None or not self.chunk_thread.isRunning():
                self.chunk_thread = ChunkRecorder(frames_per_buffer=16000 * 3, samplerate=16000, channels=1)
                self.chunk_thread.chunk_ready.connect(self._on_chunk)
                self.chunk_thread.start()
                self.log_msg('Started continuous listening')
                self.listen_btn.setText('Stop Listening')
        else:
            if self.chunk_thread is not None:
                try:
                    self.chunk_thread.stop()
                    self.chunk_thread.wait(1000)
                except Exception:
                    pass
                self.chunk_thread = None
            self.log_msg('Stopped continuous listening')
            self.listen_btn.setText('Start Listening')

    def _on_chunk(self, arr):
        if arr is None:
            self.log_msg('Error while recording chunk')
            return
        # submit processing to executor (safe, bounded)
        try:
            data_copy = np.array(arr, copy=True)
            # submit a processing job for each stored voice command
            for idx, vc in enumerate(self.voiceComands):
                self._executor.submit(self._process_chunk, data_copy, vc, idx)
        except Exception:
            # fallback synchronous: process each command inline
            for idx, vc in enumerate(self.voiceComands):
                self._process_chunk(np.array(arr, copy=True), vc, idx)

        # append to rolling buffer and plot last window
        n = len(arr)
        if n == 0:
            return
        if n >= self._plot_buffer_len:
            self._plot_buffer = arr[-self._plot_buffer_len :].astype(np.float32)
            self._plot_head = 0
        else:
            end = self._plot_head + n
            if end <= self._plot_buffer_len:
                self._plot_buffer[self._plot_head:end] = arr.astype(np.float32)
                self._plot_head = end % self._plot_buffer_len
            else:
                first = self._plot_buffer_len - self._plot_head
                self._plot_buffer[self._plot_head:] = arr[:first].astype(np.float32)
                self._plot_buffer[: n - first] = arr[first:].astype(np.float32)
                self._plot_head = (n - first)

        plot_window = min(self._plot_buffer_len, 5 * 16000)
        head = self._plot_head
        if head == 0:
            view = self._plot_buffer[-plot_window:]
        else:
            if head >= plot_window:
                view = self._plot_buffer[head - plot_window : head]
            else:
                view = np.concatenate((self._plot_buffer[-(plot_window - head):], self._plot_buffer[:head]))
        view = dsp.clearNoise(view)
        self.ax.clear()
        self.ax.plot(view, linewidth=0.5)
        self.ax.set_xlim(0, len(view))
        self.ax.set_ylim(-1.0, 1.0)
        self.canvas.draw()

    def _process_chunk(self, arr, commandArr, commandIdx):
        # runs in worker thread; schedule result emit on main thread via QTimer.singleShot
        try:
            if arr is None:
                result = {'error': 'no data'}
            else:
                rating, lag = dsp.max_normalized_cross_correlation(arr,commandArr)
        except Exception as e:
            result = {'error': str(e)}

        # emit signal (Qt will queue the signal delivery to the main thread)
        try:
            self.process_result.emit((rating, lag, commandIdx))
        except Exception:
            # fallback: schedule via singleShot if direct emit fails
            try:
                QtCore.QTimer.singleShot(0, lambda r=(rating, lag, commandIdx): self.process_result.emit(r))
            except Exception:
                pass

    def _on_process_result(self, data):
        rating, lag, commandIdx = data
        if(rating >= 0.3):
            # get command name from commandIdx
            try:
                cmd_name = self.combo_boxes[commandIdx].currentText()
            except Exception:
                cmd_name = None

            if cmd_name:
                self.log_msg(f"Detected command '{cmd_name}' (idx={commandIdx+1}) rating={rating:.2f} lag={lag}")
                # try:
                #     if cmd_name == 'open_calculator':
                #         if sys.platform.startswith('win'):
                #             subprocess.Popen(['calc.exe'])
                #         elif sys.platform.startswith('darwin'):
                #             subprocess.Popen(['open', '-a', 'Calculator'])
                #         else:
                #             subprocess.Popen(['gnome-calculator'])
                #     elif cmd_name == 'open_notepad':
                #         if sys.platform.startswith('win'):
                #             subprocess.Popen(['notepad.exe'])
                #         elif sys.platform.startswith('darwin'):
                #             subprocess.Popen(['open', '-a', 'TextEdit'])
                #         else:
                #             # fallback to common editors
                #             for cmd in (['gedit'], ['kate'], ['xed'], ['nano']):
                #                 try:
                #                     subprocess.Popen(cmd)
                #                     break
                #                 except Exception:
                #                     continue
                #     elif cmd_name == 'open_browser':
                #         webbrowser.open('http://www.google.com')
                #     elif cmd_name == 'on_light':
                #         # placeholder: integrate with actual IoT/API
                #         self.log_msg('Action: turn light ON (placeholder)')
                #     elif cmd_name == 'off_light':
                #         self.log_msg('Action: turn light OFF (placeholder)')
                #     else:
                #         # custom or unknown command
                #         self.log_msg(f'Custom command selected: {cmd_name}')
                # except Exception as e:
                #     self.log_msg(f'Failed to execute command {cmd_name}: {e}')
             


    def log_msg(self, msg: str):
        # append and keep scrolled to end
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def toggle_record(self, checked: bool, idx: int):
        btn = self.record_buttons[idx]
        status = self.status_labels[idx]
        recorder = self.recorders[idx]
        filename = f'sample/voice_{idx+1}.wav'
        if checked:
            status.setText('recording...')
            self.log_msg(f'Started recording {filename}')
            recorder.start(filename)
            btn.setText('Stop')
        else:
            recorder.stop()
            btn.setText(f'Record voice {idx+1}')
            status.setText('stopping...')
            data, _ = librosa.load(filename, sr=16000, mono=True)
            self.voiceComands.insert(idx,dsp.clearNoise(data))

    def _on_finished(self, filename: str):
        try:
            p = Path(filename)
            name = p.name
            for i in range(len(self.status_labels)):
                if name == f'voice_{i+1}.wav':
                    self.status_labels[i].setText('saved')
                    self.log_msg(f'Saved {filename}')
                    return
        except Exception:
            pass
        self.log_msg(f'Recording finished: {filename}')

    def _on_error(self, text: str):
        self.log_msg(f'Error: {text}')

    def _load_bindings(self):
        try:
            if self.bindings_file.exists():
                data = json.loads(self.bindings_file.read_text())
                for i, combo in enumerate(self.combo_boxes):
                    key = f'voice_{i+1}'
                    if key in data:
                        val = data[key]
                        idx = combo.findText(val)
                        if idx != -1:
                            combo.setCurrentIndex(idx)
        except Exception as e:
            self.log_msg(f'Failed to load bindings: {e}')

    def _save_bindings(self):
        try:
            data = {}
            for i, combo in enumerate(self.combo_boxes):
                data[f'voice_{i+1}'] = combo.currentText()
            self.bindings_file.write_text(json.dumps(data, indent=2))
            # silent save (no spam) - keep optional log
        except Exception as e:
            self.log_msg(f'Failed to save bindings: {e}')

    def closeEvent(self, event):
        # stop continuous recording thread and shutdown executor
        try:
            if self.chunk_thread is not None and self.chunk_thread.isRunning():
                try:
                    self.chunk_thread.stop()
                except Exception:
                    pass
                try:
                    self.chunk_thread.wait(1000)
                except Exception:
                    pass
                self.chunk_thread = None
        except Exception:
            pass
        try:
            if hasattr(self, '_executor'):
                try:
                    self._executor.shutdown(wait=False)
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(700, 600)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()