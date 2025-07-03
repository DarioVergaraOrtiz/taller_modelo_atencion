import sys
import os
import cv2
import uniface
import numpy as np
import onnxruntime as ort
import pandas as pd
import time
from datetime import datetime
from utils.helpers import draw_bbox_gaze

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from ffpyplayer.player import MediaPlayer


class GazeEstimationONNX:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[2:][::-1])
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size).astype(np.float32) / 255.0
        img = (img - self.input_mean) / self.input_std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits: np.ndarray, yaw_logits: np.ndarray):
        p_probs = self.softmax(pitch_logits)
        y_probs = self.softmax(yaw_logits)
        pitch = np.sum(p_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(y_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return np.radians(pitch[0]), np.radians(yaw[0])

    def estimate(self, face: np.ndarray):
        inp = self.preprocess(face)
        p, y = self.session.run(self.output_names, {self.input_name: inp})
        return self.decode(p, y)


class CameraThread(QThread):
    finished = pyqtSignal(list)

    def __init__(self, grupo: str, model_path: str):
        super().__init__()
        self.grupo = grupo
        self.model_path = model_path
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.finished.emit([])
            return
        engine = GazeEstimationONNX(self.model_path)
        detector = uniface.RetinaFace()
        prev_sec = -1
        start = time.time()
        data = []
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            sec = int(time.time() - start)
            if sec != prev_sec:
                prev_sec = sec
                bboxes, _ = detector.detect(frame)
                persons = []
                for idx, bb in enumerate(bboxes):
                    x1, y1, x2, y2 = map(int, bb[:4])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    pitch, yaw = engine.estimate(crop)
                    atn = 1.0 if -15 < np.degrees(pitch) < 15 and -15 < np.degrees(yaw) < 15 else 0.0
                    persons.append({"id": f"persona{idx+1}", "atencion": atn})
                    draw_bbox_gaze(frame, bb, pitch, yaw)
                    print(f"Segundo {sec} - persona{idx+1}: Atención = {atn}")
                data.append({"segundo": sec, "personas": persons})
            cv2.imshow("Camara - Atención", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.finished.emit(data)


class GazeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atención al Video con FFpyplayer")
        self.setGeometry(100, 100, 800, 600)
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        # Controls
        self.input_grupo = QLineEdit()
        self.input_grupo.setPlaceholderText("Ingrese número de grupo (ej. 1)")
        self.btn_start = QPushButton("Iniciar Medición")
        self.btn_start.clicked.connect(self.start_measure)
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.input_grupo)
        ctrl_layout.addWidget(self.btn_start)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(ctrl_layout)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        # Members
        self.video_path = None
        self.cap = None
        self.player = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.camera_thread = None
        self.model_path = "weights/resnet18_gaze.onnx"
        # load first video but do not auto-start
        self.load_first_video()

    def load_first_video(self):
        d = "videos"
        if os.path.isdir(d):
            fs = [f for f in os.listdir(d) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if fs:
                self.video_path = os.path.join(d, fs[0])
                self.cap = cv2.VideoCapture(self.video_path)
                # pause audio initialization
                self.player = MediaPlayer(self.video_path, ff_opts={'paused': True, 'sync_audio': True})

    def start_measure(self):
        grp = self.input_grupo.text().strip()
        if not grp.isdigit() or not self.cap:
            QMessageBox.warning(self, "Error", "Grupo inválido o video no cargado.")
            return
        group = f"Grupo{grp}"
        # start video & audio
        self.player.set_pause(False)
        self.timer.start(30)
        # start camera thread
        self.camera_thread = CameraThread(group, self.model_path)
        self.camera_thread.finished.connect(self.finish)
        self.camera_thread.start()
        self.btn_start.setEnabled(False)

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def finish(self, data):
        # stop video & audio
        self.timer.stop()
        if self.player:
            self.player.close_player()
        self.btn_start.setEnabled(True)
        # save metrics
        if data:
            df = pd.DataFrame(data)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("metricas", exist_ok=True)
            fname = f"Grupo{self.input_grupo.text()}_{ts}.json"
            path = os.path.join("metricas", fname)
            df.to_json(path, orient="records", indent=2)
            QMessageBox.information(self, "Guardado", f"Métricas guardadas en:\n{path}")
        else:
            QMessageBox.information(self, "Info", "No se capturaron métricas.")

    def closeEvent(self, event):
        # cleanup
        if self.cap:
            self.cap.release()
        if self.player:
            self.player.close_player()
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.running = False
            self.camera_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GazeMainWindow()
    win.show()
    sys.exit(app.exec_())
