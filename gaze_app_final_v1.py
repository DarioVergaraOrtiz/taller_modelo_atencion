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
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox,
    QFrame, QSizePolicy, QDesktopWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

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
    frame_ready = pyqtSignal(np.ndarray)

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
                attention_score = 0
                total_faces = 0
                for bb in bboxes:
                    x1, y1, x2, y2 = map(int, bb[:4])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    pitch, yaw = engine.estimate(crop)
                    if -15 < np.degrees(pitch) < 15 and -15 < np.degrees(yaw) < 15:
                        attention_score += 1
                    total_faces += 1
                    draw_bbox_gaze(frame, bb, pitch, yaw)
                ratio = round(attention_score / total_faces, 2) if total_faces > 0 else 0.0
                data.append({"segundo": sec, "atencion": ratio})
                print(f"Segundo {sec}: Atención = {ratio}")
            
            # Enviar el frame procesado a la interfaz principal
            self.frame_ready.emit(frame)
            
            # Comprobar si se presiona 'q' para salir
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.finished.emit(data)


class GazeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Attention Dashboard")
        self.setMinimumSize(800, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #ecf0f1;
            }
            QLineEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
            QFrame {
                background-color: #34495e;
                border-radius: 8px;
            }
        """)

        # Central widget with layout
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: transparent;")
        self.setCentralWidget(central_widget)
        
        # Main layout - vertical centrado
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("Gaze Attention Tracker")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
            padding: 8px;
        """)
        main_layout.addWidget(title)

        # Video principal (contenido) - CENTRADO
        video_main_frame = QFrame()
        video_main_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 2px solid #3498db;
                border-radius: 8px;
            }
        """)
        video_main_frame.setFixedSize(640, 360)
        video_main_layout = QVBoxLayout(video_main_frame)
        video_main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000;")
        self.video_label.setMinimumSize(640, 360)
        video_main_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_main_frame, alignment=Qt.AlignCenter)

        # Contenedor para cámara y controles (debajo del video principal)
        bottom_container = QHBoxLayout()
        bottom_container.setSpacing(20)
        bottom_container.setAlignment(Qt.AlignCenter)

        # Vista previa de cámara (secundaria)
        camera_frame = QFrame()
        camera_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 2px solid #3498db;
                border-radius: 8px;
            }
        """)
        camera_frame.setFixedSize(240, 135)  # Tamaño más pequeño
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_preview = QLabel()
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setStyleSheet("background-color: #000000;")
        self.camera_preview.setMinimumSize(240, 135)
        camera_layout.addWidget(self.camera_preview)
        
        # Título para la vista de cámara
        camera_title = QLabel("Cámara")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_title.setStyleSheet("font-size: 12px; color: #bdc3c7; padding: 5px;")
        
        # Contenedor para cámara con título
        camera_container = QVBoxLayout()
        camera_container.setSpacing(5)
        camera_container.addWidget(camera_title, alignment=Qt.AlignCenter)
        camera_container.addWidget(camera_frame)
        
        # Controls container
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(20, 10, 20, 10)
        ctrl_layout.setSpacing(15)

        # Group input
        group_container = QWidget()
        group_layout = QVBoxLayout(group_container)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(5)
        
        group_label = QLabel("Número de Grupo:")
        group_label.setStyleSheet("font-size: 14px; color: #ecf0f1;")
        group_layout.addWidget(group_label)
        
        self.input_grupo = QLineEdit()
        self.input_grupo.setPlaceholderText("Ejemplo: 1")
        self.input_grupo.setFixedWidth(200)
        group_layout.addWidget(self.input_grupo)
        ctrl_layout.addWidget(group_container)

        # Button
        self.btn_start = QPushButton("Iniciar Medición")
        self.btn_start.setFixedWidth(200)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_measure)
        ctrl_layout.addWidget(self.btn_start, alignment=Qt.AlignCenter)

        # Agregar elementos al contenedor inferior
        bottom_container.addLayout(camera_container)
        bottom_container.addWidget(ctrl_frame)
        
        main_layout.addLayout(bottom_container)

        # Status bar
        self.status_label = QLabel("Estado: Preparado")
        self.status_label.setStyleSheet("color: #bdc3c7; font-size: 12px; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Members
        self.video_path = None
        self.cap = None
        self.player = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.camera_thread = None
        self.model_path = "weights/resnet18_gaze.onnx"

        # Load first video but do not auto-start
        self.load_first_video()

    def center_window(self):
        # Centrar la ventana en la pantalla
        frame = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

    def showEvent(self, event):
        super().showEvent(event)
        self.center_window()
        event.accept()

    def load_first_video(self):
        videos_dir = "videos"
        if os.path.isdir(videos_dir):
            files = [f for f in os.listdir(videos_dir) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if files:
                self.video_path = os.path.join(videos_dir, files[0])
                self.cap = cv2.VideoCapture(self.video_path)
                self.player = MediaPlayer(
                    self.video_path,
                    ff_opts={'paused': True, 'sync_audio': True}
                )
                self.status_label.setText(f"Estado: Video cargado - {os.path.basename(self.video_path)}")

    def start_measure(self):
        grp = self.input_grupo.text().strip()
        if not grp.isdigit() or not self.cap:
            QMessageBox.warning(self, "Error", "Grupo inválido o video no cargado.")
            return
        group = f"Grupo{grp}"
        
        # Start video & audio
        self.player.set_pause(False)
        self.timer.start(30)
        self.status_label.setText(f"Estado: Midiendo atención para {group}...")
        
        # Start camera thread
        self.camera_thread = CameraThread(group, self.model_path)
        self.camera_thread.finished.connect(self.finish)
        self.camera_thread.frame_ready.connect(self.update_camera_preview)
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
        pix = QPixmap.fromImage(qt_img).scaled(
            640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def update_camera_preview(self, frame):
        """Actualiza la vista previa de la cámara"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        pix = pix.scaled(240, 135, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_preview.setPixmap(pix)

    def finish(self, data):
        self.timer.stop()
        if self.player:
            self.player.close_player()
        self.btn_start.setEnabled(True)
        
        if data:
            df = pd.DataFrame(data)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("metricas", exist_ok=True)
            fname = f"Grupo{self.input_grupo.text()}_{ts}.json"
            path = os.path.join("metricas", fname)
            df.to_json(path, orient="records", indent=2)
            self.status_label.setText(f"Estado: Métricas guardadas en {fname}")
            QMessageBox.information(
                self, "Guardado", f"Métricas guardadas en:\n{path}"
            )
        else:
            self.status_label.setText("Estado: No se capturaron métricas")
            QMessageBox.information(self, "Info", "No se capturaron métricas.")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.player:
            self.player.close_player()
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.running = False
            self.camera_thread.wait()
        event.accept()

    # Para pantalla completa
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = GazeMainWindow()
    window.show()
    
    sys.exit(app.exec_())