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
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QLineEdit, QMessageBox
)


class GazeEstimationONNX:
    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        self.session = session or ort.InferenceSession(model_path, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[2:][::-1])
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size).astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0).astype(np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits: np.ndarray, yaw_logits: np.ndarray):
        pitch = np.sum(self.softmax(pitch_logits) * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(self.softmax(yaw_logits) * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return np.radians(pitch[0]), np.radians(yaw[0])

    def estimate(self, face_image: np.ndarray):
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {"input": input_tensor})
        return self.decode(outputs[0], outputs[1])


class GazeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medición de Atención - Gaze Estimation")
        self.setGeometry(100, 100, 400, 200)

        self.label = QLabel("Ingrese número del grupo:")
        self.input_grupo = QLineEdit()
        self.button_start = QPushButton("Iniciar Medición")
        self.button_start.clicked.connect(self.iniciar_medicion)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input_grupo)
        layout.addWidget(self.button_start)

        self.setLayout(layout)

    def iniciar_medicion(self):
        grupo_num = self.input_grupo.text().strip()
        if not grupo_num.isdigit():
            QMessageBox.warning(self, "Error", "Por favor, ingrese un número de grupo válido.")
            return

        grupo = f"Grupo{grupo_num}"
        model_path = "weights/resnet18_gaze.onnx"

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"No se encontró el modelo: {model_path}")
            return

        self.capture_attention(grupo, model_path)

    def capture_attention(self, grupo, model_path):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la cámara.")
            return

        engine = GazeEstimationONNX(model_path=model_path)
        detector = uniface.RetinaFace()

        attention_data = []
        prev_second = -1
        start_time = time.time()

        QMessageBox.information(self, "Instrucciones", "Presiona 'q' para terminar la grabación.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_second = int(time.time() - start_time)
            if current_second != prev_second:
                prev_second = current_second
                bboxes, _ = detector.detect(frame)

                attention_score = 0
                total_faces = 0

                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size == 0:
                        continue

                    pitch, yaw = engine.estimate(face_crop)

                    if -15 < np.degrees(pitch) < 15 and -15 < np.degrees(yaw) < 15:
                        attention_score += 1
                    total_faces += 1

                    draw_bbox_gaze(frame, bbox, pitch, yaw)

                ratio = round(attention_score / total_faces, 2) if total_faces > 0 else 0.0
                attention_data.append({"segundo": current_second, "atencion": ratio})
                print(f"Segundo {current_second}: Atención = {ratio}")

            cv2.imshow("Atención al Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if attention_data:
            df = pd.DataFrame(attention_data)
            fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("metricas", exist_ok=True)
            json_path = os.path.join("metricas", f"Grupo{grupo}_{fecha_hora}.json")
            df.to_json(json_path, orient="records", indent=2)
            QMessageBox.information(self, "Guardado", f"Métricas guardadas como:\n{json_path}")
        else:
            QMessageBox.warning(self, "Atención", "No se capturaron métricas.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = GazeApp()
    ventana.show()
    sys.exit(app.exec_())
