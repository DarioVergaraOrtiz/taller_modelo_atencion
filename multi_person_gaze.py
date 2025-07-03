import cv2
import uniface
import numpy as np
import onnxruntime as ort
import time
import json
import os
from datetime import datetime

from utils.helpers import draw_bbox_gaze


class GazeEstimationONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = (448, 448)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.input_mean, dtype=np.float32)
        std = np.array(self.input_std, dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits, yaw_logits):
        pitch_probs = self.softmax(pitch_logits)
        yaw_probs = self.softmax(yaw_logits)
        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return np.radians(pitch[0]), np.radians(yaw[0])

    def estimate(self, face_image):
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self.decode(outputs[0], outputs[1])


def main():
    grupo = input("Ingrese el nombre del grupo (ej. Grupo1): ")
    model_path = "weights/resnet18_gaze.onnx"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    detector = uniface.RetinaFace()
    engine = GazeEstimationONNX(model_path)

    print(f"🎥 Grabando para {grupo}... Presiona 'q' para finalizar.")

    start_time = time.time()
    prev_second = -1
    attention_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_second = int(time.time() - start_time)
        if current_second != prev_second:
            prev_second = current_second
            bboxes, _ = detector.detect(frame)

            personas = []
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    continue

                pitch, yaw = engine.estimate(face_crop)
                atencion = 1.0 if -15 < np.degrees(pitch) < 15 and -15 < np.degrees(yaw) < 15 else 0.0

                personas.append({
                    "id": f"persona{i+1}",
                    "atencion": atencion
                })

                draw_bbox_gaze(frame, bbox, pitch, yaw)

                print(f"Segundo {current_second} - persona{i+1}: Atención = {atencion}")

            attention_data.append({
                "segundo": current_second,
                "personas": personas
            })

        cv2.imshow("Cámara - Atención múltiple", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Guardar JSON
    os.makedirs("metricas", exist_ok=True)
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta_json = f"metricas/{grupo}_{fecha}.json"

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump({grupo: attention_data}, f, indent=4, ensure_ascii=False)

    print(f"✅ Datos guardados en '{ruta_json}'")


if __name__ == "__main__":
    main()
