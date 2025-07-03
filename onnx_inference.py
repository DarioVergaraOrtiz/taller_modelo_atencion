# gaze_estimation_interface.py

import os
import cv2
import uniface
import numpy as np
import onnxruntime as ort
import pandas as pd
from datetime import datetime
import time
from typing import Tuple
from utils.helpers import draw_bbox_gaze


class GazeEstimationONNX:
    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        self.session = session
        if self.session is None:
            assert model_path is not None, "Model path is required for the first time initialization."
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
            )

        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        self.input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:][::-1])

        outputs = self.session.get_outputs()
        output_names = [output.name for output in outputs]

        self.output_names = output_names
        assert len(output_names) == 2, "Expected 2 output nodes, got {}".format(len(output_names))

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.input_mean, dtype=np.float32)
        std = np.array(self.input_std, dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        return image_batch

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits: np.ndarray, yaw_logits: np.ndarray) -> Tuple[float, float]:
        pitch_probs = self.softmax(pitch_logits)
        yaw_probs = self.softmax(yaw_logits)
        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return np.radians(pitch[0]), np.radians(yaw[0])

    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {"input": input_tensor})
        return self.decode(outputs[0], outputs[1])


def main():
    grupo = input("Ingrese nombre del grupo (ej. Grupo1): ")
    model_path = input("Ingrese la ruta del modelo ONNX (ej. resnet18_gaze.onnx): ")

    attention_data = []
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("No se pudo abrir la cámara")

    engine = GazeEstimationONNX(model_path=model_path)
    detector = uniface.RetinaFace()

    print(f"\n🎥 Grabando atención para '{grupo}'. Presiona 'q' o cierra la ventana para finalizar.")

    prev_second = -1

    try:
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
                print("\n🔚 Grabación finalizada por el usuario.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        if attention_data:
            df = pd.DataFrame(attention_data)
            fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Crear carpeta 'metricas' si no existe
            os.makedirs("metricas", exist_ok=True)

            # Guardar dentro de 'metricas/'
            json_path = os.path.join("metricas", f"{grupo}_{fecha_hora}.json")
            df.to_json(json_path, orient="records", indent=2)
            print(f"\n✅ Métricas guardadas en: {json_path}")
        else:
            print("⚠️ No se capturaron métricas.")


if __name__ == "__main__":
    main()
