import sys
import os
import ctypes
import cv2
import uniface
import numpy as np
import onnxruntime as ort
import pandas as pd
import time
from datetime import datetime
from utils.helpers import draw_bbox_gaze

# Carga manual de libvlc.dll (ajusta la ruta según tu instalación de VLC)
if os.name == 'nt':
    vlc_dll = r"C:\Program Files\VideoLAN\VLC\libvlc.dll"
    ctypes.CDLL(vlc_dll)

import vlc
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal


class GazeEstimationONNX:
    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        self.session = session or ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]  # sólo CPU
        )
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        self.input_size = tuple(input_cfg.shape[2:][::-1])
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size).astype(np.float32)/255.0
        img = (img - self.input_mean) / self.input_std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def softmax(self, x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

    def decode(self, p, y):
        pitch = np.sum(self.softmax(p) * self.idx_tensor, axis=1)*self._binwidth - self._angle_offset
        yaw   = np.sum(self.softmax(y) * self.idx_tensor, axis=1)*self._binwidth - self._angle_offset
        return np.radians(pitch[0]), np.radians(yaw[0])

    def estimate(self, img):
        inp = self.preprocess(img)
        p, y = self.session.run(self.output_names, {"input": inp})
        return self.decode(p, y)


class CameraThread(QThread):
    finished = pyqtSignal(list)
    def __init__(self, grupo, model_path):
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
        data=[]; prev=-1; start=time.time()
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            sec=int(time.time()-start)
            if sec!=prev:
                prev=sec
                bboxes,_=detector.detect(frame)
                a=0; tot=0
                for bb in bboxes:
                    x1,y1,x2,y2=map(int,bb[:4]); crop=frame[y1:y2,x1:x2]
                    if crop.size==0: continue
                    p,yw=engine.estimate(crop)
                    if -15<np.degrees(p)<15 and -15<np.degrees(yw)<15: a+=1
                    tot+=1
                    draw_bbox_gaze(frame, bb, p, yw)
                data.append({"segundo":sec, "atencion": round(a/tot,2) if tot>0 else 0.0})
            cv2.imshow("Camara", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        cap.release(); cv2.destroyAllWindows(); self.finished.emit(data)


class GazeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atención: Video + Cámara")
        self.setGeometry(100,100,800,600)
        # VLC
        self.vlci=vlc.Instance()
        self.player=self.vlci.media_player_new()
        # Qt widget como contenedor
        self.video_widget=QWidget(self);
        self.video_widget.setMinimumSize(640,360)
        self.player.set_hwnd(int(self.video_widget.winId()))
        # Controles
        self.inp=QLineEdit(); self.inp.setPlaceholderText("Núm. grupo")
        btn=QPushButton("Iniciar Medic."); btn.clicked.connect(self.start_capture)
        lay=QHBoxLayout(); lay.addWidget(self.inp); lay.addWidget(btn)
        main=QVBoxLayout(); main.addWidget(self.video_widget); main.addLayout(lay)
        c=QWidget(); c.setLayout(main); self.setCentralWidget(c)
        self.thread=None; self.model="weights/resnet18_gaze.onnx"
        self.load_video()
    def load_video(self):
        d='videos';
        if os.path.isdir(d):
            fs=[f for f in os.listdir(d) if f.lower().endswith(('.mp4','.avi','.mov'))]
            if fs:
                m=self.vlci.media_new(os.path.join(d,fs[0])); self.player.set_media(m); self.player.play()
    def start_capture(self):
        g=self.inp.text().strip();
        if not g.isdigit(): QMessageBox.warning(self,"Error","Número inválido"); return
        grp=f"Grupo{g}";
        if not os.path.exists(self.model): QMessageBox.critical(self,"Error","Modelo no hallado"); return
        self.player.play();
        self.thread=CameraThread(grp,self.model);
        self.thread.finished.connect(self.finish)
        self.thread.start()
    def finish(self,data):
        self.player.pause();
        if data:
            df=pd.DataFrame(data)
            ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("metricas",exist_ok=True)
            f=os.path.join("metricas",f"Grupo{self.inp.text()}_{ts}.json")
            df.to_json(f,orient="records",indent=2)
            QMessageBox.information(self,"Guardado",f"{f}")
        else: QMessageBox.information(self,"Info","Sin métricas")
    def closeEvent(self,e):
        if self.thread and self.thread.isRunning(): self.thread.running=False; self.thread.wait()
        e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=GazeMainWindow(); w.show(); sys.exit(app.exec_())
