from flask import Flask, render_template, jsonify, Response, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import base64
import threading
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Modell laden
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


class CameraManager:
    def __init__(self, camera_index=0):
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.open_camera(camera_index)

    def open_camera(self, index):
        for i in range(index, index + 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                print(f"Kamera {i} geöffnet")
                return
        print("Keine Kamera gefunden")

    def start(self):
        if self.cap:
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            thread.start()

    def _capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        if self.cap:
            self.cap.release()


camera_manager = CameraManager()
camera_manager.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/camera-feed')
def camera_feed():
    def generate():
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/classify', methods=['POST'])
def classify():
    frame = camera_manager.get_frame()
    if frame is None:
        return jsonify({'success': False, 'error': 'Kein Frame verfügbar'})

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized = (image_array.astype(np.float32) / 127.5) - 1

        interpreter.set_tensor(input_details[0]['index'],
                               np.expand_dims(normalized, axis=0).astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        index = np.argmax(output_data)
        label = class_names[index].strip()
        confidence = float(output_data[0][index])

        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
