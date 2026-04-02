from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import base64
import threading
import json
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

# Global state
camera_frame = None
camera_lock = threading.Lock()

def capture_camera():
    global camera_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    while True:
        ret, frame = cap.read()
        if ret:
            with camera_lock:
                # JPEG encode für Transfer
                _, buffer = cv2.imencode('.jpg', frame)
                camera_frame = base64.b64encode(buffer).decode('utf-8')
        else:
            time.sleep(0.1)

# Kamera-Thread starten
camera_thread = threading.Thread(target=capture_camera, daemon=True)
camera_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/camera-frame')
def get_camera_frame():
    with camera_lock:
        if camera_frame:
            return jsonify({'frame': camera_frame})
    return jsonify({'frame': None})

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized = (image_array.astype(np.float32) / 127.5) - 1
        
        # Inference
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
    app.run(debug=True, host='0.0.0.0', port=5000)