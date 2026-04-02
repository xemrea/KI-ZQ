import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import threading
import json
from datetime import datetime
from flask import Flask, render_template_string, jsonify, Response
import base64
from io import BytesIO

# =========================
# TensorFlow Lite Setup
# =========================
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

class ModelManager:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = []
        self.load_model()
    
    def load_model(self):
        """Lädt Modell und Labels"""
        try:
            # Labels laden
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            # Modell laden
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✓ Modell geladen ({len(self.class_names)} Klassen)")
        except Exception as e:
            print(f"✗ Fehler beim Laden: {e}")
            raise
    
    def preprocess(self, frame):
        """Bereitet Frame auf"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        image_array = np.asarray(image).astype(np.float32)
        normalized = (image_array / 127.5) - 1.0
        
        return np.expand_dims(normalized, axis=0).astype(np.float32)
    
    def predict(self, frame):
        """Führt Vorhersage durch"""
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        
        top_idx = np.argmax(output)
        confidence = float(output[top_idx])
        
        class_name = self.class_names[top_idx] if top_idx < len(self.class_names) else f"Unknown {top_idx}"
        # Entferne Index-Prefix
        if " " in class_name and class_name.split(" ")[0].isdigit():
            class_name = class_name.split(" ", 1)[1]
        
        # Top 3
        top_3_idx = np.argsort(output)[::-1][:3]
        top_3 = [
            {
                "name": self.class_names[idx].split(" ", 1)[1] if " " in self.class_names[idx] and self.class_names[idx].split(" ")[0].isdigit() else self.class_names[idx],
                "confidence": float(output[idx])
            }
            for idx in top_3_idx
        ]
        
        return {
            "class": class_name,
            "confidence": confidence,
            "top_3": top_3,
            "all_scores": output.tolist()
        }

class CameraManager:
    def __init__(self, camera_index=0):
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.open_camera(camera_index)
    
    def open_camera(self, index):
        """Öffnet Kamera"""
        for i in range(index, index + 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                print(f"✓ Kamera {i} geöffnet")
                return
        print("✗ Keine Kamera gefunden")
    
    def start(self):
        """Startet Frame-Capture in Thread"""
        if self.cap:
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            thread.start()
    
    def _capture_loop(self):
        """Capture Loop"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                break
    
    def get_frame(self):
        """Gibt aktuellen Frame zurück"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def release(self):
        """Gibt Kamera frei"""
        if self.cap:
            self.cap.release()

# =========================
# Flask App
# =========================
app = Flask(__name__)

# Globale Manager
model_manager = None
camera_manager = None
last_prediction = None

def init_managers():
    """Initialisiert Manager"""
    global model_manager, camera_manager
    model_manager = ModelManager()
    camera_manager = CameraManager()
    camera_manager.start()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/camera-feed')
def camera_feed():
    """Streaming Response"""
    def generate():
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue
            
            # Kleine Vorschau
            preview = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode('.jpg', preview)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture', methods=['POST'])
def capture():
    """Macht Foto und klassifiziert"""
    global last_prediction
    
    frame = camera_manager.get_frame()
    if frame is None:
        return jsonify({"error": "Kein Frame verfügbar"}), 400
    
    try:
        prediction = model_manager.predict(frame)
        last_prediction = prediction
        
        # Frame als Base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def status():
    """Status der App"""
    return jsonify({
        "model_loaded": model_manager is not None,
        "camera_ready": camera_manager is not None and camera_manager.frame is not None,
        "classes": len(model_manager.class_names) if model_manager else 0,
        "last_prediction": last_prediction
    })

# =========================
# HTML/CSS/JS Template
# =========================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vision - Kamera Klassifikation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #00d4ff;
            --primary-dark: #0099cc;
            --secondary: #ff006e;
            --bg-dark: #0a0e27;
            --bg-card: #1a1f3a;
            --text-light: #e0e0e0;
            --text-muted: #8a8a9e;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff3860;
        }

        body {
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a3e 100%);
            color: var(--text-light);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Header */
        .header {
            background: rgba(10, 14, 39, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            padding: 1.5rem 2rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            animation: slideDown 0.6s ease-out;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .logo::before {
            content: "⚡";
            font-size: 2rem;
        }

        .status {
            display: flex;
            gap: 1.5rem;
            align-items: center;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }

        .status-dot.offline {
            background: var(--danger);
            animation: none;
        }

        /* Container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            animation: fadeIn 0.8s ease-out;
        }

        /* Grid Layout */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 1000px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        /* Camera Card */
        .camera-card {
            background: var(--bg-card);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .camera-card:hover {
            border-color: var(--primary);
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
        }

        .camera-title {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: var(--primary);
            font-weight: 600;
        }

        .camera-preview {
            width: 100%;
            height: 400px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 1.5rem;
            border: 2px solid rgba(0, 212, 255, 0.1);
        }

        .camera-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .stream-container {
            width: 100%;
            height: 100%;
        }

        .stream-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .controls {
            display: flex;
            gap: 1rem;
        }

        /* Buttons */
        .btn {
            flex: 1;
            padding: 1rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: #000;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-secondary {
            background: rgba(255, 0, 110, 0.1);
            color: var(--secondary);
            border: 1px solid var(--secondary);
        }

        .btn-secondary:hover {
            background: rgba(255, 0, 110, 0.2);
        }

        /* Results Card */
        .results-card {
            background: var(--bg-card);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 2rem;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .results-empty {
            text-align: center;
            color: var(--text-muted);
        }

        .results-empty::before {
            content: "📸";
            display: block;
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .result-main {
            text-align: center;
            margin-bottom: 2rem;
        }

        .result-class {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        .result-confidence {
            font-size: 1.5rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }

        .confidence-bar {
            width: 100%;
            height: 8px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 2rem;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.5s ease;
        }

        .result-top3 {
            background: rgba(0, 212, 255, 0.05);
            border-radius: 8px;
            padding: 1.5rem;
        }

        .result-top3-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }

        .top3-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 0;
            border-bottom: 1px solid rgba(0, 212, 255, 0.1);
        }

        .top3-item:last-child {
            border-bottom: none;
        }

        .top3-name {
            font-weight: 500;
        }

        .top3-conf {
            font-size: 0.9rem;
            color: var(--primary);
            font-weight: 600;
        }

        /* Info Section */
        .info-section {
            background: var(--bg-card);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }

        .info-title {
            font-size: 1rem;
            color: var(--primary);
            margin-bottom: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }

        .info-item {
            background: rgba(0, 212, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }

        .info-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
        }

        .info-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Loading */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0, 212, 255, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        /* Animations */
        @keyframes slideDown {
            from {
                transform: translateY(-100%);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="logo">AI Vision</div>
            <div class="status">
                <div class="status-item">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">Verbunden</span>
                </div>
            </div>
        </div>
    </header>

    <div class="container">
        <div class="grid">
            <!-- Camera -->
            <div class="camera-card">
                <div class="camera-title">📹 Live Kamera Feed</div>
                <div class="camera-preview">
                    <div class="stream-container">
                        <img id="cameraStream" src="/api/camera-feed" alt="Camera Stream">
                    </div>
                </div>
                <div class="controls">
                    <button class="btn btn-primary" onclick="capturePhoto()">
                        📸 Klassifizieren
                    </button>
                    <button class="btn btn-secondary" onclick="clearResults()">
                        🔄 Zurücksetzen
                    </button>
                </div>
            </div>

            <!-- Results -->
            <div class="results-card">
                <div id="resultsContainer">
                    <div class="results-empty">
                        <p>Klicke auf "Klassifizieren" um<br>ein Bild zu analysieren</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Info -->
        <div class="info-section">
            <div class="info-title">📊 System Info</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-value" id="classCount">-</div>
                    <div class="info-label">Klassen</div>
                </div>
                <div class="info-item">
                    <div class="info-value" id="modelStatus">-</div>
                    <div class="info-label">Modell</div>
                </div>
                <div class="info-item">
                    <div class="info-value" id="cameraStatus">-</div>
                    <div class="info-label">Kamera</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Status Check
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('classCount').textContent = data.classes;
                document.getElementById('modelStatus').textContent = data.model_loaded ? '✓' : '✗';
                document.getElementById('cameraStatus').textContent = data.camera_ready ? '✓' : '✗';
                
                const statusDot = document.getElementById('statusDot');
                if (data.model_loaded && data.camera_ready) {
                    statusDot.classList.remove('offline');
                } else {
                    statusDot.classList.add('offline');
                }
            } catch (e) {
                console.error('Status update failed:', e);
            }
        }

        // Capture Photo
        async function capturePhoto() {
            const btn = event.target;
            btn.disabled = true;
            btn.innerHTML = '<div class="loading"></div> Verarbeite...';

            try {
                const res = await fetch('/api/capture', {method: 'POST'});
                const data = await res.json();

                if (data.success) {
                    displayResults(data.prediction, data.image);
                } else {
                    alert('Fehler: ' + data.error);
                }
            } catch (e) {
                alert('Fehler beim Klassifizieren: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '📸 Klassifizieren';
            }
        }

        // Display Results
        function displayResults(prediction, image) {
            const html = `
                <div class="result-main">
                    <div style="margin-bottom: 1.5rem; border-radius: 8px; overflow: hidden; border: 2px solid rgba(0, 212, 255, 0.2);">
                        <img src="${image}" style="width: 100%; height: auto; max-height: 300px; object-fit: cover;">
                    </div>
                    <div class="result-class">${prediction.class}</div>
                    <div class="result-confidence">${(prediction.confidence * 100).toFixed(1)}% Sicherheit</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${prediction.confidence * 100}%"></div>
                    </div>
                </div>
                <div class="result-top3">
                    <div class="result-top3-title">🏆 Top 3 Ergebnisse</div>
                    ${prediction.top_3.map((item, i) => `
                        <div class="top3-item">
                            <span class="top3-name">${i + 1}. ${item.name}</span>
                            <span class="top3-conf">${(item.confidence * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
            `;
            document.getElementById('resultsContainer').innerHTML = html;
        }

        // Clear Results
        function clearResults() {
            document.getElementById('resultsContainer').innerHTML = `
                <div class="results-empty">
                    <p>Klicke auf "Klassifizieren" um<br>ein Bild zu analysieren</p>
                </div>
            `;
        }

        // Init
        updateStatus();
        setInterval(updateStatus, 3000);
    </script>
</body>
</html>
'''

# =========================
# Start
# =========================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 AI Vision - Modern Camera Classification")
    print("="*50 + "\n")
    
    try:
        init_managers()
        print("\n✓ App startet auf: http://localhost:5000")
        print("  Öffne im Browser und klassifiziere Bilder!\n")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"✗ Fehler: {e}")