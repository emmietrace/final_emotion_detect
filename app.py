# app.py - RENDER VERSION (NO GEVENT, SYNC GUNICORN)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model  # Works with tensorflow-cpu
import base64

# -------------------------------
# FLASK SETUP
# -------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# LOAD MODEL (Download if missing - safe for Render)
# -------------------------------
MODEL_PATH = 'emotion_model_vortex.h5'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading model from GitHub...")
    urllib.request.urlretrieve(
        'https://github.com/emmietrace/final_emotion_detect/raw/main/emotion_model_vortex.h5',
        MODEL_PATH
    )
MODEL = load_model(MODEL_PATH)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -------------------------------
# DATABASE (SQLite - persists on Render disk)
# -------------------------------
DB_PATH = 'emotions.db'
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image_path TEXT,
            emotion TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_db()

def save_to_db(name, image_path, emotion, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO detections (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (name, image_path, emotion, confidence, timestamp)
    )
    conn.commit()
    conn.close()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    return np.expand_dims(norm, axis=[-1, 0])

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name', 'Unknown')
    file = request.files.get('image')
    if file and name:
        filename = f"{int(datetime.now().timestamp())}_{name.replace(' ', '_')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = np.argmax(pred)
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
        save_to_db(name, f"/static/uploads/{filename}", emotion, conf)
        return jsonify({
            'emotion': emotion.capitalize(),
            'confidence': round(conf * 100, 1),
            'image_url': f"/static/uploads/{filename}"
        })
    return jsonify({'error': 'Name and image required'}), 400

@app.route('/webcam', methods=['POST'])
def webcam():
    name = request.json.get('name', 'Unknown')
    data = request.json.get('image')
    if not data or not name:
        return jsonify({'error': 'Name and image required'}), 400
    try:
        data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid webcam image'}), 400
        filename = f"webcam_{int(datetime.now().timestamp())}_{name.replace(' ', '_')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = np.argmax(pred)
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
        save_to_db(name, f"/static/uploads/{filename}", emotion, conf)
        return jsonify({
            'emotion': emotion.capitalize(),
            'confidence': round(conf * 100, 1),
            'image_url': f"/static/uploads/{filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# -------------------------------
# NO VERCEL HANDLER - Use Gunicorn on Render
# -------------------------------
if __name__ == "__main__":
    # For local testing only
    app.run(host='0.0.0.0', port=8000, debug=False)
