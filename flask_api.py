from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import uuid
import gdown
import threading

app = Flask(__name__)
CORS(app)

# Upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model settings
MODEL_PATH = "model.keras"
FILE_ID = "1KRCf9V7LHATul3LeuYT_vyR39uA-HNjj"

# Lazy-loaded model (IMPORTANT FIX)
model = None
lock = threading.Lock()


def load_model_safe():
    """Download + load model only when needed"""
    global model

    with lock:
        if model is None:
            from tensorflow.keras.models import load_model

            # Download if missing
            if not os.path.exists(MODEL_PATH):
                print("Downloading model from Google Drive...")
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=True)

            print("Loading model...")
            model = load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully!")


# Preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Home route
@app.route('/')
def home():
    return render_template("index.html")


# Predict API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save file
    filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Load image
        img = Image.open(file_path).convert('RGB')
        img_np = np.array(img)

        # Basic validation (X-ray check)
        is_grayscale = (
            np.allclose(img_np[:, :, 0], img_np[:, :, 1], atol=10) and
            np.allclose(img_np[:, :, 1], img_np[:, :, 2], atol=10)
        )

        mean_val = img_np.mean()

        if not is_grayscale or mean_val < 30 or mean_val > 220:
            return jsonify({"error": "Please upload a valid chest X-ray image"}), 400

        # Ensure model is loaded
        load_model_safe()

        # Predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array, verbose=0)[0][0]

        result = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else (1 - prediction)

        return jsonify({
            "prediction": result,
            "confidence": round(float(confidence) * 100, 2),
            "filename": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Download endpoint
@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)


# IMPORTANT for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)