from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import uuid
import threading
import gdown

app = Flask(__name__)
CORS(app)

# Upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# MODEL CONFIG
# =========================
MODEL_PATH = "pneumonia_cnn_model.h5"
FILE_ID = "1z6ER6Xjagz1hMyCi0AT8sajHpTJTx2xt" 

model = None
lock = threading.Lock()

# =========================
# LAZY MODEL LOADER
# =========================
def load_model_safe():
    global model

    with lock:
        if model is None:
            print("Loading model...")

            from tensorflow.keras.models import load_model

            # download only if missing
            if not os.path.exists(MODEL_PATH):
                print("Downloading model from Google Drive...")
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)

            model = load_model(MODEL_PATH, compile=False)

            print("Model loaded successfully!")


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    filename = str(uuid.uuid4()) + ".jpg"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        # simple validation
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        is_gray = (
            np.allclose(img_np[:, :, 0], img_np[:, :, 1], atol=10) and
            np.allclose(img_np[:, :, 1], img_np[:, :, 2], atol=10)
        )

        if not is_gray:
            return jsonify({"error": "Invalid X-ray image"}), 400

        # 🔥 IMPORTANT: load model ONLY here
        load_model_safe()

        img_array = preprocess_image(path)
        pred = model.predict(img_array, verbose=0)[0][0]

        result = "Pneumonia" if pred > 0.5 else "Normal"
        confidence = float(max(pred, 1 - pred))

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)