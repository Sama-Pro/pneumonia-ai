from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid
import gdown

app = Flask(__name__)
CORS(app)

# Upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = "model.keras"
FILE_ID = "1KRCf9V7LHATul3LeuYT_vyR39uA-HNjj"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

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

# Serve frontend
@app.route('/')
def home():
    return render_template("index.html")

# Predict API with strong grayscale validation
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Open image
        img = Image.open(file_path).convert('RGB')
        img_np = np.array(img)

        # Grayscale check: X-ray images are usually grayscale
        is_grayscale = np.allclose(img_np[:,:,0], img_np[:,:,1], atol=10) and np.allclose(img_np[:,:,1], img_np[:,:,2], atol=10)
        mean_val = img_np.mean()  # simple intensity check to filter extremely dark/light images

        if not is_grayscale or mean_val < 30 or mean_val > 220:
            return jsonify({"error": "Please upload a valid chest X-ray image"}), 400

        # Preprocess and predict
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

# Download uploaded file
@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

# Run server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)