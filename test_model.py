from tensorflow.keras.models import load_model

print("Loading keras model...")
model = load_model("model.keras")

model.summary()

print("Model loaded successfully!")