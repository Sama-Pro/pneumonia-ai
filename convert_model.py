from tensorflow.keras.models import load_model

print("Loading old model...")
model = load_model("pneumonia_cnn_model.h5", compile=False)

print("Saving in Keras format...")
model.save("model.keras")

print("Done!")