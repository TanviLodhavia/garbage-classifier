import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model("garbage_classifier_model.keras")

# Load class names from folder structure
class_names = sorted(os.listdir("dataset"))  # Assumes same dataset folder

# Load and preprocess the test image
img_path = "test.png"  # Change this if needed
if not os.path.exists(img_path):
    raise FileNotFoundError(f"No image found at '{img_path}'. Make sure the file exists in the folder ")
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
