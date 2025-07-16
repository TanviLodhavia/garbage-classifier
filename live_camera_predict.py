import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os

# Load model
model = tf.keras.models.load_model("garbage_classifier_model.keras")


# Load class names (from dataset folder structure)
class_names = sorted(os.listdir("dataset"))

# Image size expected by the model
img_size=128

# Start video capture (0=default webcam)
cap=cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for prediction

    resized_frame = cv2.resize(frame, (img_size, img_size))
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict

    pred=model.predict(img_array)
    predicted_class=class_names[np.argmax(pred)]
    confidence = np.max(pred)*100

    # Overlay prediction on the frame

    label=f"{predicted_class} ({confidence:.2f}%)"
    cv2.putText(frame, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)

    # Show frame

    cv2.imshow("Garbage Classifier", frame)

    # Press 'q' to quit

    if cv2.waitKey(1) & 0xFF == ord ("q"):
        break


# Release camera

cap.release()
cv2.destroyAllWindows()