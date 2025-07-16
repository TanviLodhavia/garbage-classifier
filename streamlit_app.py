import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("garbage_classifier_model.keras")
    class_names = sorted(os.listdir("dataset"))  # Assumes dataset/ contains class folders
    return model, class_names

model, class_names = load_model()

# Streamlit UI
st.title("♻️ Garbage Classifier")
st.write("Upload an image of garbage to predict its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image_data.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.caption("⚠️ This model may make mistakes - please double-check predictions if needed.")

    with st.expander("🔍 Show all class probabilities"):
        for i, prob in enumerate(pred[0]):
            st.write(f"{class_names[i]}: {prob*100:.2f}%")
    # st.balloons()  # Celebrate the prediction with balloons