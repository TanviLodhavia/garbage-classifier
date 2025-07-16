# ♻️ Garbage Classifier using Computer Vision

A Convolutional Neural Network (CNN) based deep learning project that classifies different types of garbage using image data. This project is designed to demonstrate how computer vision and transfer learning can contribute to sustainable waste management.

---

## 🚀 Project Overview

This project uses **TensorFlow/Keras** to train a multi-class image classification model that identifies garbage into six categories:

- **Cardboard**
- **Glass**
- **Metal**
- **Paper**
- **Plastic**
- **Trash (miscellaneous)**

It includes:
- A robust CNN model with data augmentation
- A web app interface built with **Streamlit**
- Support for real-time predictions via **webcam feed**
- Ready-to-deploy `.keras` model

---

## 🧠 Technologies Used

| Tool            | Purpose                               |
|-----------------|----------------------------------------|
| **Python**      | Core programming language              |
| **TensorFlow**  | Deep learning framework                |
| **Keras**       | High-level model API                   |
| **OpenCV**      | Real-time image capture & processing   |
| **NumPy**       | Image array manipulation               |
| **Streamlit**   | Lightweight UI for ML apps             |
| **PIL**         | Image handling in Python               |

---

## 📦 Dataset

The dataset used for training is based on the open-source **TrashNet** dataset, originally published by Stanford:

- Dataset Source: [TrashNet (GitHub)](https://github.com/garythung/trashnet)
- 6 Classes, ~2,500+ total labeled images
- Images resized to `128x128` for efficient training

> ⚠️ Due to GitHub size limitations, the full dataset is not included here.  
> However, the folder structure used for training was:
>
> ```
> dataset/
> ├── cardboard/
> ├── glass/
> ├── metal/
> ├── paper/
> ├── plastic/
> └── trash/
> ```

---

## 🧪 Model Training

- Model: Custom CNN (Conv2D, MaxPooling, Dense)
- Input size: 128x128x3
- Loss function: `categorical_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`
- Data augmentation:
  - Horizontal flipping
  - Zooming
  - Rescaling

Training stops early using `EarlyStopping` with patience of 4 to prevent overfitting.

The trained model is saved as:  
`garbage_classifier_model.keras`

---

## 🎮 Live Demo Web App

Built using **Streamlit**.

### Features:
- Upload any image file (JPG/PNG)
- Predicts the garbage category with confidence %
- View all class probabilities
- Mobile and browser compatible
- Warning disclaimer that predictions are probabilistic

### Run locally:

```bash
streamlit run streamlit_app.py
