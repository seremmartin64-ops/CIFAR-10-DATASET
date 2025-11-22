🍅 Tomato Leaf Disease Detection using Convolutional Neural Networks (CNN)

🧠 Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify tomato leaf images into various disease categories. The model is trained on a Kaggle dataset containing several tomato plant diseases and healthy leaf images.
By the end of training, the model is able to identify diseases from tomato leaf images with strong accuracy, making it suitable for agricultural diagnostics and plant health monitoring.
This project is part of the AI Engineering Beginner Series Project Implementation, where learners apply their foundational deep learning knowledge to solve real-world computer vision problems.

🎯 Project Objectives

1 Build and train a CNN model for tomato leaf disease classification.
2 Understand preprocessing methods such as normalization and image augmentation.
3 Learn how to evaluate deep learning models using validation accuracy and loss.
4 Visualize predictions and interpret confidence scores.
5 Apply AI techniques to solve real agricultural challenges.

🧰 Step-by-Step Implementation

Step 1: Import Libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

Step 2: Download Dataset from Kaggle

You upload `kaggle.json`, configure Kaggle API, download the Tomato Leaf Disease dataset, and unzip it.

The dataset structure contains:

 train → images used for training
 val → images used for validation

Step 3: Data Preprocessing and Augmentation**

Images are normalized (rescale = 1/255) and augmented using:

- Rotation
- Zoom
- Horizontal flipping

This improves model generalization.

data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
---

Step 4: Build the CNN Model

The CNN consists of:

- 3 Convolutional Layers
- MaxPooling Layers
- Fully Connected Dense Layer
- Output Softmax Layer

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

Step 5: Compile and Train the Model

The model uses:

- Optimizer: Adam
- Loss: **Categorical Crossentropy**
- Metric: **Accuracy**

Training runs for 10 epochs.

Step 6: Evaluate the Model

The validation set is used to compute:

- Final Loss
- Final Accuracy

This helps measure how well the model generalizes.

Step 7: Predict New Images

The model predicts disease labels for sample images, showing:

- Actual label
- Predicted label
- Confidence percentage

Images are displayed in a grid for easy comparison.

📊 Results and Performance

Typical results after 10 epochs:

- Training Accuracy: 89.80%
- Loss:0.3321

The CNN model successfully recognizes tomato leaf diseases such as:

* Early Blight
* Late Blight
* Leaf Mold
* Healthy

Many predictions show confidence levels above 90%

💡 Application Areas

This agricultural AI solution demonstrates practical use cases across multiple sectors:

| Area                        | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| 🌾 Smart Farming            | Detect plant diseases early and reduce crop losses.   |
| 🧪 Research Labs            | Automated plant pathology analysis.                   |
| 📱 Mobile Apps              | Real-time plant disease detection for farmers.        |
| 🌍 Environmental Monitoring | Health tracking of large-scale crop fields.           |
| 🧑‍🏫 Education             | Beginner-friendly deep learning project for students. |


🚀 How to Run This Project

Requirements

- Google Colab or Jupyter Notebook
- TensorFlow 2.x
- Matplotlib
- Kaggle API Key

Steps to Run

1. Open the notebook in Google Colab.
2. Upload your `kaggle.json` file.
3. Run all cells in order.
4. View the training graph and prediction results.
5. Test with your own leaf images.
   
  📦 Submission Details
* Course: AI Engineering Beginner Series
* Module: Project Implementation
* Instructor: Martial School of IT
* Submission Portal: AI Engineering Project Implementation Portal
* Deliverables:
  * Notebook (.ipynb)
  * README Documentation
  * Model file (optional)
  * Prediction screenshots (optional)

 🧑‍💻 Author

martin serem 
Beginner Machine Learning Developer
Passionate about AI, agriculture innovation, and practical deep learning applications.

 🏁 Conclusion

This project provides a strong foundation in CNN-based image recognition. It demonstrates how deep learning can be used to diagnose plant diseases, enabling impactful real-world agricultural solutions.

Learners are encouraged to enhance the model using:

* Transfer Learning (e.g., MobileNetV2, ResNet50)
* Dropout Layers
* Increased image resolution
* More training epochs

