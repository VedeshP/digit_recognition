import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("models/mnist_model.keras")

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))  # MNIST images are grayscale
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with highest probability
    return predicted_class[0]

image_path = "B:\STUDY_VEDESH_laptop\coding_new\digit_recognition\images\d3_image_0.jpg"
predicted_class = predict_image(image_path)
print(f"The predicted digit is: {predicted_class}")

# Mostly predicting 8