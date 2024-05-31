import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Define class names
class_names = ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"]

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    if confidence > 0.5:
        print(f"This image is a {class_names[predicted_class]} with {confidence * 100:.2f}% confidence.")
    else:
        print("This image is not an animal.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Animal-Detector.py <path_to_image>")
    else:
        img_path = sys.argv[1]
        classify_image(img_path)
