import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(image_path, model_path):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (145, 145))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return "Fatigue" if prediction > 0.5 else "Active"
