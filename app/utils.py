# app/utils.py

import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path: str, target_size=(64, 64)):
    """Load and preprocess image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array
