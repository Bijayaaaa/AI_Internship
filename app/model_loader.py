# app/model_loader.py

import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import logging
from .labels import TINY_IMAGENET_LABELS

logger = logging.getLogger("model_loader")
model = None
MODEL_PATH = "/app/model/densenet_tinyimagenet_light.h5"

def load_model_once():
    global model
    if model is not None:
        return model
    logger.info(f"Loading model… {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except:
        model = tf.keras.applications.DenseNet121(weights=None, input_shape=(224,224,3), classes=200)
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    logger.info("Model loaded.")
    return model


def predict_from_bytes(image_bytes: bytes):
    global model
    if model is None:
        load_model_once()

    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224,224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    class_name = TINY_IMAGENET_LABELS.get(idx, "Unknown")

    return {
        "class_index": idx,
        "class_name": class_name,
        "confidence": conf
    }
