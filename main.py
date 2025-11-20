from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import torch

app = FastAPI(
    title="NepaliBERT Sentiment Analysis API",
    description="FastAPI + ONNX deployment of NepaliBERT sentiment classifier (3-class)",
    version="1.0.0"
)


TOKENIZER_PATH = "model/tokenizer"      
MODEL_PATH     = "model/nepaliBERT_3class.onnx"               


# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


# LOAD ONNX MODEL
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# REQUEST SCHEMA
class TextInput(BaseModel):
    text: str


# PREPROCESSING
def preprocess(text):
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    return {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }


# PREDICTION FUNCTION
def predict_sentiment(text):

    inputs = preprocess(text)

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # ONNX forward pass
    logits = session.run(None, ort_inputs)[0]

    # Softmax
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]

    # Predicted class index
    pred_idx = int(np.argmax(probs))

    # Correct 3-class label mapping
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    return {
        "label": label_map[pred_idx],
        "probabilities": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        }
    }


# API ENDPOINTS
@app.post("/predict")
def predict(input: TextInput):
    return {
        "input": input.text,
        "prediction": predict_sentiment(input.text)
    }

@app.get("/")
def home():
    return {"message": "NepaliBERT 3-Class Sentiment Analysis API is running!"}
