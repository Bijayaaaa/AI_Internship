from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import torch
from huggingface_hub import hf_hub_download

app = FastAPI(
    title="NepaliBERT Sentiment Analysis API",
    description="FastAPI + ONNX deployment of NepaliBERT sentiment classifier (3-class)",
    version="1.0.0"
)

# Hugging Face repository
MODEL_REPO = "Bijaya1/nepali-bert-3class"

# Download ONNX model from HF
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="nepaliBERT_3class.onnx")

# Load tokenizer from HF
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Request schema
class TextInput(BaseModel):
    text: str

# Preprocessing function
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

# Prediction function
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

    # Class labels
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    return {
        "label": label_map[int(np.argmax(probs))],
        "probabilities": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        }
    }

# API endpoints
@app.post("/predict")
def predict(input: TextInput):
    return {
        "input": input.text,
        "prediction": predict_sentiment(input.text)
    }

@app.get("/")
def home():
    return {"message": "NepaliBERT 3-Class Sentiment Analysis API is running!"}
