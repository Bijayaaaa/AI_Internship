#!/bin/bash

echo "Starting FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Gradio..."
python gradio_app.py &

wait
