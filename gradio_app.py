# gradio_app.py
import gradio as gr
import time
import psutil
import platform
import io
from datetime import datetime

from app.model_loader import predict_from_bytes, load_model_once

# ─────────────────────────────────────────────
# Load model once and measure load time
# ─────────────────────────────────────────────
model_load_start = time.time()
load_model_once()
MODEL_LOAD_TIME = time.time() - model_load_start


# ─────────────────────────────────────────────
# Prediction function with metrics
# ─────────────────────────────────────────────
def predict_ui(img):
    # Handle missing input
    if img is None:
        return "No image", 0.0, "0 ms", "N/A", "N/A", "N/A"

    # timestamp
    timestamp = str(datetime.now()).split(".")[0]

    # Start timer
    start = time.time()

    # Convert PIL → bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Predict
    result = predict_from_bytes(buf.read())

    latency = time.time() - start
    fps = 1 / latency if latency > 0 else 0

    # CPU + RAM
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    # System specs
    sys_specs = (
        f"Processor: {platform.processor()}\n"
        f"Machine: {platform.machine()}\n"
        f"System: {platform.system()}"
    )

    # Resource metrics
    resource = f"CPU: {cpu}%   |   RAM: {ram}%"

    return (
        result["class_name"],
        result["confidence"],
        f"{latency*1000:.2f} ms",
        f"{fps:.2f} FPS",
        sys_specs,
        resource,
        f"{MODEL_LOAD_TIME:.2f} sec",
        timestamp
    )


# ─────────────────────────────────────────────
# Gradio UI Layout
# ─────────────────────────────────────────────
with gr.Blocks(title="DenseNet Tiny-ImageNet Classifier Dashboard") as iface:

    gr.Markdown(
        """
         DenseNet Tiny-ImageNet Classifier  
         Real-Time Metrics • System Stats • Model Info
        """
    )

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            class_name = gr.Text(label="Predicted Class")
            confidence = gr.Number(label="Confidence")
            latency = gr.Text(label="Latency")
            fps = gr.Text(label="Throughput (FPS)")
            model_time = gr.Text(label="Model Load Time")
            timestamp = gr.Text(label="Timestamp")

    with gr.Row():
        system_specs = gr.Text(label="System Specifications")
        resource_usage = gr.Text(label="CPU / RAM Usage")

    submit_btn = gr.Button("Predict")

    submit_btn.click(
        predict_ui,
        inputs=img_input,
        outputs=[
            class_name,
            confidence,
            latency,
            fps,
            system_specs,
            resource_usage,
            model_time,
            timestamp
        ]
    )


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
