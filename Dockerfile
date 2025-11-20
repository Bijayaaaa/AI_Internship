# ------------------------
# Stage 1: Base image
# ------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------
# Stage 2: Copy application files
# ------------------------
COPY main.py .

# Copy model files
COPY model ./model

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
