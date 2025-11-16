# Use TensorFlow 2.13 CPU image with Python 3.10
FROM tensorflow/tensorflow:2.13.0

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
