# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for Ollama installation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libzbar0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service temporarily, pull models, then stop
# The models will be baked into the Docker image
RUN ollama serve & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama to start..." && \
    sleep 10 && \
    echo "Pulling mistral:7b..." && \
    ollama pull mistral:7b && \
    echo "Pulling llama3.2:1b..." && \
    ollama pull llama3.2:1b && \
    echo "Models pulled successfully!" && \
    kill $OLLAMA_PID

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads downloads instance

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=main.py
ENV PYTHONUNBUFFERED=1

# Start Ollama in background, wait for it, then start Flask
CMD ollama serve > /tmp/ollama.log 2>&1 & sleep 5 && flask run --host=0.0.0.0 --no-reload
