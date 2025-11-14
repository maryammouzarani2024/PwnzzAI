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


#Pull Ollama's Models: "mistral:7b", "llama3.2:1b"
RUN ollama pull mistral:7b
RUN ollama pull llama3.2:1b

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

# Run the application with no-reload option so that the user can see the logs straight in the Docker stdout
CMD ["flask", "run", "--host=0.0.0.0","--no-reload"]
