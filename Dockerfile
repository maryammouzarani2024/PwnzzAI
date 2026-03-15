FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install \
    --retries 10 \
    --timeout 600 \
    -r requirements.txt

COPY . .

RUN mkdir -p uploads downloads instance

# Expose Flask port
EXPOSE 8080

ENV FLASK_APP=main.py
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://ollama:11434

# Start Ollama in background, wait for it, then start Flask
CMD ollama serve > /tmp/ollama.log 2>&1 & sleep 5 && flask run --host=0.0.0.0 --port=8080 --no-reload
