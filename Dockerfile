FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads downloads instance

EXPOSE 8080

ENV FLASK_APP=main.py
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://ollama:11434

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080", "--no-reload"]
