# Dockerfile for GPU or CPU use
FROM python:3.10-slim

WORKDIR /app

# System dependencies for PIL, torch, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app

EXPOSE 8000

ENV KONTEXT_MODEL_ID=black-forest-labs/FLUX.1-Kontext-dev

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]