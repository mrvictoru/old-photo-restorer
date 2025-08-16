# Old Photo Restorer

A Dockerized FastAPI web app to restore, colorize, and upscale old photos using FLUX Kontext (Diffusers). This project uses PyTorch with GPU acceleration (CUDA) to restore old photos using deep learning models. It is packaged in a Docker container for easy deployment, with GPU support enabled via Docker Compose.

Note: This is also a test to see how well AI agent can complete task and build simple software via the Github copilot interface.

## Features

- Restore damaged, faded, or low-res photos.
- Optional realistic colorization for grayscale images.
- Modern upscaling (default 2x).
- Prompt-based control for restoration style.
- API endpoint for web UI integration.

## Requirements

- Docker
- NVIDIA GPU and drivers (with CUDA support)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host machine

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/mrvictoru/old-photo-restorer.git
cd old-photo-restorer
```

### 2. Build and run with Docker Compose

```bash
docker compose up --build
```

This will build the image using the `pytorch/pytorch:2.7.1-cuda12.8-cudnn8-runtime` base and launch the service, exposing the app on **port 8964**.

### 3. Access the service

By default, the API will be available at:

```
http://localhost:8964
```

## Docker Compose Configuration

- Uses Docker Compose version `0.1`
- GPU acceleration enabled (requires a compatible NVIDIA GPU and drivers)
- Exposes port **8964** (internal and external)
- Mounts your project directory into the container for live code updates

## Dockerfile Highlights

- **Base image:** `pytorch/pytorch:2.7.1-cuda12.8-cudnn8-runtime`
- Installs requirements from `requirements.txt`
- Exposes port 8964
- Runs the app using `uvicorn`

## Example Docker Compose File

```yaml
version: "0.1"

services:
  app:
    build: .
    ports:
      - "8964:8964"
    volumes:
      - .:/app
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped
```

## Example Dockerfile

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8964

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8964"]
```

