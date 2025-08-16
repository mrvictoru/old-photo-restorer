# Old Photo Restorer

A Dockerized FastAPI web app to restore, colorize, and upscale old photos using FLUX Kontext (Diffusers).
Note: This is also a test to see how well AI agent can complete task and build simple software via the Github copilot interface.

## Features

- Restore damaged, faded, or low-res photos.
- Optional realistic colorization for grayscale images.
- Modern upscaling (default 2x).
- Prompt-based control for restoration style.
- API endpoint for web UI integration.

## Usage

### Local (GPU/CPU)
1. Clone the repo
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the FLUX Kontext model (default: `"black-forest-labs/FLUX.1-Kontext-dev"` from HuggingFace).
4. Run:
   ```
   uvicorn app.main:app --reload
   ```

### Docker (recommended)

```
docker build -t old-photo-restorer .
docker run --gpus all -p 8000:8000 old-photo-restorer
```

For CPU only (no `--gpus all`).

### API

- `POST /restore` (multipart/form-data)
    - `file`: image file (jpg, png)
    - `user_prompt`: (optional) guidance for restoration
    - other params: `guidance_scale`, `strength`, `num_inference_steps`, `upscale_factor`, `seed`
- Returns: Restored image (JPEG)

## Environment Variables

Copy `.env.sample` to `.env` and edit if needed:

```
KONTEXT_MODEL_ID=black-forest-labs/FLUX.1-Kontext-dev
```

## License

MIT
