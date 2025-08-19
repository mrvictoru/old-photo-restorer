import io
import os
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from .restoration import KontextRestorer

app = FastAPI(
    title="Old Photo Restorer",
    description="Restore, colorize, and upscale old photos using FLUX Kontext.",
    version="1.0.0"
)

# Allow CORS for local testing and frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Instantiate once per process at startup
    access_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    app.state.restorer = KontextRestorer(
        model_id=os.getenv("KONTEXT_MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev"),
        access_token=access_token,
    )

@app.post("/restore")
async def restore_photo(
    request: Request,
    file: UploadFile = File(...),
    user_prompt: str = Form(""),
    guidance_scale: float = Form(3.5),
    strength: float = Form(0.35),
    num_inference_steps: int = Form(28),
    upscale_factor: float = Form(2.0),
    seed: int = Form(None),
):
    img = Image.open(io.BytesIO(await file.read()))
    result_img = request.app.state.restorer.restore(
        img,
        user_prompt=user_prompt,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        upscale_factor=upscale_factor,
        seed=seed
    )
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

@app.get("/")
def index():
    return JSONResponse({
        "message": "Old Photo Restorer API. POST an image to /restore."
    })
