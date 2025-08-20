import os
import uuid
import json
import shutil
import datetime
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from PIL import Image, ImageFilter

# Try to import your restoration implementation
MODEL_AVAILABLE = True
try:
    from app.restoration import restore_image as model_restore_image  # type: ignore
except Exception:
    MODEL_AVAILABLE = False
    model_restore_image = None  # type: ignore

# ------------------------------------------------------------------------------
# Paths and folders
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (app/..)
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
THUMBS_DIR = DATA_DIR / "thumbs"
HISTORY_DB = DATA_DIR / "history.json"

for p in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, THUMBS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Simple JSON history store with in-process lock
# ------------------------------------------------------------------------------
class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        if not self.db_path.exists():
            self._write({"items": {}})

    def _read(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {"items": {}}
        with self.db_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self.db_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.db_path)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read()
            items = list(data.get("items", {}).values())
        items.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return items

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._read()
            return data.get("items", {}).get(item_id)

    def add(self, item: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data.setdefault("items", {})
            data["items"][item["id"]] = item
            self._write(data)

    def update(self, item_id: str, fields: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            items = data.setdefault("items", {})
            if item_id not in items:
                raise KeyError("Item not found")
            items[item_id].update(fields)
            self._write(data)

    def delete_many(self, ids: List[str]) -> None:
        with self._lock:
            data = self._read()
            items = data.setdefault("items", {})
            for i in ids:
                items.pop(i, None)
            self._write(data)

store = HistoryStore(HISTORY_DB)

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(title="Old Photo Restorer")

# Static and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class Prompts(BaseModel):
    positive: str = Field("", description="Positive prompt")

class AdvancedSettings(BaseModel):
    guidance_scale: float = Field(7.5, ge=0, le=50)
    strength: float = Field(0.8, ge=0, le=1)
    steps: int = Field(30, ge=1, le=200)
    seed: Optional[int] = None

class RunRequest(BaseModel):
    id: str
    prompts: Prompts = Prompts()
    advanced: AdvancedSettings = AdvancedSettings()

class DeleteRequest(BaseModel):
    ids: List[str]

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _iso_now() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def _ext_from_filename(name: str) -> str:
    n = name.lower()
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        if n.endswith(ext):
            return ext
    return ".png"

def create_thumbnail(src_path: Path, dst_path: Path, size=(128, 128)) -> None:
    with Image.open(src_path) as im:
        im.thumbnail(size)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path)

def simulate_restore(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as im:
        im = im.convert("RGB").filter(ImageFilter.DETAIL).filter(ImageFilter.SHARPEN)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, quality=95)

def call_remote_api_restore(
    src_path: Path,
    dst_path: Path,
    prompts: Prompts,
    adv: AdvancedSettings,
    api_url: str,
) -> bool:
    """
    Call the remote /restore API endpoint.
    Returns True if successful, False if failed (for fallback).
    """
    try:
        # Get timeout from environment variable, default to 120 seconds
        timeout = int(os.getenv("RESTORE_API_TIMEOUT", "120"))
        
        # Prepare the files and data for the POST request
        with open(src_path, 'rb') as f:
            files = {'file': (src_path.name, f, 'image/*')}
            
            # Map server.py parameters to main.py API format
            data = {
                'user_prompt': prompts.positive,  # Use positive prompt as user_prompt
                'guidance_scale': adv.guidance_scale,
                'strength': adv.strength,
                'num_inference_steps': adv.steps,
                'upscale_factor': 2.0,  # Default value, not configurable in server.py
            }
            
            # Only include seed if it's set
            if adv.seed is not None:
                data['seed'] = adv.seed
            
            # Make the API request
            response = requests.post(
                f"{api_url.rstrip('/')}/restore",
                files=files,
                data=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                # Save the response image to dst_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_path, 'wb') as out_f:
                    out_f.write(response.content)
                return True
            else:
                print(f"Remote API call failed with status {response.status_code}: {response.text}")
                return False
                
    except Exception as e:
        print(f"Error calling remote API: {e}")
        return False

def call_model_restore(
    src_path: Path,
    dst_path: Path,
    prompts: Prompts,
    adv: AdvancedSettings,
):
    # Check if RESTORE_API_URL is set and try remote API first
    restore_api_url = os.getenv("RESTORE_API_URL")
    if restore_api_url:
        if call_remote_api_restore(src_path, dst_path, prompts, adv, restore_api_url):
            return  # Success, we're done
        print("Remote API call failed, falling back to local model")
    
    # Fall back to local model if remote API is not configured or failed
    if not MODEL_AVAILABLE or model_restore_image is None:
        simulate_restore(src_path, dst_path)
        return

    # Try 1: keyword-rich
    try:
        return model_restore_image(
            src_path,
            dst_path,
            prompts={"positive": prompts.positive},
            guidance_scale=adv.guidance_scale,
            steps=adv.steps,
            seed=adv.seed,
            strength=adv.strength,
        )
    except TypeError:
        pass

    # Try 2: positional prompts + adv dict
    try:
        return model_restore_image(
            src_path,
            dst_path,
            {"positive": prompts.positive},
            {
                "guidance_scale": adv.guidance_scale,
                "steps": adv.steps,
                "seed": adv.seed,
                "strength": adv.strength,
            },
        )
    except TypeError:
        pass

    # Try 3: simplest
    try:
        return model_restore_image(src_path, dst_path)
    except TypeError as e:
        raise RuntimeError(
            "app.restoration.restore_image signature not compatible. "
            "Expected one of: "
            "(src, dst, prompts=..., guidance_scale=..., steps=..., seed=..., strength=...), "
            "(src, dst, prompts_dict, adv_dict), or (src, dst)."
        ) from e

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root_redirect(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/history")
async def get_history():
    items = store.list()
    return {"items": items}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    item_id = str(uuid.uuid4())
    ext = _ext_from_filename(file.filename or "")
    input_path = UPLOADS_DIR / f"{item_id}{ext}"
    result_path = RESULTS_DIR / f"{item_id}.png"
    thumb_path = THUMBS_DIR / f"{item_id}.jpg"

    # Persist upload
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create thumbnail
    try:
        create_thumbnail(input_path, thumb_path)
    except Exception:
        pass

    uploaded_at = _iso_now()
    item = {
        "id": item_id,
        "original_filename": file.filename,
        "uploaded_at": uploaded_at,
        "input_url": f"/data/uploads/{input_path.name}",
        "result_url": None,
        "thumb_url": f"/data/thumbs/{thumb_path.name}" if thumb_path.exists() else None,
        "status": "idle",
        "error": None,
        "prompts": {"positive": ""},
        "advanced": AdvancedSettings().model_dump(),
    }
    store.add(item)
    return {"item": item}

def _run_restore_background(item_id: str, prompts: Prompts, advanced: AdvancedSettings):
    entry = store.get(item_id)
    if not entry:
        return
    try:
        store.update(item_id, {"status": "running", "error": None, "prompts": prompts.model_dump(), "advanced": advanced.model_dump()})
        input_url = entry["input_url"]
        input_path = DATA_DIR / input_url.lstrip("/")
        out_path = RESULTS_DIR / f"{item_id}.png"

        call_model_restore(input_path, out_path, prompts, advanced)

        store.update(item_id, {
            "status": "done",
            "result_url": f"/data/results/{out_path.name}",
            "error": None
        })
    except Exception as e:
        store.update(item_id, {"status": "error", "error": str(e)})

@app.post("/api/run")
async def run_restore(req: RunRequest, background: BackgroundTasks):
    entry = store.get(req.id)
    if not entry:
        raise HTTPException(status_code=404, detail="Item not found")
    background.add_task(_run_restore_background, req.id, req.prompts, req.advanced)
    return {"ok": True}

@app.get("/api/status/{item_id}")
async def get_status(item_id: str):
    entry = store.get(item_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"status": entry["status"], "error": entry.get("error"), "result_url": entry.get("result_url")}

@app.delete("/api/history")
async def delete_history(payload: DeleteRequest):
    ids = payload.ids or []
    for item_id in ids:
        entry = store.get(item_id)
        if entry:
            for key in ["input_url", "result_url", "thumb_url"]:
                url = entry.get(key)
                if url:
                    path = DATA_DIR / url.lstrip("/")
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception:
                        pass
    store.delete_many(ids)
    return {"deleted": ids}

# If RESTORE_API_URL is not set, the container entrypoint/run.sh will set it to the local API (http://127.0.0.1:8000).
# call_model_restore already uses RESTORE_API_URL to call the API when set.
