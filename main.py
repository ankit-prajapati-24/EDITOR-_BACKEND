# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import socketio
import threading
import uuid
import json
import asyncio

from final.generate_video_final import create_video_from_image_and_audio

# -------------------------
# Socket.IO Setup
# -------------------------
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*"   # ✅ important
)

fastapi_app = FastAPI()

# CORS (ONLY here)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Final ASGI app (IMPORTANT)
app = socketio.ASGIApp(sio, fastapi_app)

# -------------------------
# Globals
# -------------------------
active_tasks = {}
SERVER_LOOP = None
PROGRESS_QUEUE = None
PROGRESS_WORKER = None

# -------------------------
# Socket Events
# -------------------------
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    global SERVER_LOOP, PROGRESS_QUEUE, PROGRESS_WORKER

    if SERVER_LOOP is None:
        SERVER_LOOP = asyncio.get_running_loop()

    if PROGRESS_QUEUE is None:
        PROGRESS_QUEUE = asyncio.Queue()

    if PROGRESS_WORKER is None or PROGRESS_WORKER.done():
        PROGRESS_WORKER = asyncio.create_task(_progress_worker())


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


# -------------------------
# Startup
# -------------------------
@fastapi_app.on_event("startup")
async def startup_event():
    global SERVER_LOOP, PROGRESS_QUEUE, PROGRESS_WORKER

    SERVER_LOOP = asyncio.get_running_loop()

    if PROGRESS_QUEUE is None:
        PROGRESS_QUEUE = asyncio.Queue()

    if PROGRESS_WORKER is None or PROGRESS_WORKER.done():
        PROGRESS_WORKER = asyncio.create_task(_progress_worker())


# -------------------------
# Progress Worker
# -------------------------
async def _progress_worker():
    while True:
        payload = await PROGRESS_QUEUE.get()
        await sio.emit('progress', payload)


def _emit_progress(payload: Dict[str, Any]):
    loop = SERVER_LOOP
    if loop is None:
        print("No loop found")
        return

    def _put():
        if PROGRESS_QUEUE:
            PROGRESS_QUEUE.put_nowait(payload)

    loop.call_soon_threadsafe(_put)


# -------------------------
# Routes
# -------------------------
@fastapi_app.get("/")
def read_root():
    return {"message": "API is running!"}


# -------------------------
# File Upload
# -------------------------
@fastapi_app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "outputs"

ASSETS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve output videos
fastapi_app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")


# -------------------------
# Models
# -------------------------
class VideoSettings(BaseModel):
    duration: Optional[float] = 10
    cinematic: Optional[bool] = True
    transition_duration: Optional[float] = 0.6
    transition_type: Optional[str] = "fade"
    motion_type: Optional[str] = "zoom_in"
    motion_speed: Optional[float] = 1.0
    layout_mode: Optional[str] = "blur_bg"
    frame_size: Optional[List[int]] = Field(default_factory=lambda: [1080, 1920])


class GenerateVideoRequest(BaseModel):
    images: List[Union[str, Dict[str, Any]]]
    audio_path: str
    output_name: Optional[str] = "output.mp4"
    settings: Optional[VideoSettings] = VideoSettings()


# -------------------------
# Helpers
# -------------------------
def _resolve_path(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (ASSETS_DIR / path).resolve())


def _normalize_images(images):
    fixed = []
    for entry in images:
        if isinstance(entry, dict):
            entry["path"] = _resolve_path(entry["path"])
            fixed.append(entry)
        else:
            fixed.append(_resolve_path(entry))
    return fixed


# -------------------------
# Video Generation Thread
# -------------------------
def run_video_generation(task_id, image_paths, audio_path, output_path, settings):

    def progress_callback(message):
        percent = None
        if "(25%)" in message: percent = 25
        elif "(50%)" in message: percent = 50
        elif "(75%)" in message: percent = 75
        elif "(100%)" in message: percent = 100

        _emit_progress({
            "task_id": task_id,
            "message": message,
            "percent": percent
        })

    try:
        create_video_from_image_and_audio(
            image_paths=image_paths,
            audio_path=audio_path,
            output_path=output_path,
            progress_callback=progress_callback,
            **settings
        )

        _emit_progress({
            "task_id": task_id,
            "message": "Completed",
            "percent": 100,
            "completed": True,
            "output_path": str(output_path)
        })

    except Exception as e:
        _emit_progress({
            "task_id": task_id,
            "message": str(e),
            "error": True
        })

    finally:
        active_tasks.pop(task_id, None)


# -------------------------
# Generate API
# -------------------------
@fastapi_app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    task_id = str(uuid.uuid4())

    image_paths = _normalize_images(req.images)
    audio_path = _resolve_path(req.audio_path)
    output_path = (OUTPUT_DIR / req.output_name).resolve()

    settings = req.settings.model_dump()

    active_tasks[task_id] = threading.Thread(
        target=run_video_generation,
        args=(task_id, image_paths, audio_path, output_path, settings)
    )
    active_tasks[task_id].start()

    return {
        "status": "started",
        "task_id": task_id
    }
