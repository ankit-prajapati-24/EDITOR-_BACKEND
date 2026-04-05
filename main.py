# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
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

# Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
fastapi_app = FastAPI()

# Wrap FastAPI with Socket.IO
socket_app = socketio.ASGIApp(sio, fastapi_app)

# CORS (allow all origins for local dev)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    global SERVER_LOOP
    if SERVER_LOOP is None:
        SERVER_LOOP = asyncio.get_running_loop()
    global PROGRESS_QUEUE, PROGRESS_WORKER
    if PROGRESS_QUEUE is None:
        PROGRESS_QUEUE = asyncio.Queue()
    if PROGRESS_WORKER is None or PROGRESS_WORKER.done():
        PROGRESS_WORKER = asyncio.create_task(_progress_worker())

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# Store active tasks
active_tasks = {}
SERVER_LOOP = None
PROGRESS_QUEUE = None
PROGRESS_WORKER = None

# Home route
@fastapi_app.get("/")
def read_root():
    return {"message": "API is running!"}

# Example: image upload route
@fastapi_app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # file.filename me file ka name milega
    # file.content_type me file type (image/png etc) milega
    contents = await file.read()  # bytes me file ka data
    # yahan tu Cloudinary me upload kar sakta hai
    return {"filename": file.filename, "size": len(contents)}


# -------------------------
# Generate video endpoint
# -------------------------
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve generated videos
fastapi_app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")


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
    output_name: Optional[str] = "final_output.mp4"
    settings: Optional[VideoSettings] = VideoSettings()


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    # default to assets for relative paths
    return str((ASSETS_DIR / path).resolve())


def _normalize_images(images: List[Union[str, Dict[str, Any]]]) -> List[Union[str, Dict[str, Any]]]:
    fixed: List[Union[str, Dict[str, Any]]] = []
    for entry in images:
        if isinstance(entry, dict):
            new_entry = dict(entry)
            if "path" not in new_entry:
                raise HTTPException(status_code=400, detail="Each image dict must include 'path'.")
            new_entry["path"] = _resolve_path(str(new_entry["path"]))
            fixed.append(new_entry)
        else:
            fixed.append(_resolve_path(str(entry)))
    return fixed


@fastapi_app.on_event("startup")
async def _store_loop():
    loop = asyncio.get_running_loop()
    fastapi_app.state.loop = loop
    global SERVER_LOOP
    SERVER_LOOP = loop
    # Start a single worker that emits Socket.IO events from this loop
    global PROGRESS_QUEUE, PROGRESS_WORKER
    if PROGRESS_QUEUE is None:
        PROGRESS_QUEUE = asyncio.Queue()
    if PROGRESS_WORKER is None or PROGRESS_WORKER.done():
        PROGRESS_WORKER = asyncio.create_task(_progress_worker())


async def _progress_worker():
    while True:
        payload = await PROGRESS_QUEUE.get()
        await sio.emit('progress', payload)


def _emit_progress(payload: Dict[str, Any]):
    loop = getattr(fastapi_app.state, "loop", None) or SERVER_LOOP
    if loop is None:
        # No loop available yet; log so we can diagnose missing callbacks
        print("[socket] No event loop available; progress not emitted.")
        return

    def _put():
        if PROGRESS_QUEUE is not None:
            PROGRESS_QUEUE.put_nowait(payload)

    # Thread-safe handoff to the ASGI event loop
    loop.call_soon_threadsafe(_put)


def run_video_generation(task_id, image_paths, audio_path, output_path, settings):
    def progress_callback(message):
        print(f"Progress update for task {task_id}: {message}")
        # Extract percentage if present
        percent = None
        if "(25%)" in message:
            percent = 25
        elif "(50%)" in message:
            percent = 50
        elif "(75%)" in message:
            percent = 75
        elif "(85%)" in message:
            percent = 85
        elif "(90%)" in message:
            percent = 90
        elif "(100%)" in message:
            percent = 100
        elif "Rendering:" in message:
            import re
            match = re.search(r'Rendering:\s*(\d+)%', message)
            if match:
                percent = int(match.group(1))
        
        _emit_progress({
            'task_id': task_id,
            'message': message,
            'percent': percent
        })
    
    try:
        create_video_from_image_and_audio(
            image_paths=image_paths,
            audio_path=audio_path,
            output_path=output_path,
            progress_callback=progress_callback,
            **settings,
        )
        _emit_progress({
            'task_id': task_id,
            'message': 'Video generation completed!',
            'percent': 100,
            'completed': True,
            'output_path': str(output_path)
        })
    except Exception as e:
        _emit_progress({
            'task_id': task_id,
            'message': f'Error: {str(e)}',
            'error': True
        })
    finally:
        # Remove from active tasks
        active_tasks.pop(task_id, None)


@fastapi_app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):
    try:
        task_id = str(uuid.uuid4())
        image_paths = _normalize_images(req.images)
        audio_path = _resolve_path(req.audio_path)
        output_path = (OUTPUT_DIR / req.output_name).resolve()

        settings = req.settings.model_dump() if req.settings else {}
        if "frame_size" in settings and isinstance(settings["frame_size"], list):
            settings["frame_size"] = tuple(settings["frame_size"])

        # Start background task
        active_tasks[task_id] = threading.Thread(
            target=run_video_generation,
            args=(task_id, image_paths, audio_path, output_path, settings)
        )
        active_tasks[task_id].start()

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Video generation started. Monitor progress via Socket.IO."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Generate video from uploads (FormData)
# -------------------------
@fastapi_app.post("/generate-video-upload")
async def generate_video_upload(
    images: List[UploadFile] = File(...),
    audio: UploadFile = File(...),
    output_name: Optional[str] = Form(None),
    settings_json: Optional[str] = Form(None),
    plan_json: Optional[str] = Form(None),
):
    try:
        task_id = str(uuid.uuid4())
        
        # Save uploaded images with dynamic names
        saved_images: List[str] = []
        saved_by_name: Dict[str, str] = {}
        for img in images:
            ext = Path(img.filename).suffix or ".png"
            safe_name = Path(img.filename).name
            fname = f"{uuid.uuid4().hex}_{safe_name}"
            img_path = ASSETS_DIR / fname
            content = await img.read()
            img_path.write_bytes(content)
            saved_path = str(img_path.resolve())
            saved_images.append(saved_path)
            saved_by_name[safe_name] = saved_path

        # Save uploaded audio with dynamic name
        audio_ext = Path(audio.filename).suffix or ".mp3"
        audio_name = f"audio_{uuid.uuid4().hex}{audio_ext}"
        audio_path = ASSETS_DIR / audio_name
        audio_bytes = await audio.read()
        audio_path.write_bytes(audio_bytes)

        # Output file name
        if not output_name:
            output_name = f"output_{uuid.uuid4().hex}.mp4"
        output_path = (OUTPUT_DIR / output_name).resolve()

        # Settings (optional JSON)
        settings: Dict[str, Any] = {}
        if settings_json:
            settings = json.loads(settings_json)
            if "frame_size" in settings and isinstance(settings["frame_size"], list):
                settings["frame_size"] = tuple(settings["frame_size"])

        # Optional AI plan (per-image timing/transitions)
        image_paths: List[Union[str, Dict[str, Any]]] = saved_images
        if plan_json:
            plan = json.loads(plan_json)
            plan_images = plan.get("images", [])
            mapped: List[Union[str, Dict[str, Any]]] = []
            for entry in plan_images:
                if isinstance(entry, dict):
                    name = Path(str(entry.get("path", ""))).name
                    if not name or name not in saved_by_name:
                        raise HTTPException(status_code=400, detail=f"Plan image not found: {name}")
                    new_entry = dict(entry)
                    new_entry["path"] = saved_by_name[name]
                    mapped.append(new_entry)
                else:
                    name = Path(str(entry)).name
                    if not name or name not in saved_by_name:
                        raise HTTPException(status_code=400, detail=f"Plan image not found: {name}")
                    mapped.append(saved_by_name[name])
            image_paths = mapped

        # Start background task
        active_tasks[task_id] = threading.Thread(
            target=run_video_generation,
            args=(task_id, image_paths, str(audio_path.resolve()), str(output_path), settings)
        )
        active_tasks[task_id].start()

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Video generation started. Monitor progress via Socket.IO.",
            "saved_images": saved_images,
            "saved_audio": str(audio_path.resolve()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For uvicorn to run the Socket.IO wrapped app
app = socket_app

