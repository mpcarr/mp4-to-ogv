import asyncio
import json
import uuid
import subprocess
import traceback
import re
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

# ---------------------------------------------------------------------------
# PyInstaller helpers
# ---------------------------------------------------------------------------

def resource_path(*parts: str) -> Path:
    """
    Get absolute path to resource (works in dev + PyInstaller onefile).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).parent
    return base.joinpath(*parts)


def resolve_ffmpeg_binaries():
    """
    Use bundled ffmpeg if running from PyInstaller,
    otherwise rely on system ffmpeg.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        return (
            str(base / "ffmpeg" / "ffmpeg.exe"),
            str(base / "ffmpeg" / "ffprobe.exe"),
        )
    return "ffmpeg", "ffprobe"


FFMPEG_BIN, FFPROBE_BIN = resolve_ffmpeg_binaries()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

# Static files (optional; for completeness)
static_dir = resource_path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return resource_path("static", "index.html").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def ffmpeg_args_for_quality(q: int) -> list[str]:
    """
    Theora/Vorbis quality scale:
      video: 0–10 (higher = better)
      audio: 0–10
    UI sends 1..10.
    """
    q = max(1, min(10, q))
    qa = max(0, q - 2)
    return [
        "-c:v", "libtheora",
        "-q:v", str(q),
        "-c:a", "libvorbis",
        "-q:a", str(qa),
    ]


def probe_duration_ms(path: Path) -> int:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_entries", "format=duration",
        str(path),
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
    data = json.loads(raw)
    dur = data.get("format", {}).get("duration")
    if not dur:
        raise RuntimeError("Could not determine duration")
    return max(1, int(float(dur) * 1000))


def parse_ffmpeg_time_to_ms(t: str) -> Optional[int]:
    try:
        hh, mm, ss = t.split(":")
        return int((int(hh) * 3600 + int(mm) * 60 + float(ss)) * 1000)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Job model + queue
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    status: str = "queued"          # queued | processing | done | error
    percent: int = 0               # 0..100
    message: str = "Queued"
    queue_pos: int = 0
    in_path: Optional[Path] = None
    out_path: Optional[Path] = None
    quality: int = 7
    err_detail: Optional[str] = None
    duration_ms: int = 1
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # For download naming
    original_stem: str = "converted"


JOBS: Dict[str, Job] = {}
JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
WORKER_STARTED = False

# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def cleanup_job_files(job_id: str):
    """
    Delete input/output files and forget the job.
    Called after download completes (BackgroundTasks).
    """
    job = JOBS.pop(job_id, None)
    if not job:
        return

    try:
        if job.in_path and job.in_path.exists():
            job.in_path.unlink()
    except Exception:
        pass

    try:
        if job.out_path and job.out_path.exists():
            job.out_path.unlink()
    except Exception:
        pass

    # If tmp folder is empty, optionally remove it (best-effort)
    try:
        tmp_dir = Path.cwd() / "tmp"
        if tmp_dir.exists() and tmp_dir.is_dir():
            if not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Conversion worker
# ---------------------------------------------------------------------------

def run_ffmpeg_blocking(job: Job, loop: asyncio.AbstractEventLoop):
    def emit(payload: dict):
        loop.call_soon_threadsafe(job.queue.put_nowait, payload)

    try:
        job.status = "processing"
        job.queue_pos = 0
        emit({"type": "status", "status": "processing", "percent": 0})

        if not job.in_path or not job.out_path:
            raise RuntimeError("Job missing input/output path")

        job.duration_ms = probe_duration_ms(job.in_path)

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(job.in_path),
            *ffmpeg_args_for_quality(job.quality),
            "-stats_period", "0.5",
            str(job.out_path),
        ]

        time_re = re.compile(r"time=(\d+:\d+:\d+(?:\.\d+)?)")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        last_pct = -1
        if proc.stderr:
            for line in proc.stderr:
                m = time_re.search(line)
                if not m:
                    continue
                out_ms = parse_ffmpeg_time_to_ms(m.group(1))
                if out_ms is None:
                    continue

                pct = int(min(99, max(0, (out_ms / job.duration_ms) * 100)))
                if pct != last_pct:
                    last_pct = pct
                    job.percent = pct
                    emit({"type": "progress", "status": "processing", "percent": pct})

        rc = proc.wait()
        if rc != 0:
            job.status = "error"
            job.err_detail = f"ffmpeg exited with code {rc}"
            emit({"type": "error", "status": "error", "percent": job.percent, "detail": job.err_detail})
            return

        job.status = "done"
        job.percent = 100
        emit({"type": "done", "status": "done", "percent": 100})

    except Exception:
        job.status = "error"
        job.err_detail = traceback.format_exc()
        emit({
            "type": "error",
            "status": "error",
            "percent": job.percent,
            "detail": (job.err_detail or "")[-4000:],
        })


async def run_conversion(job: Job):
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(run_ffmpeg_blocking, job, loop)


async def queue_worker():
    while True:
        job_id = await JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job:
            await run_conversion(job)
        JOB_QUEUE.task_done()


@app.on_event("startup")
async def startup():
    global WORKER_STARTED
    if not WORKER_STARTED:
        WORKER_STARTED = True
        asyncio.create_task(queue_worker())

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    quality: int = Form(7),
):
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Only MP4 files supported")
    if quality < 1 or quality > 10:
        raise HTTPException(400, "Quality must be between 1 and 10")

    # Make a safe-ish stem for download filename
    orig_name = Path(file.filename).name
    print(orig_name)
    stem = Path(orig_name).stem or "converted"
    print(stem)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip() or "converted"
    print(stem)
    # Use a local tmp folder so it works consistently in PyInstaller
    workdir = Path.cwd() / "tmp"
    workdir.mkdir(exist_ok=True)

    job_id = str(uuid.uuid4())
    in_path = workdir / f"{job_id}.mp4"
    out_path = workdir / f"{job_id}.ogv"

    with open(in_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    job = Job(
        job_id=job_id,
        in_path=in_path,
        out_path=out_path,
        quality=quality,
        original_stem=stem,
    )
    JOBS[job_id] = job
    print(job)
    pos = JOB_QUEUE.qsize() + 1
    job.queue_pos = pos
    await JOB_QUEUE.put(job_id)

    await job.queue.put({
        "type": "status",
        "status": "queued",
        "queue_pos": pos,
    })

    return {"job_id": job_id}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404)

    async def stream():
        yield f"data: {json.dumps({'type':'status','status':job.status,'queue_pos':job.queue_pos,'percent':job.percent})}\n\n"
        while True:
            msg = await job.queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") in {"done", "error"}:
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/download")
async def download(job_id: str, background_tasks: BackgroundTasks):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status == "error":
        raise HTTPException(500, job.err_detail or "Conversion failed")

    if job.status != "done" or not job.out_path or not job.out_path.exists():
        raise HTTPException(425, "Not ready yet")

    download_name = f"{job.original_stem}.ogv"
    print(download_name)
    # Cleanup files and remove job after the response is sent
    background_tasks.add_task(cleanup_job_files, job_id)

    return FileResponse(
        path=str(job.out_path),
        filename=download_name,
        media_type="video/ogg",
    )

# ---------------------------------------------------------------------------
# Run as executable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
