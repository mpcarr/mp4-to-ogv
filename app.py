import asyncio
import json
import uuid
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("static/index.html").read_text(encoding="utf-8")


def ffmpeg_args_for_quality(q: int) -> list[str]:
    """
    Theora/Vorbis quality scale:
      video: 0–10 (higher = better)
      audio: 0–10
    We accept 1..10 from UI.
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
    """
    Return duration in milliseconds using ffprobe JSON output.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_entries", "format=duration",
        str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Is ffmpeg/ffprobe installed and on PATH?")
    except subprocess.CalledProcessError as e:
        msg = (e.output or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffprobe failed: {msg[-2000:]}")

    data = json.loads(raw)
    dur = data.get("format", {}).get("duration", None)
    if dur is None:
        raise RuntimeError("ffprobe returned no duration.")
    seconds = float(dur)
    if seconds <= 0:
        raise RuntimeError(f"ffprobe duration invalid: {dur}")
    return max(1, int(seconds * 1000))


@dataclass
class Job:
    job_id: str
    status: str = "queued"          # queued | processing | done | error
    percent: int = 0               # 0..100
    message: str = "Queued"
    queue_pos: int = 0             # 1..N while queued, 0 once processing
    out_path: Optional[Path] = None
    err_detail: Optional[str] = None
    duration_ms: int = 1
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # required for worker
    in_path: Optional[Path] = None
    quality: int = 7


JOBS: Dict[str, Job] = {}

# Single global FIFO queue of job_ids
JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
WORKER_STARTED = False


def run_ffmpeg_blocking(job: Job, loop: asyncio.AbstractEventLoop):
    """
    Run ffmpeg in a normal blocking subprocess and parse -progress output.
    Push updates back into the asyncio world via loop.call_soon_threadsafe.
    Only called by the single worker (so conversions are serialized).
    """
    assert job.in_path is not None
    assert job.out_path is not None

    in_path = job.in_path
    out_path = job.out_path
    quality = job.quality

    def emit(payload: dict):
        loop.call_soon_threadsafe(job.queue.put_nowait, payload)

    try:
        job.status = "processing"
        job.queue_pos = 0
        job.message = "Starting ffmpeg..."
        emit({"type": "status", "status": job.status, "percent": job.percent, "message": job.message, "queue_pos": 0})

        job.duration_ms = probe_duration_ms(in_path)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            *ffmpeg_args_for_quality(quality),
            "-progress", "pipe:1",
            "-nostats",
            str(out_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        last_percent = -1
        assert proc.stdout is not None

        for line in proc.stdout:
            s = line.strip()
            if not s or "=" not in s:
                continue

            k, v = s.split("=", 1)

            if k == "out_time_ms":
                try:
                    out_ms = int(v)
                except ValueError:
                    continue

                pct = int(min(100, max(0, (out_ms / job.duration_ms) * 100)))
                if pct != last_percent:
                    last_percent = pct
                    job.percent = pct
                    emit({"type": "progress", "status": "processing", "percent": pct})

            elif k == "progress" and v == "end":
                break

        rc = proc.wait()

        if rc != 0:
            err = ""
            if proc.stderr:
                err = proc.stderr.read() or ""
            job.status = "error"
            job.err_detail = err[-4000:] if err else "ffmpeg failed"
            emit({
                "type": "error",
                "status": "error",
                "percent": job.percent,
                "message": "ffmpeg failed",
                "detail": job.err_detail,
            })
            return

        job.status = "done"
        job.percent = 100
        job.message = "Conversion complete"
        emit({"type": "done", "status": "done", "percent": 100, "message": job.message})

    except Exception as e:
        job.status = "error"
        job.err_detail = f"{type(e).__name__}: {repr(e)}\n\n{traceback.format_exc()}"
        emit({
            "type": "error",
            "status": "error",
            "percent": job.percent,
            "message": "Conversion error",
            "detail": job.err_detail[-4000:],
        })
    finally:
        # cleanup input
        try:
            if in_path.exists():
                in_path.unlink()
        except:
            pass


async def run_conversion(job: Job):
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(run_ffmpeg_blocking, job, loop)


async def queue_worker():
    while True:
        job_id = await JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if not job:
            JOB_QUEUE.task_done()
            continue

        # This is the only place conversions happen -> ensures one at a time
        await run_conversion(job)

        JOB_QUEUE.task_done()


@app.on_event("startup")
async def startup():
    global WORKER_STARTED
    if not WORKER_STARTED:
        WORKER_STARTED = True
        asyncio.create_task(queue_worker())


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    quality: int = Form(7),
):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Please upload an .mp4 file")

    if quality < 1 or quality > 10:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 10")

    workdir = Path("/tmp")
    job_id = str(uuid.uuid4())
    in_path = workdir / f"{job_id}.mp4"
    out_path = workdir / f"{job_id}.ogv"

    # Save upload
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    job = Job(
        job_id=job_id,
        status="queued",
        percent=0,
        message="Queued",
        queue_pos=0,        # will set below
        in_path=in_path,
        out_path=out_path,
        quality=quality,
    )
    JOBS[job_id] = job

    # Enqueue (FIFO). Queue position estimate:
    # qsize() here is number currently waiting, before adding this job.
    waiting_before = JOB_QUEUE.qsize()
    await JOB_QUEUE.put(job_id)
    job.queue_pos = waiting_before + 1

    # Emit queued status immediately so the client sees their position
    await job.queue.put({
        "type": "status",
        "status": "queued",
        "percent": 0,
        "message": "Queued",
        "queue_pos": job.queue_pos
    })

    return {"job_id": job_id}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        # Send an initial snapshot immediately
        initial = {
            "type": "status",
            "status": job.status,
            "percent": job.percent,
            "message": job.message,
            "queue_pos": job.queue_pos,
        }
        yield f"data: {json.dumps(initial)}\n\n"

        while True:
            payload = await job.queue.get()
            yield f"data: {json.dumps(payload)}\n\n"

            if payload.get("type") in {"done", "error"}:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.err_detail or "Conversion failed")

    if job.status != "done" or not job.out_path or not job.out_path.exists():
        raise HTTPException(status_code=425, detail="Not ready yet")

    return FileResponse(
        path=str(job.out_path),
        media_type="video/ogg",
        filename="converted.ogv",
    )
