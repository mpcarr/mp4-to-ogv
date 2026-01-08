import os
import uuid
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve the single page
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = Path("static/index.html")
    return index_path.read_text(encoding="utf-8")

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    # Basic validation
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Please upload an .mp4 file")

    workdir = Path("/tmp")  # ephemeral but fine for conversions
    job_id = str(uuid.uuid4())
    in_path = workdir / f"{job_id}.mp4"
    out_path = workdir / f"{job_id}.ogv"

    # Save upload to disk
    try:
        with open(in_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)

        # Convert with ffmpeg
        # -c:v libtheora, -c:a libvorbis = typical OGV codecs
        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-c:v", "libtheora",
            "-q:v", "7",
            "-c:a", "libvorbis",
            "-q:a", "5",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        return FileResponse(
            path=str(out_path),
            media_type="video/ogg",
            filename="converted.ogv",
        )

    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise HTTPException(status_code=500, detail=f"ffmpeg failed:\n{stderr[-2000:]}")
    finally:
        # Best-effort cleanup (out file may be in-flight; OK if delete fails)
        try:
            if in_path.exists():
                in_path.unlink()
        except:
            pass
