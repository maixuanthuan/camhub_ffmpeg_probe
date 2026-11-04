# camhub

**Decode every camera once, on the GPU — share the frames with every project.**

camhub is a small service for a single-GPU server running several computer-vision projects
against the same RTSP cameras. One process decodes all cameras with NVDEC, converts them to BGR on
the GPU, and publishes the latest frame of each camera to shared memory. Projects read frames
through a drop-in replacement for `cv2.VideoCapture` instead of running their own decoders.

```
RTSP cameras ──► camhub (1 process, 1 CUDA context)
                   ├─ PyAV + NVDEC, frames stay on the GPU
                   ├─ NV12 → BGR on the GPU (CUDA kernel), only frames someone asked for
                   └─ latest frame per camera → /dev/shm/camhub
                              │
            ┌─────────────────┼─────────────────┐
        project A         project B         project C
        HubCapture        HubCapture        HubCapture      (one Python file, numpy only)
```

## Why

The usual setup — every project spawns one `ffmpeg -hwaccel cuda ...` per camera and reads BGR
from a pipe — scales badly:

- **Every ffmpeg process creates its own CUDA context.** Most of the ~300–470 MB of VRAM a decoding
  ffmpeg holds is that context, not frames. N cameras × M projects = N×M contexts.
- **The same camera is decoded several times** when several projects (or several features of one
  project) read it.
- **Cameras limit concurrent RTSP sessions**; duplicate readers get refused.
- **NV12 → BGR conversion runs on the CPU** in every process — for 4 MP frames this is the most
  expensive part of reading a camera, even with hardware decoding.

camhub shares one CUDA context across all decoders, decodes each camera once over a single RTSP
session, and does the color conversion on the GPU.

## Results

Measured on an RTX 3060 12 GB with 2688×1520 HEVC/H.264 cameras at 25 fps
(details: [docs/BENCHMARK.md](docs/BENCHMARK.md)).

| | One ffmpeg per camera | camhub |
|---|---|---|
| VRAM per camera | 286–466 MB (per project) | ~190 MB (shared, +~300 MB once) |
| CPU per BGR frame (4 MP) | 19–21 ms (swscale) | 11 ms |
| RTSP sessions per camera | one per reader | 1 |

Three existing projects (4 cameras) moved to camhub: total CPU of the involved containers
**~486% → ~321%**, frame rates unchanged, no change to the projects' processing logic.

## Requirements

- NVIDIA GPU with NVDEC, recent driver (tested with a CUDA 13.x driver)
- Docker with the NVIDIA Container Toolkit
- Cameras reachable over RTSP (H.264 / HEVC)
- Client side: Python ≥ 3.8 and `numpy`

## Quick start — run the hub

```bash
git clone <this repo> camhub && cd camhub/hub
cp camhub.env.example camhub.env        # one line per camera: CAM1=rtsp://user:pass@host:port/path
docker compose up -d --build
docker logs -f camhub
```

`camhub.env` holds camera credentials and is git-ignored. The hub re-reads it automatically when it
changes — adding a camera needs no restart.

The hub only connects to a camera while at least one client is reading it, and disconnects after
30 s without readers. Only one hub can run per machine (file lock).

## Use it from a project — 3 steps

**1. Mount the shared directory** in the project's `docker-compose.yml`:

```yaml
    volumes:
      - /dev/shm/camhub:/camhub
```

(Processes running directly on the host need nothing — the client finds `/dev/shm/camhub`.)

**2. Copy [`camhub_client.py`](camhub_client.py)** into the project. Single file, depends only on `numpy`.

**3. Replace the place where frames are read from the camera:**

```python
from camhub_client import HubCapture

cap = HubCapture("rtsp://user:pass@host:554/live")   # same URL as before, or "cam1"
ok, frame = cap.read()                                 # numpy uint8, H×W×3, BGR — like cv2
```

```python
# before                                     # after
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)   cap = HubCapture(url, cv2.CAP_FFMPEG)
cap = MyFfmpegReader(url, fps=8)              cap = HubCapture(url, fps=8)
```

Everything downstream stays the same:

- frames are identical in type and layout to `cv2.VideoCapture` output;
- `read()`, `grab()`, `retrieve()`, `isOpened()`, `release()`, `get(CAP_PROP_FRAME_WIDTH / HEIGHT / FPS)`
  behave like OpenCV; `set()` is accepted and ignored;
- a lost camera makes `read()` return `(False, None)` after `read_timeout`, so existing reconnect logic keeps working;
- every frame is a private copy — keep it, modify it, queue it.

Keeping the old reader behind a config flag makes rollback a one-line change.

## Client API

```python
HubCapture(source, api_preference=None, *, fps=0, mode="bgr24", read_timeout=10.0, open_timeout=20.0)
```

| Argument | Meaning |
|---|---|
| `source` | Full RTSP URL (matched on `host:port/path`, credentials ignored) or a camera name `"camN"` |
| `api_preference` | Ignored — lets `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` be renamed in place |
| `fps` | Frames per second to deliver, like ffmpeg's `fps=` filter. `0` = every frame, like OpenCV |
| `mode` | Color conversion, see below |
| `read_timeout` | Seconds without a new frame before `read()` returns `(False, None)` |
| `open_timeout` | Seconds to wait for the first frame; after that `isOpened()` is `False` |

Two color modes, matching the two common ffmpeg/OpenCV pipelines:

| `mode` | Equivalent to | Difference from that pipeline |
|---|---|---|
| `bgr24` (default) | `ffmpeg ... hwdownload,format=nv12,format=bgr24` / `cv2.VideoCapture` | mean 0.6–0.7 / 255 (GPU), 0.3 (CPU) |
| `tv` | `colorspace_cuda=range=tv` + `cv2.cvtColor(..., COLOR_YUV2BGR_NV12)` | bit-exact |

Quick test inside any container with `/camhub` mounted:

```bash
python3 camhub_client.py cam1 --fps 8 --seconds 20
```

## Hub configuration

Environment variables (set in `hub/docker-compose.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `CAMHUB_ENV` | `/app/hub/camhub.env` | Camera list |
| `CAMHUB_DIR` | `/camhub` | Shared directory inside the container |
| `CAMHUB_GPU_COLOR` | `1` | `0` = convert colors on the CPU |
| `CAMHUB_GPU_MODES` | `tv,bgr24` | Modes converted on the GPU (others fall back to CPU) |
| `CAMHUB_BGR24_CHROMA` | `0,0.5,cubic,-0.6` | Chroma interpolation for GPU `bgr24` (see `hub.py check`) |

If GPU color conversion fails for any reason, the hub falls back to the CPU and logs it once.

## Operations

```bash
docker logs --tail 30 camhub                        # one [TÓM TẮT] summary line per camera every minute
python3 -m json.tool /dev/shm/camhub/status.json    # per-camera state, fps, clients, last error
docker restart camhub                               # clients reconnect by themselves within seconds
docker compose run --rm camhub python3 hub.py check cam1   # GPU vs CPU color diff + timing on a live camera
```

Camera `state` in `status.json`: `idle` (nobody reading), `connecting` / `reconnecting`
(retry 5 s → 30 s), `running`, `stalled` (connected, no frames for > 5 s).

`tools/` contains the proof-of-concept used to validate the design (`camhub_poc.py`: VRAM per
camera, pixel diff against ffmpeg) and a resource logger (`camhub_monitor.sh`).

## Limitations

- **Single point of failure**: if the hub stops, every client loses frames until it is back
  (clients reconnect automatically).
- Frames travel GPU → RAM → GPU; there is no zero-copy sharing between processes.
- One NVDEC engine: four 4 MP cameras use ~40–60% of it on an RTX 3060.
- Decoding only. Encoding result streams stays in the projects.
- Python client only.

Code comments and log messages are in Vietnamese.

## Layout

```
camhub_client.py        client — the only file a project needs
hub/                    hub.py, Dockerfile, docker-compose.yml, camhub.env.example
tools/                  camhub_poc.py, camhub_monitor.sh
docs/DESIGN.md          architecture, shared-memory protocol, design decisions
docs/BENCHMARK.md       measurements
```

## License

[MIT](LICENSE)
