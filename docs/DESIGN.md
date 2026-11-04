# Design

## Problem

Several projects on one GPU server each run one ffmpeg process per camera
(`-hwaccel cuda ... -vf fps=N,hwdownload,format=nv12,format=bgr24 ... pipe:1`).

- Each process owns a CUDA context; the context plus CUDA libraries dominate the process's VRAM
  (466 MB with `-hwaccel cuda` + CUDA filters, 286 MB with `h264_cuvid`).
- A camera read by k readers is decoded k times and uses k RTSP sessions; cameras refuse
  sessions beyond their limit.
- NV12 → BGR conversion of every delivered frame runs on the CPU in every process.

## Idea

One process decodes every camera. All decoders share the CUDA primary context, each camera is
decoded once over one RTSP session, color conversion runs on the GPU once per camera, and clients
read the latest frame from shared memory through an OpenCV-compatible API.

## Components

| | |
|---|---|
| `hub/hub.py` | The hub: one thread per camera, shared-memory publisher, `check` sub-command |
| `hub/camhub.env` | Camera list `CAM1=rtsp://...` (not in git; template `camhub.env.example`) |
| `camhub_client.py` | `HubCapture`, single file, numpy only, Python ≥ 3.8 |

Shared directory: `/dev/shm/camhub` on the host, `/camhub` inside containers.

| File | |
|---|---|
| `cameras.json` | `camN → host:port/path` (no credentials); the client resolves URLs here |
| `status.json` | Per-camera state, decode fps, output fps, clients, last error; every 5 s |
| `<cam>.<mode>.buf` | Latest frame, 3-slot ring (~37 MB per 4 MP camera per mode) |
| `want/<cam>.<mode>.<id>` | Per-client request `{"fps": N, "ack": true}`, refreshed every 2 s |
| `hub.lock` | `flock`: one hub per machine |

## Protocol

**Camera lifecycle.** A client writes a request file. The hub scans every second: a fresh request
(< 10 s old) opens the camera; 30 s without requests closes it and frees its VRAM. A dead client's
request simply expires. On error the camera thread reconnects (5 s, +3 s per attempt, max 30 s),
independently of other cameras.

**Frame buffer.** 64-byte header `magic "CHB1", version, w, h, c, nslots, frame_bytes, seq, consumed`,
followed by 3 slots `[seq u64, ts f64][pixels]`.

- Writer: `slot.seq = 0` → copy pixels → `slot.ts` → `slot.seq = n` → `header.seq = n`.
- Reader: `n = header.seq`, copy slot `n % 3`, accept only if `slot.seq` is still `n`.
- A size change or hub restart creates a new file (new inode); clients reopen it.

**Frame pacing.** For each decoded frame and each mode the hub decides whether to convert it:

- clients asking `fps = N`: wall-clock pacing at the largest N; each client re-filters to its own N.
  Both stages accept frames up to half a period early — camera frames arrive with jitter, and a
  strict comparison turns 25 → 20 fps into ~17 fps.
- clients asking `fps = 0` (every frame, OpenCV-style sequential reads): a new frame is converted
  only once the previous one has been taken (the client writes `consumed`). A slow consumer causes
  only as many conversions as it can read. Old clients without `ack` get every frame.

## Color

Cameras deliver full-range YUV 4:2:0.

- `tv`: LUT full → limited range, then the exact integer arithmetic of OpenCV's
  `COLOR_YUV2BGR_NV12` (BT.601 coefficients << 20, nearest chroma) — bit-exact with
  `colorspace_cuda=range=tv` + `cv2.cvtColor`.
- `bgr24`: BT.601 full range, bicubic chroma upsampling (table-driven; 16 variants compared with
  `hub.py check`, all within 0.6–0.7 / 255 of swscale — the remainder is swscale's internal
  rounding, not interpolation).

## Decisions

| Decision | Reason |
|---|---|
| PyAV `HWAccel("cuda", options={"primary_ctx": "1"})` | All decoders share the CUDA primary context; impossible with one ffmpeg CLI process per camera |
| `is_hw_owned=True` | Frames stay on the GPU; only selected frames are copied to RAM. Copying all 25 fps dropped 6 cameras to 21–23 fps |
| `allow_software_fallback=False` | NVDEC failures surface instead of silently decoding on the CPU |
| Color conversion in the hub | Clients get plain BGR, no GPU code in projects, one conversion per camera |
| cupy `RawKernel` on PyAV's device pointers | No DLPack/stream plumbing; same primary context as the decoders |
| Create an FFmpeg `HWDevice` before initialising cupy | FFmpeg requires the primary context with `BLOCKING_SYNC`; if cupy creates it first (default flags) every later decoder fails with `Operation not supported` |
| GPU color failure → CPU fallback | A GPU issue never costs clients their frames |
| Shared memory, not CUDA IPC | Simple and container-friendly; IPC's extra saving is not worth the complexity |
| OpenCV-compatible client | Swap exactly where the camera is opened; reconnect, queues and logic stay untouched |
| Generic camera names, URL lookup | Not tied to any project; projects keep their own URLs |
| Compose mounts the whole directory | Editors / rsync replace files with new inodes; single-file bind mounts keep showing the old file |
| `flock` taken before anything else | A second hub started by mistake exits before it can clean up the running hub's buffers |

## Limitations

- Single point of failure; clients reconnect automatically after a hub restart.
- One GPU / one NVDEC engine.
- Frames go GPU → RAM → GPU; no zero-copy across processes.
- Decoding only; encoding stays in the projects.
- Python client only.
