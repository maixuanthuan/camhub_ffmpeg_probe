# Benchmark

Hardware: 1× RTX 3060 12 GB (one NVDEC engine). Cameras: 2688×1520, HEVC and H.264, 25 fps, RTSP over TCP.
VRAM from `nvidia-smi`; CPU from `docker stats` (30 s average unless noted).

## Baseline: one ffmpeg per camera per project

| Process | VRAM |
|---|---|
| ffmpeg decode, `-hwaccel cuda` + CUDA filters | 466 MB |
| ffmpeg decode, `h264_cuvid` | 286 MB |
| ffmpeg encode, NVENC | 133 MB |

Observed on the shared server before camhub: 15 ffmpeg processes holding ~5.8 GB of 12 GB VRAM;
6 cameras decoded 9 times on the GPU (~3.9 GB, ~229% CPU) plus 3 times on the CPU; one camera
refusing its 4th RTSP session.

## One process, shared CUDA context (`tools/camhub_poc.py`)

| Cameras | camhub process | One ffmpeg per camera |
|---|---|---|
| 1 | 290 MB | 466 MB |
| 2 | 472 MB | 932 MB |
| 3 | 654–684 MB | 1398 MB |
| 6 | 1232 MB | 2796 MB |

~190 MB per additional camera. Decoding holds 25 fps with 6 cameras when frames stay on the GPU
(copying every frame to RAM: 21–23 fps).

## Color accuracy (same recorded clip, pixel by pixel)

| | Mean abs diff (0–255) | Pixels off by > 2 |
|---|---|---|
| camhub `tv` vs ffmpeg `colorspace_cuda=range=tv` + OpenCV | 0.000 | 0% |
| camhub `bgr24` (CPU) vs ffmpeg `format=bgr24` | 0.31 | 2.2% |
| camhub `bgr24` (GPU) vs camhub `bgr24` (CPU) | 0.61–0.74 | 1.1–2.5% |
| reference: ffmpeg `bgr24` vs ffmpeg `tv` + OpenCV | 0.95 | 3.6% |

## Color conversion cost, one 2688×1520 frame (CPU time consumed)

| | CPU | GPU |
|---|---|---|
| `bgr24` | 19–21 ms | 11 ms |
| `tv` | 57–62 ms | 7–8 ms |

## Production: three projects moved to camhub

| Project | Cameras | Request |
|---|---|---|
| A | 1 | `tv`, 20 fps |
| B | 2 | `bgr24`, 12 fps |
| C | 3 (2 shared with B) | `bgr24`, every frame |

CPU:

| Container | Before | camhub, CPU color | camhub, GPU color |
|---|---|---|---|
| camhub | — | 219% | 35.5% |
| C | 200% (decoding on CPU) | 146% | 156% |
| B | 70% | 73% | 72% |
| A | 65% | 65% | 57% |
| **Total** | **~486%** | **~503%** | **~321%** |

- Consolidating decoding while still converting colors on the CPU does not reduce CPU: the
  conversion is the dominant cost and only moves from the projects to the hub.
- Converting on the GPU reduces total CPU by ~34%.
- Project C processes more frames than before (~16 → ~18 fps per camera) because the hub delivers
  frames on demand; its own CPU rises accordingly.
- VRAM: camhub 819 MB for 4 cameras (including cupy) vs 1218 MB for the 3 ffmpeg decoders of A and B before.
- Frame rates unchanged: A camera 20.0 → 19.9–20.0 fps, AI 14.6 → 14.7–14.9 fps.

## Caveats

- The "Before" CPU column is a single sample; later columns are 30 s averages. AI load varies
  with scene activity — long averages (`tools/camhub_monitor.sh`) are more reliable.
- Short production run so far; long-term stability not yet measured.
