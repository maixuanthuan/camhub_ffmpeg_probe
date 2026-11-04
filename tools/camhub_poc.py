# -*- coding: utf-8 -*-
"""
camhub_poc.py
=============
Script THỬ cho camhub — chưa phải hub thật. Chỉ ĐỌC camera, không ghi gì,
không đụng tới container nào đang chạy.

Trả lời 2 câu hỏi trước khi viết hub:

  1. VRAM: MỘT process giải mã N camera (dùng chung 1 CUDA context) tốn bao
     nhiêu, so với N process ffmpeg như hiện tại (~466 MB/cam)?
  2. Màu: frame giải mã kiểu hub có giống frame các project đang nhận từ
     ffmpeg không, lệch bao nhiêu mức pixel?

Camera khai trong file camhub.env (cùng thư mục), dạng:
    CAM1=rtsp://user:pass@host:port/live
    CAM2=...
Lệnh gọi camera theo tên: cam1, cam2, ...

Ba lệnh:

    vram   [cam1 cam2 ...]        giải mã live, thêm từng cam một, đo VRAM sau mỗi lần thêm
                                  (bỏ trống = tất cả cam trong camhub.env)
    record cam2 OUT.mkv           ghi 10s camera ra file (copy, KHÔNG giải mã)
    diff   CLIP.mkv               giải mã cùng 1 file bằng ffmpeg (cách cũ) và kiểu hub, so pixel

Cách giải mã kiểu hub:
    PyAV (FFmpeg bên trong) + NVDEC, tùy chọn primary_ctx=1 → mọi decoder trong
    process dùng CHUNG CUDA primary context, thay vì mỗi ffmpeg một context.

Hai chế độ màu (khớp 2 cách các project đang làm):
    bgr24 — như project dùng ffmpeg xuất bgr24 qua pipe:
            -vf ...,hwdownload,format=nv12,format=bgr24  (swscale)
    tv    — như project dùng colorspace_cuda + cv2:
            -vf ...,colorspace_cuda=range=tv,hwdownload,format=nv12
            rồi cv2.cvtColor(COLOR_YUV2BGR_NV12)
"""

import os
import sys
import time
import argparse
import threading
import subprocess

import av
import cv2
import numpy as np
from av.codec.hwaccel import HWAccel

RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "timeout":        "10000000",   # µs — mạng đứng 10s thì báo lỗi
}

STOP = threading.Event()

ENV_FILE = "camhub.env"


def load_cameras(path=ENV_FILE) -> dict:
    """Đọc camhub.env → {"cam1": url, "cam2": url, ...}. Bỏ dòng trống, dòng #, giá trị rỗng."""
    cams = {}
    if not os.path.exists(path):
        return cams
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key.upper().startswith("CAM") and key[3:].isdigit() and val:
                cams[f"cam{int(key[3:])}"] = val
    return dict(sorted(cams.items(), key=lambda kv: int(kv[0][3:])))


def resolve(names) -> list:
    """['cam1', 'cam3'] → [(tên, url)]. Bỏ trống = tất cả cam trong camhub.env."""
    cams = load_cameras()
    if not cams:
        sys.exit(f"Không thấy camera nào trong {ENV_FILE} (cần dòng dạng CAM1=rtsp://...).")
    if not names:
        return list(cams.items())
    out = []
    for n in names:
        key = n.lower()
        if key not in cams:
            sys.exit(f"'{n}' không có trong {ENV_FILE}. Đang có: {', '.join(cams)}")
        out.append((key, cams[key]))
    return out


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def mask(url):
    """Che user:pass khi in log."""
    if "@" in url and "://" in url:
        head, rest = url.split("://", 1)
        return f"{head}://***@{rest.split('@', 1)[1]}"
    return url


def make_hwaccel(cpu: bool, hw_owned: bool = False):
    if cpu:
        return None
    # allow_software_fallback=False: NVDEC không chạy được thì BÁO LỖI,
    # không lặng lẽ giải mã bằng CPU (đo sai)
    # hw_owned=True: frame giải mã xong NẰM LẠI trên GPU, chỉ frame nào được
    # chọn (theo fps) mới chép về RAM — như filter fps= đứng trước hwdownload.
    return HWAccel("cuda", allow_software_fallback=False, options={"primary_ctx": "1"},
                   is_hw_owned=hw_owned)


# ══════════════════════════════════════════════════════════════════════════
#  Chuyển màu — 2 chế độ
# ══════════════════════════════════════════════════════════════════════════

def _tv_luts(rounding: str):
    """LUT đổi full range (0-255) → tv range (Y 16-235, UV 16-240)."""
    x = np.arange(256, dtype=np.float64)
    y = x * 219.0 / 255.0 + 16.0
    c = (x - 128.0) * 224.0 / 255.0 + 128.0
    f = np.round if rounding == "round" else np.floor
    return (np.clip(f(y), 0, 255).astype(np.uint8),
            np.clip(f(c), 0, 255).astype(np.uint8))


_LUTS = {r: _tv_luts(r) for r in ("round", "floor")}


def frame_nv12(frame) -> np.ndarray:
    """Frame PyAV → numpy NV12 (H*3/2, W)."""
    if frame.format.name != "nv12":
        frame = frame.reformat(format="nv12")
    return frame.to_ndarray()


def nv12_full_to_bgr_tv(nv12: np.ndarray, rounding: str = "round") -> np.ndarray:
    """Giả lập colorspace_cuda=range=tv rồi cv2.cvtColor NV12→BGR."""
    h = nv12.shape[0] * 2 // 3
    ylut, clut = _LUTS[rounding]
    out = np.empty_like(nv12)
    out[:h] = ylut[nv12[:h]]
    out[h:] = clut[nv12[h:]]
    return cv2.cvtColor(out, cv2.COLOR_YUV2BGR_NV12)


def to_bgr(frame, mode: str, interp: str = "BICUBIC") -> np.ndarray:
    """Frame (trên GPU hoặc RAM) → BGR. Frame trên GPU được chép về ngay tại đây."""
    if mode == "bgr24":
        return frame.to_ndarray(format="bgr24", interpolation=interp)
    return nv12_full_to_bgr_tv(frame_nv12(frame))


# ══════════════════════════════════════════════════════════════════════════
#  Đo VRAM / CPU của chính process này
# ══════════════════════════════════════════════════════════════════════════

def my_vram_mb():
    """VRAM của process này theo nvidia-smi. Cần chạy container với --pid=host."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None
    me = os.getpid()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == me:
            return int(parts[1])
    return 0   # chưa có CUDA context (chưa mở cam nào)


def gpu_total_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.decoder",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        used, total, dec = [p.strip() for p in out.split(",")]
        return f"GPU {used}/{total} MB, NVDEC {dec}%"
    except Exception:
        return "GPU ?"


class CpuMeter:
    def __init__(self):
        self.t0 = time.monotonic()
        self.c0 = sum(os.times()[:2])

    def percent(self):
        t1, c1 = time.monotonic(), sum(os.times()[:2])
        p = (c1 - self.c0) / max(t1 - self.t0, 1e-6) * 100
        self.t0, self.c0 = t1, c1
        return p


# ══════════════════════════════════════════════════════════════════════════
#  [vram] — mỗi camera 1 thread, giải mã live
# ══════════════════════════════════════════════════════════════════════════

class CamTest(threading.Thread):

    def __init__(self, idx, url, fps, mode, cuvid, cpu, hw_owned):
        super().__init__(daemon=True, name=idx)
        self.idx, self.url, self.fps = idx, url, fps
        self.mode, self.cuvid, self.cpu = mode, cuvid, cpu
        self.hw_owned = hw_owned
        self.decoded = self.converted = 0
        self.info = "đang kết nối..."
        self.err = None
        self.latest = None

    def run(self):
        while not STOP.is_set():
            try:
                self._run_once()
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"
                log(f"[{self.idx}] LỖI {self.err} — thử lại sau 5s")
            for _ in range(50):
                if STOP.is_set():
                    return
                time.sleep(0.1)

    def _frames(self, container, stream):
        """Lặp frame đã giải mã. cuvid: tách demux (CPU) và decoder *_cuvid riêng."""
        if not self.cuvid:
            yield from container.decode(stream)
            return
        name = f"{stream.codec_context.name}_cuvid"
        ctx = av.CodecContext.create(name, "r", hwaccel=make_hwaccel(self.cpu, self.hw_owned))
        if stream.codec_context.extradata:
            ctx.extradata = stream.codec_context.extradata
        for pkt in container.demux(stream):
            yield from ctx.decode(pkt)

    def _run_once(self):
        container = av.open(
            self.url, options=RTSP_OPTS, timeout=(20, 10),
            hwaccel=None if self.cuvid else make_hwaccel(self.cpu, self.hw_owned),
        )
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            cc = stream.codec_context
            if self.cpu:
                dec = f"{cc.name} (CPU)"
            else:
                dec = f"{cc.name}_cuvid" if self.cuvid else f"{cc.name} (nvdec hwaccel)"
            self.info = f"{cc.name} {cc.width}x{cc.height} | decoder={dec}"
            log(f"[{self.idx}] mở {mask(self.url)} | {self.info}")
            self.err = None

            period, next_t = (1.0 / self.fps if self.fps else 0.0), None
            for frame in self._frames(container, stream):
                if STOP.is_set():
                    return
                self.decoded += 1
                # lọc fps theo đồng hồ thật (camera live). Không dùng frame.time:
                # frame chép từ GPU về có thể mất pts → lọc sai (lần chạy trước out=0fps)
                t = time.monotonic()
                if period:
                    if next_t is not None and t < next_t:
                        continue
                    next_t = t + period if next_t is None or t - next_t > period else next_t + period
                img = to_bgr(frame, self.mode)
                if self.latest is None or self.latest.shape != img.shape:
                    self.latest = np.empty_like(img)
                np.copyto(self.latest, img)   # giả lập chép vào shared memory
                self.converted += 1
        finally:
            container.close()


def cmd_vram(a):
    urls = resolve(a.cams)
    log(f"PID={os.getpid()} | {len(urls)} camera | mode={a.mode} fps={a.fps} "
        f"| chép về RAM: {'MỌI frame' if a.download_all else 'chỉ frame được chọn'} "
        f"| thêm 1 cam mỗi {a.add_every}s, sau đó chạy thêm {a.seconds}s")
    if my_vram_mb() is None:
        log("⚠ không gọi được nvidia-smi — kiểm tra NVIDIA_DRIVER_CAPABILITIES có 'utility'")

    cpu = CpuMeter()
    cams, steps = [], []

    def report(tag):
        vram = my_vram_mb()
        dt = a.report_every
        parts = []
        for c in cams:
            d, v = c.decoded, c.converted
            parts.append(f"{c.idx}: dec={(d - c._d0) / dt:4.1f}fps out={(v - c._v0) / dt:4.1f}fps"
                         + (f" LỖI" if c.err else ""))
            c._d0, c._v0 = d, v
        log(f"[{tag}] VRAM process={vram} MB | {gpu_total_mb()} | CPU process={cpu.percent():.0f}%")
        for p in parts:
            log(f"      {p}")
        return vram

    def run_for(seconds, tag):
        end = time.monotonic() + seconds
        vram = None
        while time.monotonic() < end and not STOP.is_set():
            time.sleep(a.report_every)
            vram = report(tag)
        return vram

    try:
        steps.append((0, my_vram_mb()))
        for i, (name, url) in enumerate(urls, 1):
            c = CamTest(name, url, a.fps, a.mode, a.cuvid, a.cpu, not a.download_all)
            c._d0 = c._v0 = 0
            cams.append(c)
            c.start()
            steps.append((i, run_for(a.add_every, f"{i} cam, vừa thêm {name}")))
        run_for(a.seconds, f"{len(urls)} cam")
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()

    print("\n" + "=" * 70)
    print("KẾT QUẢ VRAM (process này)")
    print("=" * 70)
    prev = None
    for n, v in steps:
        delta = "" if prev is None or v is None or prev is None else f"   (+{v - prev} MB)"
        print(f"  {n} cam : {v} MB{delta}")
        prev = v
    for c in cams:
        print(f"  {c.idx}: {mask(c.url)} | {c.info}" + (f" | LỖI CUỐI: {c.err}" if c.err else ""))
    print("  So sánh: hiện tại mỗi ffmpeg ~466 MB (hwaccel cuda) / ~286 MB (h264_cuvid)")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
#  [record] — ghi camera ra file, KHÔNG giải mã
# ══════════════════════════════════════════════════════════════════════════

def cmd_record(a):
    name, url = resolve([a.cam])[0]
    cmd = [a.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-rtsp_transport", "tcp", "-i", url,
           "-an", "-c:v", "copy", "-t", str(a.seconds), a.out]
    log(f"ghi {a.seconds}s từ {name} {mask(url)} → {a.out}")
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(a.out):
        sys.exit("ghi thất bại")
    log(f"xong: {os.path.getsize(a.out) / 1e6:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════
#  [diff] — cùng 1 file, giải mã cách cũ (ffmpeg) và kiểu hub, so pixel
# ══════════════════════════════════════════════════════════════════════════

def ffmpeg_frames(a, w, h, vf, pix_fmt):
    """Chạy ffmpeg đúng như các project, đọc N frame thô từ pipe."""
    cmd = [a.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if not a.cpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += ["-i", a.clip, "-an", "-vf", vf, "-fps_mode", "passthrough",
            "-frames:v", str(a.frames), "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    log("  " + " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        log(f"  ⚠ ffmpeg lỗi: {out.stderr.decode(errors='replace')[-500:]}")
    size = w * h * 3 if pix_fmt == "bgr24" else w * h * 3 // 2
    data = out.stdout
    frames = []
    for i in range(len(data) // size):
        buf = np.frombuffer(data[i * size:(i + 1) * size], dtype=np.uint8)
        frames.append(buf.reshape(h, w, 3) if pix_fmt == "bgr24"
                      else cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12))
    return frames


def best_offset(ref, cand, max_off=3):
    """Hai đường giải mã có thể lệch nhau 1-2 frame đầu — tìm độ lệch khớp nhất."""
    best = (None, 0)
    n = min(len(ref), len(cand), 5)
    for off in range(-max_off, max_off + 1):
        diffs = []
        for i in range(n):
            j = i + off
            if 0 <= j < len(cand):
                diffs.append(np.abs(ref[i].astype(np.int16) - cand[j]).mean())
        if diffs and (best[0] is None or np.mean(diffs) < best[0]):
            best = (np.mean(diffs), off)
    return best[1]


def compare(ref, cand):
    off = best_offset(ref, cand)
    mean, mx, over2, n = [], 0, [], 0
    for i in range(len(ref)):
        j = i + off
        if not (0 <= j < len(cand)):
            continue
        d = np.abs(ref[i].astype(np.int16) - cand[j])
        mean.append(d.mean())
        mx = max(mx, int(d.max()))
        over2.append((d > 2).mean() * 100)
        n += 1
    if not n:
        return "không có frame chung"
    return (f"lệch TB={np.mean(mean):.3f}  max={mx:3d}  "
            f"pixel lệch>2={np.mean(over2):.3f}%  ({n} frame, offset={off})")


def cmd_diff(a):
    log(f"giải mã kiểu hub (PyAV {av.__version__}{', CPU' if a.cpu else ', NVDEC primary_ctx'}) ...")
    container = av.open(a.clip, hwaccel=make_hwaccel(a.cpu))
    frames = []
    for f in container.decode(video=0):
        frames.append(f)
        if len(frames) >= a.frames:
            break
    container.close()
    if not frames:
        sys.exit("không giải mã được frame nào")
    w, h = frames[0].width, frames[0].height
    log(f"  {len(frames)} frame {w}x{h}, color_range={frames[0].color_range}, "
        f"colorspace={frames[0].colorspace}, format={frames[0].format.name}")

    log("giải mã cách cũ bằng ffmpeg ...")
    cpu_note = a.cpu
    ref_bgr24 = ffmpeg_frames(a, w, h,
                              "format=nv12,format=bgr24" if cpu_note else "hwdownload,format=nv12,format=bgr24",
                              "bgr24")
    ref_tv = [] if cpu_note else ffmpeg_frames(a, w, h, "colorspace_cuda=range=tv,hwdownload,format=nv12", "nv12")

    nv12 = [frame_nv12(f) for f in frames]
    cands = {
        "hub bgr24 BILINEAR": [f.to_ndarray(format="bgr24", interpolation="BILINEAR") for f in frames],
        "hub bgr24 BICUBIC ": [f.to_ndarray(format="bgr24", interpolation="BICUBIC") for f in frames],
        "hub tv (round)    ": [nv12_full_to_bgr_tv(x, "round") for x in nv12],
        "hub tv (floor)    ": [nv12_full_to_bgr_tv(x, "floor") for x in nv12],
    }

    print("\n" + "=" * 70)
    print("KẾT QUẢ SO PIXEL (0 = giống hệt)")
    print("=" * 70)
    for ref_name, ref in (("ffmpeg kiểu bgr24 (format=nv12,format=bgr24)", ref_bgr24),
                          ("ffmpeg kiểu tv (colorspace_cuda=range=tv + cv2)", ref_tv)):
        if not ref:
            print(f"\n  [{ref_name}] — bỏ qua (không có frame)")
            continue
        print(f"\n  So với: {ref_name}  [{len(ref)} frame]")
        for name, cand in cands.items():
            print(f"    {name} : {compare(ref, cand)}")
    if ref_bgr24 and ref_tv:
        print(f"\n  Tham khảo — 2 cách cũ so với nhau: {compare(ref_bgr24, ref_tv)}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="camhub — script thử VRAM & màu")
    p.add_argument("--cpu", action="store_true", help="giải mã CPU (chỉ để thử script, không đo GPU)")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vram")
    v.add_argument("cams", nargs="*", help="cam1 cam2 ... (bỏ trống = tất cả)")
    v.add_argument("--fps", type=float, default=8, help="fps đưa ra (như fps= của ffmpeg), 0 = giữ hết")
    v.add_argument("--mode", choices=["bgr24", "tv"], default="bgr24")
    v.add_argument("--cuvid", action="store_true", help="dùng decoder h264_cuvid/hevc_cuvid thay vì hwaccel")
    v.add_argument("--download-all", action="store_true",
                   help="chép MỌI frame về RAM như lần chạy trước (để so sánh)")
    v.add_argument("--add-every", type=int, default=30, help="giây giữa 2 lần thêm cam")
    v.add_argument("--seconds", type=int, default=60, help="chạy thêm sau khi đủ cam")
    v.add_argument("--report-every", type=int, default=10)

    r = sub.add_parser("record")
    r.add_argument("cam", help="cam1, cam2, ...")
    r.add_argument("out")
    r.add_argument("--seconds", type=int, default=10)
    r.add_argument("--ffmpeg", default="ffmpeg")

    d = sub.add_parser("diff")
    d.add_argument("clip")
    d.add_argument("--frames", type=int, default=30)
    d.add_argument("--ffmpeg", default="ffmpeg")

    a = p.parse_args()
    {"vram": cmd_vram, "record": cmd_record, "diff": cmd_diff}[a.cmd](a)


if __name__ == "__main__":
    main()
