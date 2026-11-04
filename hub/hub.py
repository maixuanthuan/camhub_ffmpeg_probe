# -*- coding: utf-8 -*-
"""
hub.py — camhub
===============
MỘT process giải mã MỌI camera bằng NVDEC, chia frame cho các project qua
shared memory. Thay cho việc mỗi project tự chạy ffmpeg riêng.

Đã đo (RTX 3060, cam HEVC/H.264 2688x1520) — xem docs/BENCHMARK.md:
    VRAM  ~290 MB cho cam đầu + ~190 MB mỗi cam thêm  (ffmpeg riêng: ~466 MB/cam)
    CPU   đổi màu GPU: tv 7-8 ms CPU/frame (CPU: 57-62), bgr24 11 ms (CPU: 19-21)
    decode giữ đủ 25 fps

Nguyên tắc:
    - Camera khai DUY NHẤT trong camhub.env:  CAM1=rtsp://..., CAM2=...
      Sửa file này hub tự đọc lại, KHÔNG cần restart.
    - Hub chỉ mở camera nào ĐANG CÓ project đọc. Không ai đọc 30s → đóng cam,
      trả VRAM. Mỗi camera chỉ 1 kết nối RTSP dù bao nhiêu project dùng.
    - Mỗi camera 1 luồng, tự kết nối lại (5s → 30s). Cam này chết không ảnh
      hưởng cam khác.
    - Frame giải mã xong NẰM TRÊN GPU, chỉ frame cần (theo fps project xin)
      mới chép về RAM và đổi màu.
    - Hai loại nhu cầu, mỗi frame giải mã xong hub xét cả hai:
        client xin fps=N   → đổi màu theo đồng hồ, N frame/giây (fps lớn nhất các client)
        client xin mọi frame (fps=0) → đổi màu khi frame trước ĐÃ có client đó lấy
            (client báo qua ô "consumed" trong header). Project xin mọi frame mà
            xử lý chậm thì hub chỉ đổi màu đúng số frame nó đọc kịp — giống cv2
            đọc tuần tự trước đây, không đổi màu 25 fps rồi bỏ.

Hai chế độ màu (khớp 2 cách các project đang làm, đã so pixel):
    bgr24 — như "-vf hwdownload,format=nv12,format=bgr24"   (lệch TB 0.3 mức)
    tv    — như "colorspace_cuda=range=tv" + cv2 NV12→BGR    (lệch TB 0.000)

Đổi màu trên GPU (mặc định, CAMHUB_GPU_COLOR=1): kernel CUDA nhỏ (cupy) đổi
NV12 → BGR ngay trên GPU rồi mới chép về RAM — CPU chỉ còn công chép.
    tv    : đúng phép tính số nguyên của cv2 → giống hệt bản CPU từng bit
    bgr24 : BT.601 full range, chroma nội suy bicubic → lệch TB ~0.6-0.7 so với swscale
            (CAMHUB_GPU_MODES=tv để bgr24 về CPU nếu project nào nhạy màu)
GPU lỗi (thiếu cupy, kernel không biên dịch...) → tự quay về đổi màu CPU, log 1 lần.
Kiểm tra trên camera thật:  python3 hub.py check cam7

Thư mục chung (host /dev/shm/camhub, trong container /camhub):
    cameras.json              cam → địa chỉ (KHÔNG có mật khẩu), client tra URL
    status.json               trạng thái từng cam, cập nhật mỗi 5s
    <cam>.<mode>.buf          frame mới nhất, 3 slot xoay vòng
    want/<cam>.<mode>.<id>    "đơn xin" của từng client, client tự làm mới mỗi 2s

Định dạng <cam>.<mode>.buf (camhub_client.py đọc đúng định dạng này):
    header 64 byte: magic "CHB1", version, width, height, channels, nslots,
                    frame_bytes, seq (frame hoàn chỉnh mới nhất, 0 = chưa có),
                    consumed (byte 40: seq client lấy gần nhất — CLIENT ghi)
    nslots × [slot header 16 byte: seq, timestamp] + [frame_bytes dữ liệu]
    Ghi: slot.seq = 0 → chép dữ liệu → slot.ts → slot.seq = n → header.seq = n
    Đọc: lấy header.seq = n, slot = n % nslots, chép, kiểm tra slot.seq vẫn = n
"""

import os
import sys
import json
import fcntl
import mmap
import time
import struct
import signal
import logging
import threading

import av
import cv2
import numpy as np
from av.codec.hwaccel import HWAccel

HUB_DIR  = os.environ.get("CAMHUB_DIR", "/camhub")
ENV_FILE = os.environ.get("CAMHUB_ENV", "/app/camhub.env")
WANT_DIR = os.path.join(HUB_DIR, "want")

WANT_TTL        = 10     # giây — đơn xin cũ hơn mức này coi như client đã chết
WANT_DELETE     = 300    # giây — đơn xin cũ hơn mức này thì xoá file
IDLE_STOP       = 30     # giây không ai đọc → đóng camera
SCAN_EVERY      = 1.0
STATUS_EVERY    = 5.0
LOG_EVERY       = 60.0
RECONNECT_BASE  = 5      # giây, tăng 3s mỗi lần, tối đa RECONNECT_MAX
RECONNECT_MAX   = 30

MODES = ("bgr24", "tv")

RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "timeout":        "10000000",   # µs
}

MAGIC       = b"CHB1"
VERSION     = 1
NSLOTS      = 3
HEADER_SIZE = 64
HEADER_FMT  = "<4sIIIIIQQ"          # magic ver w h c nslots frame_bytes seq
SEQ_OFFSET  = 32
CONSUMED_OFFSET = 40                # client ghi seq vừa lấy
SLOT_HDR    = 16                    # seq (Q) + ts (d)
CONSUME_WAIT = 1.0                  # giây — frame chưa ai lấy quá lâu thì vẫn làm mới

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("camhub")

STOP = threading.Event()
_TEST_PACE = os.environ.get("CAMHUB_CPU") == "1"   # máy không GPU, nguồn là file: chỉ để thử
GPU_COLOR  = os.environ.get("CAMHUB_GPU_COLOR", "1") == "1"
# mode nào đổi màu trên GPU (đo trên server: tv giống CPU từng bit; bgr24 lệch TB ~0.6-0.7
# so với swscale). Project nào nhạy màu với bgr24 → đặt CAMHUB_GPU_MODES=tv
GPU_MODES  = {m.strip() for m in os.environ.get("CAMHUB_GPU_MODES", "tv,bgr24").split(",") if m.strip()}


# ══════════════════════════════════════════════════════════════════════════
#  Tiện ích
# ══════════════════════════════════════════════════════════════════════════

def mask(url: str) -> str:
    """Che user:pass khi in log."""
    if "://" in url and "@" in url:
        head, rest = url.split("://", 1)
        return f"{head}://***@{rest.rsplit('@', 1)[1]}"
    return url


def cam_addr(url: str) -> str:
    """rtsp://user:pass@Host:port/path → host:port/path (để client tra theo URL)."""
    u = url.strip()
    if "://" in u:
        u = u.split("://", 1)[1]
    host_port, _, path = u.partition("/")
    if "@" in host_port:
        host_port = host_port.rsplit("@", 1)[1]
    host, _, port = host_port.partition(":")
    return f"{host.lower()}:{port or '554'}/{path.rstrip('/')}"


def write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.chmod(tmp, 0o666)
    os.replace(tmp, path)


def load_cameras(path: str = ENV_FILE) -> dict:
    """camhub.env → {"cam1": url, ...}. Bỏ dòng trống, dòng #, giá trị rỗng."""
    cams = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key.upper().startswith("CAM") and key[3:].isdigit() and val:
                    cams[f"cam{int(key[3:])}"] = val
    except FileNotFoundError:
        logger.error(f"Không thấy {path}")
    return dict(sorted(cams.items(), key=lambda kv: int(kv[0][3:])))


# ══════════════════════════════════════════════════════════════════════════
#  Đổi màu — giống camhub_poc.py đã so pixel trên server
# ══════════════════════════════════════════════════════════════════════════

def _tv_luts():
    x = np.arange(256, dtype=np.float64)
    y = np.clip(np.round(x * 219.0 / 255.0 + 16.0), 0, 255).astype(np.uint8)
    c = np.clip(np.round((x - 128.0) * 224.0 / 255.0 + 128.0), 0, 255).astype(np.uint8)
    return y, c


_Y_LUT, _C_LUT = _tv_luts()


def _is_limited(frame) -> bool:
    """Nguồn đã là tv range (MPEG) → không đổi range nữa. Camera hiện tại đều full (JPEG)."""
    try:
        return int(frame.color_range) == 1
    except Exception:
        return False


def to_bgr(frame, mode: str) -> np.ndarray:
    """Đổi màu bằng CPU. Frame (đang trên GPU) được chép về RAM ngay tại đây."""
    if mode == "bgr24":
        return frame.to_ndarray(format="bgr24", interpolation="BICUBIC")
    # tv: full range → tv range (như colorspace_cuda=range=tv), rồi cv2 NV12→BGR
    if frame.format.name != "nv12":
        frame = frame.reformat(format="nv12")
    nv12 = frame.to_ndarray()
    if _is_limited(frame):
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    h = nv12.shape[0] * 2 // 3
    out = np.empty_like(nv12)
    out[:h] = _Y_LUT[nv12[:h]]
    out[h:] = _C_LUT[nv12[h:]]
    return cv2.cvtColor(out, cv2.COLOR_YUV2BGR_NV12)


# ── đổi màu trên GPU ────────────────────────────────────────────────────────

_KERNEL_SRC = r"""
extern "C" __global__
void nv12_to_bgr(const unsigned char* __restrict__ Y, const int ypitch,
                 const unsigned char* __restrict__ UV, const int uvpitch,
                 unsigned char* __restrict__ out, const int W, const int H,
                 const int mode, const int limited,
                 const unsigned char* __restrict__ ylut,
                 const unsigned char* __restrict__ clut,
                 const int* __restrict__ hoff, const float* __restrict__ hw,
                 const int* __restrict__ voff, const float* __restrict__ vw)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int cx = x >> 1, cy = y >> 1;
    const int yy = Y[y * ypitch + x];
    int b, g, r;
    if (mode == 1) {
        // tv: LUT full→tv (như colorspace_cuda=range=tv) rồi ĐÚNG phép tính số nguyên
        // của cv2 COLOR_YUV2BGR_NV12 (chroma lấy gần nhất, hệ số BT.601 << 20)
        const int u = clut[UV[cy * uvpitch + 2 * cx]] - 128;
        const int v = clut[UV[cy * uvpitch + 2 * cx + 1]] - 128;
        const int yp = max(0, (int)ylut[yy] - 16) * 1220542;
        r = (yp + 524288 + 1673527 * v) >> 20;
        g = (yp + 524288 - 852492 * v - 409993 * u) >> 20;
        b = (yp + 524288 + 2116026 * u) >> 20;
    } else {
        // bgr24: BT.601 full range. Chroma nội suy 4x4 tap theo bảng trọng số
        // (hoff/hw ngang, voff/vw dọc; mỗi bảng 2 hàng = pixel chẵn / lẻ) —
        // bảng tính ở Python (chroma_taps), chọn kiểu khớp swscale nhất bằng "check".
        const int px = x & 1, py = y & 1;
        float u = 0.f, v = 0.f;
        for (int j = 0; j < 4; ++j) {
            const float wv = vw[py * 4 + j];
            if (wv == 0.f) continue;
            const int ry = min(max(cy + voff[py * 4 + j], 0), H / 2 - 1);
            const unsigned char* row = UV + ry * uvpitch;
            for (int i = 0; i < 4; ++i) {
                const float wh = hw[px * 4 + i];
                if (wh == 0.f) continue;
                const int rx = min(max(cx + hoff[px * 4 + i], 0), W / 2 - 1);
                const float w2 = wv * wh;
                u += w2 * row[2 * rx];
                v += w2 * row[2 * rx + 1];
            }
        }
        u -= 128.f;
        v -= 128.f;
        float fy = (float)yy;
        if (limited) { fy = (fy - 16.f) * 1.164383f; u *= 1.138393f; v *= 1.138393f; }
        r = __float2int_rn(fy + 1.402f * v);
        g = __float2int_rn(fy - 0.344136f * u - 0.714136f * v);
        b = __float2int_rn(fy + 1.772f * u);
    }
    unsigned char* o = out + ((size_t)y * W + x) * 3;
    o[0] = (unsigned char)min(max(b, 0), 255);
    o[1] = (unsigned char)min(max(g, 0), 255);
    o[2] = (unsigned char)min(max(r, 0), 255);
}
"""


def chroma_taps(pos: float, kind: str, a: float = -0.6):
    """Bảng 4 tap nội suy chroma 4:2:0 → full cho pixel chẵn/lẻ.
    pos: vị trí mẫu chroma tính theo luma (0 = trùng luma chẵn, 0.5 = giữa 2 luma).
    kind: "linear" | "cubic" (Keys, tham số a; swscale BICUBIC ~ a=-0.6)."""
    def k(t):
        t = abs(t)
        if kind == "linear":
            return max(0.0, 1.0 - t)
        if t < 1:
            return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
        if t < 2:
            return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
        return 0.0
    offs, ws = [], []
    for parity in (0, 1):
        p = (parity - pos) / 2.0                      # vị trí cần lấy, theo chỉ số chroma
        k0 = int(np.floor(p))
        o = [k0 - 1, k0, k0 + 1, k0 + 2]
        w = np.array([k(p - q) for q in o], dtype=np.float64)
        offs.append(o)
        ws.append(w / w.sum())
    return np.array(offs, dtype=np.int32), np.array(ws, dtype=np.float32)


# Kiểu nội suy chroma cho bgr24: (vị trí ngang, vị trí dọc, kiểu, a).
# Đặt lại bằng biến môi trường CAMHUB_BGR24_CHROMA="0,0.5,cubic,-0.6" nếu "check" chỉ ra kiểu khác khớp hơn.
BGR24_CHROMA = os.environ.get("CAMHUB_BGR24_CHROMA", "0,0.5,cubic,-0.6")
CHROMA_VARIANTS = [f"{h},{v},{kd},{a}" for h in ("0", "0.5") for v in ("0.5", "0")
                   for kd, a in (("linear", "0"), ("cubic", "-0.5"), ("cubic", "-0.6"), ("cubic", "-0.75"))]


def parse_chroma(spec: str):
    h, v, kind, a = spec.split(",")
    return chroma_taps(float(h), kind, float(a)), chroma_taps(float(v), kind, float(a))


class GpuColor:
    """NV12 (frame NVDEC trên GPU) → BGR trên GPU, chép thẳng vào chỗ đích trong RAM.
    Dùng chung CUDA primary context với decoder (FFmpeg primary_ctx=1)."""

    MODES = {"bgr24": 0, "tv": 1}

    def __init__(self):
        # FFmpeg đòi CUDA primary context có cờ BLOCKING_SYNC; cupy mà tạo context
        # trước (cờ mặc định) thì mọi lần mở decoder sau đó báo "Operation not
        # supported". → Cho FFmpeg tạo/giữ context TRƯỚC, cupy dùng chung sau.
        from av.codec.hwaccel import HWDevice
        self._ff_device = HWDevice("cuda", options={"primary_ctx": "1"})
        import cupy as cp
        self.cp = cp
        self.kernel = cp.RawKernel(_KERNEL_SRC, "nv12_to_bgr")
        self.ylut = cp.asarray(_Y_LUT)
        self.clut = cp.asarray(_C_LUT)
        self.ident = cp.arange(256, dtype=cp.uint8)
        self._out = {}
        self.set_chroma(BGR24_CHROMA)

    def set_chroma(self, spec: str) -> None:
        (ho, hw), (vo, vw) = parse_chroma(spec)
        cp = self.cp
        self.taps = (cp.asarray(ho), cp.asarray(hw), cp.asarray(vo), cp.asarray(vw))
        self.chroma = spec

    def convert_into(self, frame, mode: str, dst: np.ndarray) -> None:
        if frame.format.name != "cuda":
            raise TypeError(f"frame không nằm trên GPU ({frame.format.name})")
        cp = self.cp
        h, w = frame.height, frame.width
        key = (threading.get_ident(), h, w)            # mỗi luồng camera 1 vùng nhớ riêng
        out = self._out.get(key)
        if out is None:
            out = self._out[key] = cp.empty((h, w, 3), dtype=cp.uint8)
        p0, p1 = frame.planes[0], frame.planes[1]     # Y, UV xen kẽ (con trỏ GPU)
        limited = _is_limited(frame)
        ylut, clut = (self.ident, self.ident) if limited else (self.ylut, self.clut)
        self.kernel(
            ((w + 31) // 32, (h + 7) // 8), (32, 8),
            (np.uint64(p0.buffer_ptr), np.int32(p0.line_size),
             np.uint64(p1.buffer_ptr), np.int32(p1.line_size),
             out, np.int32(w), np.int32(h),
             np.int32(self.MODES[mode]), np.int32(1 if limited else 0),
             ylut, clut, *self.taps),
        )
        out.get(out=dst)                               # chờ GPU xong, chép về RAM


_GPU = None
_GPU_LOCK = threading.Lock()
_GPU_FAILED = None


def gpu_color():
    """GpuColor dùng chung cho mọi camera, hoặc None nếu tắt/lỗi (→ đổi màu CPU)."""
    global _GPU, _GPU_FAILED
    if not GPU_COLOR or _TEST_PACE or _GPU_FAILED:
        return None
    with _GPU_LOCK:
        if _GPU is None and not _GPU_FAILED:
            try:
                _GPU = GpuColor()
                logger.info(f"Đổi màu trên GPU: BẬT (cupy) | mode: {','.join(sorted(GPU_MODES))} "
                            f"| bgr24 chroma={BGR24_CHROMA}")
            except Exception as e:
                _GPU_FAILED = f"{type(e).__name__}: {e}"
                logger.warning(f"Đổi màu trên GPU KHÔNG dùng được → đổi màu CPU. Lý do: {_GPU_FAILED}")
        return _GPU


def gpu_disable(reason: str) -> None:
    global _GPU_FAILED
    if not _GPU_FAILED:
        _GPU_FAILED = reason
        logger.warning(f"Đổi màu GPU lỗi lúc chạy → chuyển sang CPU. Lý do: {reason}")


# ══════════════════════════════════════════════════════════════════════════
#  Shared memory — 1 file / (cam, mode)
# ══════════════════════════════════════════════════════════════════════════

class FrameBuffer:

    def __init__(self, path: str, h: int, w: int, c: int):
        self.path = path
        self.shape = (h, w, c)
        self.frame_bytes = h * w * c
        self.seq = 0
        self.last_write = 0.0
        size = HEADER_SIZE + NSLOTS * (SLOT_HDR + self.frame_bytes)

        # tạo ở file tạm rồi rename → client không bao giờ thấy file dở dang
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.truncate(size)
        os.chmod(tmp, 0o666)
        fd = os.open(tmp, os.O_RDWR)
        try:
            self.mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        struct.pack_into(HEADER_FMT, self.mm, 0,
                         MAGIC, VERSION, w, h, c, NSLOTS, self.frame_bytes, 0)
        os.replace(tmp, path)
        self.ino = os.stat(path).st_ino

        self.slots = []
        for i in range(NSLOTS):
            off = HEADER_SIZE + i * (SLOT_HDR + self.frame_bytes)
            view = np.frombuffer(self.mm, dtype=np.uint8, count=self.frame_bytes,
                                 offset=off + SLOT_HDR).reshape(self.shape)
            self.slots.append((off, view))

    def write(self, img: np.ndarray, ts: float) -> None:
        self.write_with(lambda view: np.copyto(view, img), ts)

    def write_with(self, fill, ts: float) -> None:
        """fill(view) ghi thẳng frame vào slot (đổi màu GPU chép thẳng vào đây)."""
        n = self.seq + 1
        off, view = self.slots[n % NSLOTS]
        struct.pack_into("<Q", self.mm, off, 0)          # slot đang ghi
        fill(view)
        struct.pack_into("<d", self.mm, off + 8, ts)
        struct.pack_into("<Q", self.mm, off, n)          # slot xong
        struct.pack_into("<Q", self.mm, SEQ_OFFSET, n)   # công bố frame mới
        self.seq = n
        self.last_write = time.monotonic()

    def consumed(self) -> int:
        return struct.unpack_from("<Q", self.mm, CONSUMED_OFFSET)[0]

    def waiting(self, now: float) -> bool:
        """Frame mới nhất CHƯA client nào lấy (và chưa quá CONSUME_WAIT giây)."""
        return (self.seq > 0 and self.consumed() < self.seq
                and now - self.last_write < CONSUME_WAIT)

    def close(self, remove: bool = True) -> None:
        try:
            self.slots = []
            self.mm.close()
        except Exception:
            pass
        if remove:
            # chỉ xoá đúng file của mình — worker mới của cùng cam có thể đã tạo file mới
            try:
                if os.stat(self.path).st_ino == self.ino:
                    os.remove(self.path)
            except FileNotFoundError:
                pass


# ══════════════════════════════════════════════════════════════════════════
#  Một camera = một luồng
# ══════════════════════════════════════════════════════════════════════════

class CamWorker(threading.Thread):

    def __init__(self, name: str, url: str):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.cam = name
        self.url = url
        self.targets = {}            # mode → (fps_giới_hạn, có_client_mọi_frame, mọi_frame_có_báo); main cập nhật
        self._stop_ev = threading.Event()
        self._bufs = {}              # mode → FrameBuffer
        self._next_t = {}            # mode → mốc frame kế tiếp

        self.state = "connecting"
        self.error = None
        self.color = None            # "gpu" / "cpu" — đổi màu ở đâu
        self.codec = None
        self.size = None
        self.reconnects = 0
        self.started_at = time.time()
        self.connected_at = None
        self.last_frame_at = None
        self._dec = 0
        self._out = {}               # mode → số frame đã ghi
        self._read = {}              # mode → (consumed lần đo trước)
        self.read_fps = {}
        self._fps_t = time.monotonic()
        self.dec_fps = 0.0
        self.out_fps = {}

    # ── điều khiển ───────────────────────────────────────────────────────
    def stop(self):
        self._stop_ev.set()

    def _sleep(self, seconds):
        self._stop_ev.wait(seconds)

    # ── vòng chạy ────────────────────────────────────────────────────────
    def run(self):
        attempt = 0
        while not self._stop_ev.is_set() and not STOP.is_set():
            try:
                self._run_once()
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}".replace(self.url, mask(self.url))
                logger.warning(f"[{self.cam}] lỗi: {self.error}")
            if self._stop_ev.is_set() or STOP.is_set():
                break
            # kết nối giữ được ≥ 60s thì coi như ổn định → reset backoff
            if self.connected_at and time.time() - self.connected_at >= 60:
                attempt = 0
            attempt += 1
            self.reconnects += 1
            self.state = "reconnecting"
            delay = min(RECONNECT_BASE + (attempt - 1) * 3, RECONNECT_MAX)
            logger.warning(f"[{self.cam}] kết nối lại lần {attempt} sau {delay}s")
            self._sleep(delay)
        self._close_bufs()
        self.state = "stopped"

    def _run_once(self):
        self.state = "connecting"
        self.connected_at = None
        hw = None if os.environ.get("CAMHUB_CPU") == "1" else HWAccel(   # CAMHUB_CPU=1: chỉ để thử máy không GPU
            "cuda", allow_software_fallback=False,
            options={"primary_ctx": "1"}, is_hw_owned=True)
        container = av.open(self.url, options=RTSP_OPTS, timeout=(20, 10), hwaccel=hw)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            cc = stream.codec_context
            self.codec = cc.name
            logger.info(f"[{self.cam}] đã mở {mask(self.url)} | {cc.name} {cc.width}x{cc.height}")
            self.connected_at = time.time()
            self.state = "running"
            self.error = None
            self._next_t = {}

            for frame in container.decode(stream):
                if self._stop_ev.is_set() or STOP.is_set():
                    return
                if _TEST_PACE:
                    time.sleep(0.04)                   # chỉ khi thử (file video): giả 25fps như camera
                now = time.monotonic()
                self._dec += 1
                self.last_frame_at = time.time()

                for mode, (fps, full, full_ack) in list(self.targets.items()):
                    buf = self._bufs.get(mode)
                    # client xin fps=N: tới lượt theo đồng hồ chưa
                    need = bool(fps) and self._due(mode, fps, now)
                    # client xin mọi frame: client bản cũ (không báo) → frame nào cũng đổi;
                    # bản mới → chỉ khi frame trước đã có người lấy
                    if full and not need:
                        need = not full_ack or buf is None or not buf.waiting(now)
                    if not need:
                        continue
                    self._convert(frame, mode)
                    self._out[mode] = self._out.get(mode, 0) + 1
                    self.size = (frame.width, frame.height)
                self._update_fps(now)
            self.error = "stream kết thúc (camera ngắt)"
            logger.warning(f"[{self.cam}] {self.error}")
        finally:
            container.close()

    def _convert(self, frame, mode):
        gpu = gpu_color() if mode in GPU_MODES else None
        if gpu is not None and frame.format.name == "cuda":
            buf = self._buf_for(mode, frame.height, frame.width, 3)
            try:
                buf.write_with(lambda view: gpu.convert_into(frame, mode, view), self.last_frame_at)
                self.color = "gpu"
                return
            except Exception as e:
                gpu_disable(f"{type(e).__name__}: {e}")
        img = to_bgr(frame, mode)
        self._buf(mode, img).write(img, self.last_frame_at)
        self.color = "cpu"

    def _due(self, mode, fps, now) -> bool:
        """Lọc fps theo đồng hồ thật, giống filter fps= của ffmpeg."""
        period = 1.0 / fps
        nt = self._next_t.get(mode)
        # cho sớm nửa chu kỳ: frame camera về không đều (jitter mạng) — so chặt
        # thì 25→20fps chỉ còn ~17fps. Mốc vẫn cộng đều period nên fps trung bình đúng.
        if nt is not None and now < nt - 0.5 * period:
            return False
        self._next_t[mode] = now + period if nt is None or now - nt > period else nt + period
        return True

    def _buf(self, mode, img) -> FrameBuffer:
        h, w = img.shape[:2]
        return self._buf_for(mode, h, w, img.shape[2] if img.ndim == 3 else 1)

    def _buf_for(self, mode, h, w, c) -> FrameBuffer:
        buf = self._bufs.get(mode)
        if buf is None or buf.shape != (h, w, c):
            if buf is not None:
                logger.info(f"[{self.cam}] đổi kích thước frame → {w}x{h}")
                buf.close(remove=False)          # file mới sẽ thay chỗ (rename)
            buf = FrameBuffer(os.path.join(HUB_DIR, f"{self.cam}.{mode}.buf"), h, w, c)
            self._bufs[mode] = buf
        return buf

    def _close_bufs(self):
        for buf in self._bufs.values():
            buf.close(remove=True)
        self._bufs = {}

    def _update_fps(self, now):
        dt = now - self._fps_t
        if dt >= 5.0:
            self.dec_fps = self._dec / dt
            self.out_fps = {m: n / dt for m, n in self._out.items()}
            for m, buf in self._bufs.items():
                c = buf.consumed()
                prev = self._read.get(m)
                self.read_fps[m] = 0.0 if prev is None else max(0, c - prev) / dt
                self._read[m] = c
            self._dec = 0
            self._out = {}
            self._fps_t = now

    def status(self) -> dict:
        stale = self.last_frame_at is None or time.time() - self.last_frame_at > 5
        return {
            "state":      self.state if not (self.state == "running" and stale) else "stalled",
            "codec":      self.codec,
            "color":      self.color,
            "size":       self.size,
            "dec_fps":    round(self.dec_fps, 1),
            # fps_out = số frame hub đổi màu/giây
            # fps_read_full = số frame client "mọi frame" lấy/giây (từ ô consumed)
            # fps_target: fps giới hạn lớn nhất; full: có client xin mọi frame
            "outputs":    {m: {"fps_target": f, "full": full, "fps_out": round(self.out_fps.get(m, 0.0), 1),
                               "fps_read_full": round(self.read_fps.get(m, 0.0), 1)}
                           for m, (f, full, _a) in self.targets.items()},
            "reconnects": self.reconnects,
            "error":      self.error,
            "last_frame": self.last_frame_at,
        }


# ══════════════════════════════════════════════════════════════════════════
#  Đơn xin của client
# ══════════════════════════════════════════════════════════════════════════

def scan_wants() -> dict:
    """want/<cam>.<mode>.<id> → {cam: {mode: [(fps, ack), ...]}} (chỉ đơn còn hạn).
    ack = client có báo "đã lấy frame" (camhub_client bản mới) hay không."""
    now = time.time()
    wants = {}
    try:
        entries = list(os.scandir(WANT_DIR))
    except FileNotFoundError:
        return wants
    for e in entries:
        if e.name.endswith(".tmp"):
            continue
        parts = e.name.split(".", 2)
        if len(parts) != 3 or parts[1] not in MODES:
            continue
        try:
            age = now - e.stat().st_mtime
        except FileNotFoundError:
            continue
        if age > WANT_DELETE:
            try:
                os.remove(e.path)
            except OSError:
                pass
            continue
        if age > WANT_TTL:
            continue
        try:
            with open(e.path, encoding="utf-8") as f:
                d = json.load(f)
            fps, ack = float(d.get("fps", 0) or 0), bool(d.get("ack", False))
        except Exception:
            fps, ack = 0.0, False
        cam, mode = parts[0], parts[1]
        wants.setdefault(cam, {}).setdefault(mode, []).append((fps, ack))
    return wants


def merge_target(lst) -> tuple:
    """Nhiều client xin cùng cam/mode → (fps_giới_hạn, có_client_mọi_frame, mọi_frame_có_báo).
    fps_giới_hạn: lớn nhất trong các client xin fps=N (0 = không có).
    mọi_frame_có_báo: mọi client fps=0 đều là bản mới (báo "đã lấy") — còn bản cũ
    thì đổi màu mọi frame như trước, không để nó bị đói frame."""
    limited = [f for f, _ in lst if f > 0]
    full = [a for f, a in lst if f <= 0]
    return (max(limited) if limited else 0.0), bool(full), all(full)


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def _single_instance_lock():
    """Mỗi máy CHỈ 1 hub. Hub thứ hai (ai đó chạy nhầm) → thoát ngay, không đụng hub đang chạy."""
    os.makedirs(HUB_DIR, exist_ok=True)
    fd = os.open(os.path.join(HUB_DIR, "hub.lock"), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("ĐÃ CÓ camhub khác đang chạy trên máy này (%s/hub.lock). "
                     "KHÔNG chạy 2 hub — project chỉ cần camhub_client.py, xem README.md. Thoát."
                     % HUB_DIR)
        sys.exit(1)
    return fd                         # giữ fd mở suốt đời process = giữ khoá


def main():
    _lock = _single_instance_lock()   # PHẢI trước mọi thứ: dọn buffer bên dưới sẽ phá hub đang chạy
    os.makedirs(WANT_DIR, exist_ok=True)
    for d in (HUB_DIR, WANT_DIR):
        os.chmod(d, 0o777)   # client trong container khác (hoặc user khác) ghi được

    # buffer sót từ lần chạy trước
    for name in os.listdir(HUB_DIR):
        if name.endswith(".buf") or name.endswith(".tmp"):
            try:
                os.remove(os.path.join(HUB_DIR, name))
            except OSError:
                pass

    def _on_term(signum, _frame):
        logger.info(f"Nhận tín hiệu {signum}, đang dừng...")
        STOP.set()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    cams, env_mtime = {}, None
    workers = {}          # cam → CamWorker
    last_wanted = {}      # cam → lần cuối có client xin
    warned = set()
    last_status = last_log = 0.0

    logger.info(f"camhub khởi động | PyAV {av.__version__} | dir={HUB_DIR} | env={ENV_FILE}")

    while not STOP.is_set():
        now = time.time()

        # ── đọc lại camhub.env khi file đổi ─────────────────────────────
        try:
            m = os.stat(ENV_FILE).st_mtime
        except FileNotFoundError:
            m = None
        if m != env_mtime:
            env_mtime = m
            new = load_cameras()
            for cam, w in list(workers.items()):
                if new.get(cam) != w.url:          # cam bị xoá hoặc đổi link
                    logger.info(f"[{cam}] link đổi/bị xoá trong camhub.env → khởi động lại")
                    w.stop()
                    workers.pop(cam)
            cams = new
            write_json(os.path.join(HUB_DIR, "cameras.json"),
                       {c: {"addr": cam_addr(u)} for c, u in cams.items()})
            logger.info(f"camhub.env: {len(cams)} camera — {', '.join(cams) or '(trống)'}")
            warned.clear()

        # ── khớp đơn xin với camera ──────────────────────────────────────
        wants = scan_wants()
        for cam, modes in wants.items():
            if cam not in cams:
                if cam not in warned:
                    logger.warning(f"Có client xin '{cam}' nhưng camhub.env không có cam này")
                    warned.add(cam)
                continue
            last_wanted[cam] = now
            w = workers.get(cam)
            if w is None or not w.is_alive():
                w = CamWorker(cam, cams[cam])
                workers[cam] = w
                w.start()
                logger.info(f"[{cam}] có client xin → mở camera")
            w.targets = {mode: merge_target(lst) for mode, lst in modes.items()}

        for cam, w in list(workers.items()):
            if cam in wants:
                continue
            w.targets = {}                          # không ai đọc → thôi đổi màu
            if now - last_wanted.get(cam, 0) > IDLE_STOP:
                logger.info(f"[{cam}] {IDLE_STOP}s không ai đọc → đóng camera")
                w.stop()
                workers.pop(cam)

        # ── trạng thái ───────────────────────────────────────────────────
        if now - last_status >= STATUS_EVERY:
            last_status = now
            clients = {c: {m: len(l) for m, l in md.items()} for c, md in wants.items()}
            st = {"updated": now, "cameras": {}}
            for cam in cams:
                w = workers.get(cam)
                s = w.status() if w else {"state": "idle"}
                s["clients"] = clients.get(cam, {})
                st["cameras"][cam] = s
            write_json(os.path.join(HUB_DIR, "status.json"), st)

        if now - last_log >= LOG_EVERY:
            last_log = now
            if not workers:
                logger.info("[TÓM TẮT] chưa có camera nào được đọc")
            for cam, w in workers.items():
                s = w.status()
                outs = ", ".join(f"{m} {o['fps_out']}fps"
                                 + (f" (mọi frame: lấy {o['fps_read_full']}fps)" if o["full"] else "")
                                 for m, o in s["outputs"].items()) or "không ai đọc"
                n = sum(len(l) for l in wants.get(cam, {}).values())
                logger.info(f"[TÓM TẮT] {cam}: {s['state']} | decode {s['dec_fps']}fps | màu {s['color']} | "
                            f"ra: {outs} | {n} client | reconnect {s['reconnects']}"
                            + (f" | lỗi: {s['error']}" if s["error"] else ""))

        STOP.wait(SCAN_EVERY)

    for w in workers.values():
        w.stop()
    for w in workers.values():
        w.join(timeout=15)
    logger.info("camhub đã dừng")


def check(cam: str, frames: int = 30) -> None:
    """python3 hub.py check cam7 — so đổi màu GPU với CPU trên camera thật (mở thêm 1 kết nối).
    In: độ lệch, thời gian chờ (wall) và CPU thật bị chiếm (process CPU) mỗi frame,
    và thử mọi kiểu nội suy chroma cho bgr24 để chọn kiểu khớp swscale nhất."""
    cams = load_cameras()
    if cam not in cams:
        sys.exit(f"'{cam}' không có trong {ENV_FILE}. Đang có: {', '.join(cams)}")
    gpu = gpu_color()
    if gpu is None:
        sys.exit(f"GPU đổi màu không dùng được: {_GPU_FAILED}")
    hw = HWAccel("cuda", allow_software_fallback=False, options={"primary_ctx": "1"}, is_hw_owned=True)
    c = av.open(cams[cam], options=RTSP_OPTS, timeout=(20, 10), hwaccel=hw)

    def timed(fn):
        w0, c0 = time.perf_counter(), time.process_time()
        r = fn()
        return r, time.perf_counter() - w0, time.process_time() - c0

    def stats(a, b):
        d = np.abs(a.astype(np.int16) - b)
        return d.mean(), int(d.max()), (d > 2).mean() * 100

    t = {k: [] for k in ("cpu_bgr24", "gpu_bgr24", "cpu_tv", "gpu_tv")}   # (wall, cpu)
    tv = []
    var = {v: [] for v in CHROMA_VARIANTS}
    n = 0
    try:
        for frame in c.decode(c.streams.video[0]):
            n += 1
            if n <= 5:
                continue                               # bỏ vài frame đầu
            for m in MODES:
                a, w_, c_ = timed(lambda: to_bgr(frame, m))
                t[f"cpu_{m}"].append((w_, c_))
                b = np.empty_like(a)
                gpu.set_chroma(BGR24_CHROMA)
                _, w_, c_ = timed(lambda: gpu.convert_into(frame, m, b))
                t[f"gpu_{m}"].append((w_, c_))
                if m == "tv":
                    tv.append(stats(a, b))
                else:
                    for spec in CHROMA_VARIANTS:
                        gpu.set_chroma(spec)
                        gpu.convert_into(frame, m, b)
                        var[spec].append(stats(a, b))
            if n - 5 >= frames:
                break
    finally:
        c.close()
        gpu.set_chroma(BGR24_CHROMA)

    def ms(k, i):
        return np.mean([x[i] for x in t[k]]) * 1000

    print("\n" + "=" * 78)
    print(f"ĐỔI MÀU GPU so với CPU — {cam} {frame.width}x{frame.height}, {n - 5} frame")
    print("=" * 78)
    print("  Thời gian mỗi frame:           chờ (wall)    CPU thật bị chiếm")
    for m in MODES:
        print(f"    {m:6s} CPU (cách hiện tại) : {ms('cpu_' + m, 0):6.1f} ms    {ms('cpu_' + m, 1):6.1f} ms")
        print(f"    {m:6s} GPU                 : {ms('gpu_' + m, 0):6.1f} ms    {ms('gpu_' + m, 1):6.1f} ms")
    d = np.mean(tv, axis=0)
    print(f"\n  tv    : lệch TB={d[0]:.3f}  max={int(max(x[1] for x in tv)):3d}  lệch>2={d[2]:.3f}%   (phải = 0)")
    print("\n  bgr24 — các kiểu nội suy chroma (vị trí ngang, dọc, kiểu, a), xếp từ khớp nhất:")
    rank = sorted(var.items(), key=lambda kv: np.mean([x[0] for x in kv[1]]))
    for spec, lst in rank:
        d = np.mean(lst, axis=0)
        mark = "  ← đang dùng" if spec == BGR24_CHROMA else ""
        print(f"    {spec:22s} lệch TB={d[0]:.3f}  max={int(max(x[1] for x in lst)):3d}  lệch>2={d[2]:.3f}%{mark}")
    print(f"\n  Khớp nhất: CAMHUB_BGR24_CHROMA=\"{rank[0][0]}\"")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        check(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
        sys.exit(0)
    sys.exit(main())
