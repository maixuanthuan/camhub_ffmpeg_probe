# -*- coding: utf-8 -*-
"""
camhub_client.py
================
Đọc frame từ camhub thay cho cv2.VideoCapture / ffmpeg tự chạy.
MỘT file, chỉ cần numpy. Chạy được Python 3.8+.

Dùng y như cv2.VideoCapture:

    from camhub_client import HubCapture

    cap = HubCapture("rtsp://user:pass@host:port/live", fps=8)   # đúng link đang dùng
    # hoặc theo tên trong camhub.env:  HubCapture("cam7", fps=8)
    while True:
        ret, frame = cap.read()        # numpy HxWx3 BGR, như cv2
        if not ret:
            ...                        # như mất camera → code reconnect cũ tự xử lý
    cap.release()

Tham số:
    fps          frame/giây muốn nhận (như filter fps= của ffmpeg). 0 = mọi frame.
    mode         "bgr24" (mặc định) — giống ffmpeg "...,hwdownload,format=nv12,format=bgr24"
                 "tv"                — giống ffmpeg "colorspace_cuda=range=tv" + cv2 NV12→BGR
    read_timeout giây không có frame mới → read() trả (False, None)
    open_timeout giây chờ frame đầu tiên lúc mở → quá thì isOpened() = False

Project chạy trong docker: thêm vào docker-compose.yml
    volumes:
      - /dev/shm/camhub:/camhub

Chạy thử (trong container có mount /camhub):
    python3 camhub_client.py cam1 --fps 8 --seconds 20
"""

import os
import re
import json
import mmap
import time
import socket
import struct
import logging
import threading

import numpy as np

logger = logging.getLogger("camhub")

MAGIC       = b"CHB1"
HEADER_SIZE = 64
HEADER_FMT  = "<4sIIIIIQQ"          # magic ver w h c nslots frame_bytes seq
SEQ_OFFSET  = 32
CONSUMED_OFFSET = 40                # ghi seq vừa lấy → hub chỉ đổi màu frame mới khi có người lấy
SLOT_HDR    = 16
MODES       = ("bgr24", "tv")
HEARTBEAT   = 2.0                   # giây — làm mới đơn xin, hub coi >10s là client chết

CAP_PROP_FRAME_WIDTH  = 3           # cùng số với cv2.CAP_PROP_*
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS          = 5


def _hub_dir():
    d = os.environ.get("CAMHUB_DIR")
    if d:
        return d
    for d in ("/camhub", "/dev/shm/camhub"):
        if os.path.isdir(d):
            return d
    return "/camhub"


def _mask(url):
    if "://" in url and "@" in url:
        head, rest = url.split("://", 1)
        return "%s://***@%s" % (head, rest.rsplit("@", 1)[1])
    return url


def cam_addr(url):
    """rtsp://user:pass@Host:port/path → host:port/path (giống hub.py)."""
    u = url.strip()
    if "://" in u:
        u = u.split("://", 1)[1]
    host_port, _, path = u.partition("/")
    if "@" in host_port:
        host_port = host_port.rsplit("@", 1)[1]
    host, _, port = host_port.partition(":")
    return "%s:%s/%s" % (host.lower(), port or "554", path.rstrip("/"))


class _Buffer:
    """Đọc <cam>.<mode>.buf do hub ghi. Tự mở lại khi hub tạo file mới."""

    def __init__(self, path):
        self.path = path
        self.mm = None
        self.ino = None
        self.shape = None
        self.frame_bytes = 0
        self.nslots = 0
        self.ack_mm = None
        self.ack = False            # chỉ client xin mọi frame (fps=0) mới báo "đã lấy"

    def _open(self):
        self.close()
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False
        try:
            st = os.fstat(fd)
            if st.st_size < HEADER_SIZE:
                return False
            try:
                mm = mmap.mmap(fd, st.st_size, access=mmap.ACCESS_READ)
            except Exception:
                return False
        finally:
            os.close(fd)
        # mở thêm bản ghi được để báo "đã lấy" (không có quyền ghi thì thôi, vẫn đọc được)
        self.ack_mm = None
        try:
            fdw = os.open(self.path, os.O_RDWR)
            try:
                self.ack_mm = mmap.mmap(fdw, HEADER_SIZE)
            finally:
                os.close(fdw)
        except OSError:
            pass
        magic, _ver, w, h, c, nslots, fb, _seq = struct.unpack_from(HEADER_FMT, mm, 0)
        if magic != MAGIC:
            mm.close()
            return False
        self.mm, self.ino = mm, st.st_ino
        self.shape = (h, w, c) if c > 1 else (h, w)
        self.frame_bytes, self.nslots = fb, nslots
        return True

    def check_reopen(self):
        """Hub restart / đổi kích thước → file mới (inode khác)."""
        try:
            ino = os.stat(self.path).st_ino
        except FileNotFoundError:
            self.close()
            return
        if ino != self.ino:
            self._open()

    def peek(self, after_seq):
        """(seq, ts) của frame mới nhất nếu mới hơn after_seq, không thì None. Chưa chép."""
        if self.mm is None and not self._open():
            return None
        mm = self.mm
        seq = struct.unpack_from("<Q", mm, SEQ_OFFSET)[0]
        if seq == 0 or seq == after_seq:
            return None
        off = HEADER_SIZE + (seq % self.nslots) * (SLOT_HDR + self.frame_bytes)
        if struct.unpack_from("<Q", mm, off)[0] != seq:
            return None                                       # hub đang ghi slot này
        return seq, struct.unpack_from("<d", mm, off + 8)[0]

    def take(self, seq):
        """Chép frame seq. None nếu slot bị ghi đè lúc đang chép. Lấy được thì báo hub."""
        mm = self.mm
        off = HEADER_SIZE + (seq % self.nslots) * (SLOT_HDR + self.frame_bytes)
        frame = np.frombuffer(mm, dtype=np.uint8, count=self.frame_bytes,
                              offset=off + SLOT_HDR).reshape(self.shape).copy()
        if struct.unpack_from("<Q", mm, off)[0] != seq:
            return None
        if self.ack and self.ack_mm is not None:
            try:
                if struct.unpack_from("<Q", self.ack_mm, CONSUMED_OFFSET)[0] < seq:
                    struct.pack_into("<Q", self.ack_mm, CONSUMED_OFFSET, seq)
            except (ValueError, struct.error):
                pass
        return frame

    def close(self):
        for m in (self.mm, self.ack_mm):
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
        self.mm = None
        self.ack_mm = None
        self.ino = None


class HubCapture:
    """Thay cv2.VideoCapture: isOpened() / read() / grab() / retrieve() / release() / get() / set().

    Tham số thứ 2 (vị trí) được bỏ qua để đổi thẳng được
    cv2.VideoCapture(url, cv2.CAP_FFMPEG) → HubCapture(url, cv2.CAP_FFMPEG).
    Các tham số camhub (fps, mode, ...) truyền theo tên."""

    def __init__(self, source, api_preference=None, *, fps=0, mode="bgr24",
                 read_timeout=10.0, open_timeout=20.0, hub_dir=None):
        if mode not in MODES:
            raise ValueError("mode phải là một trong %s" % (MODES,))
        self.source = source
        self.fps = float(fps or 0)
        self.mode = mode
        self.read_timeout = float(read_timeout)
        self.hub_dir = hub_dir or _hub_dir()
        self.cam = None
        self._opened = False
        self._released = False
        self._last_seq = 0
        self._next_t = None
        self._pending = None
        self.last_ts = 0.0           # giờ hub giải mã xong frame vừa trả (time.time())
        self._buf = None
        self._want = None
        self._hb_stop = threading.Event()
        self._hb = None
        self.width = self.height = 0

        self.cam = self._resolve(source)
        if self.cam is None:
            return

        cid = "%s-%d-%x" % (socket.gethostname(), os.getpid(), id(self))
        self._want = os.path.join(self.hub_dir, "want", "%s.%s.%s" % (self.cam, mode, cid))
        self._write_want()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True,
                                    name="camhub-hb-%s" % self.cam)
        self._hb.start()
        self._buf = _Buffer(os.path.join(self.hub_dir, "%s.%s.buf" % (self.cam, mode)))
        self._buf.ack = not self.fps

        first = self._wait_frame(open_timeout)
        if first is None:
            logger.error("[camhub] %s: hub chưa đưa frame sau %.0fs — xem status.json của hub"
                         % (self.cam, open_timeout))
            self.release()
            return
        self._pending = first
        self.height, self.width = first.shape[:2]
        self._opened = True
        logger.info("[camhub] đã mở %s (%s) %dx%d mode=%s fps=%s"
                    % (self.cam, _mask(str(source)), self.width, self.height, mode,
                       self.fps or "mọi frame"))

    # ── tra camera ───────────────────────────────────────────────────────
    def _resolve(self, source):
        s = str(source).strip()
        if re.fullmatch(r"(?i)cam\d+", s):
            return "cam%d" % int(s[3:])
        path = os.path.join(self.hub_dir, "cameras.json")
        try:
            with open(path, encoding="utf-8") as f:
                cams = json.load(f)
        except FileNotFoundError:
            logger.error("[camhub] không thấy %s — camhub chưa chạy hoặc chưa mount "
                         "/dev/shm/camhub vào container" % path)
            return None
        except Exception as e:
            logger.error("[camhub] đọc %s lỗi: %s" % (path, e))
            return None
        addr = cam_addr(s)
        for name, info in cams.items():
            if info.get("addr") == addr:
                return name
        logger.error("[camhub] link %s chưa khai trong camhub.env của hub" % _mask(s))
        return None

    # ── đơn xin ──────────────────────────────────────────────────────────
    def _write_want(self):
        try:
            os.makedirs(os.path.dirname(self._want), exist_ok=True)   # hub chưa kịp tạo
            tmp = self._want + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"fps": self.fps, "ack": True}, f)   # ack: bản client biết báo "đã lấy"
            os.replace(tmp, self._want)
        except OSError as e:
            logger.warning("[camhub] không ghi được đơn xin %s: %s" % (self._want, e))

    def _heartbeat(self):
        while not self._hb_stop.wait(HEARTBEAT):
            self._write_want()

    # ── đọc frame ────────────────────────────────────────────────────────
    def _due_peek(self, ts):
        """Frame có ts này tới lượt nhận chưa (theo fps). Chưa đổi mốc."""
        if not self.fps or self._next_t is None:
            return True
        # cho sớm nửa chu kỳ (giống hub.py): frame về không đều, so chặt sẽ rơi frame
        return ts >= self._next_t - 0.5 / self.fps

    def _commit_due(self, ts):
        if not self.fps:
            return
        period = 1.0 / self.fps
        nt = self._next_t
        self._next_t = ts + period if nt is None or ts - nt > period else nt + period

    def _wait_frame(self, timeout):
        end = time.monotonic() + timeout
        last_check = 0.0
        while not self._released:
            got = self._buf.peek(self._last_seq)
            if got is not None:
                seq, ts = got
                if not self._due_peek(ts):
                    self._last_seq = seq        # chưa tới lượt (fps) → bỏ qua
                    continue
                frame = self._buf.take(seq)
                if frame is None:
                    continue                    # bị ghi đè lúc chép → lấy lại frame mới hơn
                self._last_seq = seq
                self._commit_due(ts)
                self.last_ts = ts
                return frame
            now = time.monotonic()
            if now >= end:
                return None
            if now - last_check > 0.5:
                last_check = now
                self._buf.check_reopen()
            time.sleep(0.003)
        return None

    def isOpened(self):
        return self._opened and not self._released

    def read(self):
        """(True, frame) khi có frame MỚI; (False, None) nếu quá read_timeout không có."""
        if not self.isOpened():
            return False, None
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return True, frame
        frame = self._wait_frame(self.read_timeout)
        if frame is None:
            if not self._released:
                logger.warning("[camhub] %s: không có frame mới sau %.0fs"
                               % (self.cam, self.read_timeout))
            return False, None
        return True, frame

    def grab(self):
        """Như cv2: lấy frame kế tiếp, chưa trả ra. Lấy ra bằng retrieve()."""
        ok, frame = self.read()
        self._grabbed = frame if ok else None
        return ok

    def retrieve(self, image=None, flag=None):
        frame, self._grabbed = getattr(self, "_grabbed", None), None
        return (frame is not None), frame

    def get(self, prop):
        if prop == CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == CAP_PROP_FPS:
            return float(self.fps) if self.fps else self._camera_fps()
        return 0.0

    def _camera_fps(self):
        """fps=0 (mọi frame): trả fps thật của camera như cv2 (không trả 0 — code hay chia cho fps)."""
        try:
            with open(os.path.join(self.hub_dir, "status.json"), encoding="utf-8") as f:
                fps = json.load(f)["cameras"][self.cam].get("dec_fps") or 0
            if fps > 0:
                return float(round(fps))
        except Exception:
            pass
        return 25.0

    def set(self, prop, value):
        return False      # cấu hình nằm ở hub, giữ cho code gọi cap.set(...) không lỗi

    def release(self):
        if self._released:
            return
        self._released = True
        self._opened = False
        self._hb_stop.set()
        if self._want:
            for p in (self._want, self._want + ".tmp"):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if self._buf is not None:
            self._buf.close()


# ── chạy thử ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Thử đọc frame từ camhub")
    p.add_argument("source", help="cam1 / cam2 ... hoặc link rtsp đầy đủ")
    p.add_argument("--fps", type=float, default=8)
    p.add_argument("--mode", choices=MODES, default="bgr24")
    p.add_argument("--seconds", type=int, default=20)
    a = p.parse_args()

    cap = HubCapture(a.source, fps=a.fps, mode=a.mode)
    if not cap.isOpened():
        raise SystemExit("Không mở được — xem log ở trên")
    n, lags, t0 = 0, [], time.time()
    while time.time() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            print("read() = False")
            continue
        n += 1
        lags.append((time.time() - cap.last_ts) * 1000)
        if n % max(1, int(a.fps or 25)) == 0:
            print("%5.1fs  %d frame  %s  %.1f fps  trễ hub→client %.1f ms"
                  % (time.time() - t0, n, frame.shape, n / (time.time() - t0),
                     sum(lags) / len(lags)))
            lags = []
    cap.release()
    print("Xong: %d frame trong %ds = %.1f fps" % (n, a.seconds, n / a.seconds))
