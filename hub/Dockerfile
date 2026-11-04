# camhub — image của hub.
# FFmpeg nằm sẵn trong wheel PyAV; NVDEC (libnvcuvid) do nvidia runtime mount vào
# khi NVIDIA_DRIVER_CAPABILITIES có "video". Code (hub.py, camhub_client.py) mount
# từ ngoài vào → sửa code chỉ cần restart, không phải build lại.
#
# cupy + NVRTC/cudart (CUDA 12, pip): biên dịch kernel đổi màu NV12→BGR trên GPU.
# Không cài cả bộ CUDA toolkit — chỉ 2 thư viện kernel cần. Driver máy (CUDA 13.x)
# chạy được CUDA 12. Thiếu/lỗi thì hub tự đổi màu bằng CPU như trước.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        av==19.0.2 \
        numpy==2.5.4 \
        opencv-python-headless==5.0.0.93 \
        cupy-cuda12x==14.2.0 \
        "nvidia-cuda-nvrtc-cu12==12.9.*" \
        "nvidia-cuda-runtime-cu12==12.9.*"

# để cupy chắc chắn tìm thấy NVRTC / cudart cài bằng pip
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

WORKDIR /app
CMD ["python3", "-u", "hub.py"]
