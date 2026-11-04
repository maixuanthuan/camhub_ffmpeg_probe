#!/bin/bash
# camhub_monitor.sh — mỗi 60s ghi CPU từng container + GPU vào monitor.csv để tính trung bình sau.
#   Lọc container: CAMHUB_MONITOR_FILTER (regex tên container, mặc định: tất cả)
# Chỉ ĐỌC số liệu (docker stats, nvidia-smi), không đụng container nào.
#   Bật : nohup bash tools/camhub_monitor.sh >/dev/null 2>&1 &     (chạy trong thư mục camhub/)
#   Tắt : pkill -f camhub_monitor.sh
#   Kết quả: tools/monitor.csv
OUT="$(cd "$(dirname "$0")" && pwd)/monitor.csv"
[ -f "$OUT" ] || echo "time,item,value" > "$OUT"
while true; do
  ts=$(date '+%F %T')
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}}' \
    | grep -E "^(${CAMHUB_MONITOR_FILTER:-.*})," \
    | sed 's/%//; s/^/'"$ts"',cpu_/' >> "$OUT"
  nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.decoder --format=csv,noheader,nounits \
    | awk -F', ' -v t="$ts" '{print t",gpu_mem_mb,"$1; print t",gpu_util_%,"$2; print t",nvdec_%,"$3}' >> "$OUT"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
    | awk -F', ' -v t="$ts" '$2 ~ /ffmpeg/ {n++; m+=$3} END {print t",ffmpeg_count,"n+0; print t",ffmpeg_mem_mb,"m+0}' >> "$OUT"
  sleep 60
done
