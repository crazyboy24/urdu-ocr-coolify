#!/bin/bash
set -e

UTRNET_MODEL="/app/models/best_norm_ED.pth"
YOLO_MODEL="/app/models/yolov8m_UrduDoc.pt"

# --- UTRNet-Large weights (Google Drive) ---
if [ ! -f "$UTRNET_MODEL" ]; then
    echo "==> Downloading UTRNet-Large weights (~300 MB)..."
    python -c "
import gdown
gdown.download(
    'https://drive.google.com/uc?id=1xXG7vsSePBw4vtapIEdPWEZ-qrbR9Q9K',
    '/app/models/best_norm_ED.pth',
    quiet=False
)
"
    echo "==> UTRNet-Large downloaded."
else
    echo "==> UTRNet-Large already present."
fi

# --- YOLOv8 detection model (GitHub releases — direct download) ---
if [ ! -f "$YOLO_MODEL" ]; then
    echo "==> Downloading YOLOv8 Urdu text detector (~50 MB)..."
    wget -q --show-progress \
        -O /app/models/yolov8m_UrduDoc.pt \
        https://github.com/abdur75648/urdu-text-detection/releases/download/v1.0.0/yolov8m_UrduDoc.pt
    echo "==> YOLOv8 model downloaded."
else
    echo "==> YOLOv8 model already present."
fi

echo "==> Starting UTRNet PDF API on port 5000..."
exec python /app/utrnet/api.py
