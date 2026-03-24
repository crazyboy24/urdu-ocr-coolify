"""
UTRNet PDF OCR API
==================
POST /ocr-pdf   — multipart/form-data with field "pdf" (PDF file)
                  Returns a ZIP containing one .txt file per page.
GET  /health    — liveness check
"""

import os, io, sys, zipfile, traceback

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
from ultralytics import YOLO

# ── Silence argparse-heavy UTRNet modules at import time ──────────────────
sys.argv = [sys.argv[0]]

# The ORIGINAL UTRNet repo's modules (HRNet model)
from model import Model
from dataset import NormalizePAD
from utils import CTCLabelConverter

# ── Paths ──────────────────────────────────────────────────────────────────
UTRNET_MODEL_PATH = "/app/models/best_norm_ED.pth"
YOLO_MODEL_PATH   = "/app/models/yolov8m_UrduDoc.pt"
GLYPH_FILE        = "/app/utrnet/UrduGlyphs.txt"

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cpu")

# ── Character set ──────────────────────────────────────────────────────────
with open(GLYPH_FILE, "r", encoding="utf-8") as f:
    content = "".join(line.strip() for line in f.readlines())
content = content + " "

# ── Opt object — mirrors what the original UTRNet CLI passes ───────────────
class Opt:
    FeatureExtraction  = "HRNet"
    SequenceModeling   = "DBiLSTM"
    Prediction         = "CTC"
    imgH               = 32
    imgW               = 400
    PAD                = True
    input_channel      = 1
    output_channel     = 512
    hidden_size        = 256
    batch_max_length   = 100
    workers            = 0
    batch_size         = 1
    sensitive          = False
    rgb                = False
    character          = content
    num_class          = len(content) + 1   # +1 for CTC blank
    device             = torch.device("cpu")   # ← ADD THIS LINE

opt = Opt()

# ── UTRNet recognition model ───────────────────────────────────────────────
print("[BOOT] Loading UTRNet-Large recognition model...", flush=True)
recognition_model = Model(opt)
recognition_model = torch.nn.DataParallel(recognition_model)
state_dict = torch.load(UTRNET_MODEL_PATH, map_location=device)
recognition_model.load_state_dict(state_dict)
recognition_model = recognition_model.to(device)
recognition_model.eval()
print("[BOOT] UTRNet-Large ready.", flush=True)

converter  = CTCLabelConverter(content)
transform  = NormalizePAD((1, opt.imgH, opt.imgW))

# ── YOLOv8 text detection model ────────────────────────────────────────────
print("[BOOT] Loading YOLOv8 Urdu line detector...", flush=True)
detection_model = YOLO(YOLO_MODEL_PATH)
print("[BOOT] YOLOv8 ready.", flush=True)

# ── UTRNet inference for a single cropped line image ──────────────────────
def recognize_line(pil_image: Image.Image) -> str:
    img = pil_image.convert("L")
    w, h = img.size
    ratio = opt.imgH / float(h)
    new_w = min(int(w * ratio), opt.imgW)
    img = img.resize((new_w, opt.imgH), Image.BICUBIC)

    img_tensor = transform(img).unsqueeze(0).to(device)   # (1, 1, H, W)

    with torch.no_grad():
        preds = recognition_model(img_tensor, text=[], is_train=False)

    preds_size = torch.IntTensor([preds.size(1)])
    _, preds_index = preds.max(2)
    result = converter.decode(preds_index, preds_size)
    return result[0]

# ── Full page pipeline ─────────────────────────────────────────────────────
def process_page(page_image: Image.Image) -> str:
    results = detection_model.predict(
        source=page_image,
        conf=0.2,
        imgsz=1280,
        save=False,
        nms=True,
        device=device,
        verbose=False,
    )
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
    if not bounding_boxes:
        return ""

    bounding_boxes.sort(key=lambda b: b[1])   # top-to-bottom

    lines = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = 4
        crop = page_image.crop((
            max(0, x1 - pad), max(0, y1 - pad),
            min(page_image.width, x2 + pad), min(page_image.height, y2 + pad),
        ))
        text = recognize_line(crop)
        if text.strip():
            lines.append(text.strip())

    return "\n".join(lines)

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": ["UTRNet-Large", "YOLOv8-UrduDoc"]}), 200

@app.route("/ocr-pdf", methods=["POST"])
def ocr_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "Send a PDF as multipart field 'pdf'"}), 400

    pdf_file = request.files["pdf"]
    original_name = os.path.splitext(pdf_file.filename or "document")[0]
    pdf_bytes = pdf_file.read()

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200, fmt="RGB")
        print(f"[OCR] {len(pages)} pages.", flush=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF conversion failed: {e}"}), 500

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, page_img in enumerate(pages, start=1):
            page_num = str(i).zfill(len(str(len(pages))))
            print(f"[OCR] Processing page {i}/{len(pages)}...", flush=True)
            try:
                page_text = process_page(page_img)
            except Exception as e:
                traceback.print_exc()
                page_text = f"[ERROR on page {i}: {e}]"
            zf.writestr(f"{original_name}_page_{page_num}.txt", "\ufeff" + page_text)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{original_name}_ocr.zip",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
