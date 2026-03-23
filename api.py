"""
UTRNet PDF OCR API
==================
POST /ocr-pdf   — multipart/form-data with field "pdf" (PDF file)
                  Returns a ZIP containing one .txt file per page.

GET  /health    — liveness check
"""

import os, io, sys, zipfile, traceback, tempfile

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
from ultralytics import YOLO

# ── Silence argparse-heavy UTRNet modules ──────────────────────────────────
sys.argv = [sys.argv[0]]
from model import Model
from utils import CTCLabelConverter
from read import text_recognizer

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

# ── UTRNet recognition model ───────────────────────────────────────────────
print("[BOOT] Loading UTRNet-Large recognition model...", flush=True)
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
state_dict = torch.load(UTRNET_MODEL_PATH, map_location=device)
recognition_model.load_state_dict(state_dict)
recognition_model.eval()
print("[BOOT] UTRNet-Large ready.", flush=True)

# ── YOLOv8 text detection model ────────────────────────────────────────────
print("[BOOT] Loading YOLOv8 Urdu line detector...", flush=True)
detection_model = YOLO(YOLO_MODEL_PATH)
print("[BOOT] YOLOv8 ready.", flush=True)

# ── Core pipeline ──────────────────────────────────────────────────────────
def process_page(page_image: Image.Image) -> str:
    """
    Given a PIL image of one PDF page:
    1. Run YOLOv8 to detect text line bounding boxes
    2. Sort boxes top-to-bottom (reading order for Urdu = RTL, but line order is top→bottom)
    3. Crop each line and run UTRNet recognition
    4. Return the page's full text as a string
    """
    # ── Detection ──────────────────────────────────────────────────────────
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

    # Sort top-to-bottom by y1 coordinate
    bounding_boxes.sort(key=lambda b: b[1])

    # ── Recognition ────────────────────────────────────────────────────────
    lines = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        # Add a small vertical padding to avoid clipping ascenders/descenders
        pad = 4
        crop = page_image.crop((
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(page_image.width,  x2 + pad),
            min(page_image.height, y2 + pad),
        ))
        text = text_recognizer(crop, recognition_model, converter, device)
        if text.strip():
            lines.append(text.strip())

    return "\n".join(lines)


def pdf_to_page_images(pdf_bytes: bytes, dpi: int = 200):
    """Convert PDF bytes to a list of PIL Images, one per page."""
    return convert_from_bytes(pdf_bytes, dpi=dpi, fmt="RGB")


# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# Max upload: 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


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
        print(f"[OCR] Converting PDF to images...", flush=True)
        pages = pdf_to_page_images(pdf_bytes, dpi=200)
        print(f"[OCR] {len(pages)} pages found.", flush=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF conversion failed: {e}"}), 500

    # Build ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, page_img in enumerate(pages, start=1):
            page_num = str(i).zfill(len(str(len(pages))))
            txt_filename = f"{original_name}_page_{page_num}.txt"
            print(f"[OCR] Processing page {i}/{len(pages)}...", flush=True)
            try:
                page_text = process_page(page_img)
            except Exception as e:
                traceback.print_exc()
                page_text = f"[ERROR on page {i}: {e}]"

            # Write with UTF-8 BOM so Windows Notepad renders Urdu correctly
            zf.writestr(txt_filename, "\ufeff" + page_text)

    zip_buffer.seek(0)
    zip_name = f"{original_name}_ocr.zip"
    print(f"[OCR] Done. Returning ZIP: {zip_name}", flush=True)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
