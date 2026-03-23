# UTRNet PDF OCR — Coolify Container

End-to-end Urdu PDF OCR. Upload a PDF, get back a ZIP with one `.txt` file per page.

**Pipeline:** `PDF → Page Images (200 DPI) → YOLOv8 line detection → UTRNet-Large recognition → per-page .txt files → ZIP`

---

## Files

```
utrnet-pdf/
├── Dockerfile          # Python 3.10 slim, poppler, CPU PyTorch 2.0.1
├── docker-compose.yml  # Persistent model volume, healthcheck
├── entrypoint.sh       # Downloads both models on first boot, starts Flask
├── api.py              # Flask REST API — the full PDF pipeline
└── README.md
```

---

## Models downloaded automatically on first boot

| Model | Size | Source |
|---|---|---|
| UTRNet-Large (`best_norm_ED.pth`) | ~300 MB | Google Drive |
| YOLOv8 Urdu detector (`yolov8m_UrduDoc.pt`) | ~50 MB | GitHub Releases |

Both are saved to the persistent volume `/app/models` — downloaded once, reused forever.

---

## Deploying on Coolify

1. Push these 4 files to a new GitHub repo (e.g. `utrnet-pdf`)
2. Coolify → **New Resource → Application → Git Repository**
3. Build pack: `Dockerfile` | Port: `5000`
4. Add **Persistent Volume**:
   - Name: `utrnet_models`
   - Mount path: `/app/models`
5. Deploy — first boot downloads ~350 MB (watch logs, takes 2–5 min on typical VPS)
6. Health check: `GET /health` → `{"status":"ok","models":["UTRNet-Large","YOLOv8-UrduDoc"]}`

---

## API Usage

### POST /ocr-pdf

Send a PDF, receive a ZIP of `.txt` files.

```bash
curl -X POST http://YOUR_VPS:5000/ocr-pdf \
  -F "pdf=@/path/to/urdu_book.pdf" \
  -o output.zip
```

Then unzip:
```bash
unzip output.zip
# urdu_book_page_001.txt
# urdu_book_page_002.txt
# ...
```

**Python example:**
```python
import requests

with open("urdu_book.pdf", "rb") as f:
    resp = requests.post(
        "http://YOUR_VPS:5000/ocr-pdf",
        files={"pdf": ("urdu_book.pdf", f, "application/pdf")}
    )

with open("output.zip", "wb") as out:
    out.write(resp.content)
```

### GET /health

```bash
curl http://YOUR_VPS:5000/health
# {"status": "ok", "models": ["UTRNet-Large", "YOLOv8-UrduDoc"]}
```

---

## Performance expectations (CPU VPS)

| Pages | Approximate time |
|---|---|
| 1 page | ~15–30 seconds |
| 10 pages | ~3–5 minutes |
| 50 pages | ~15–25 minutes |

CPU inference is slow for large books. For a 200-page Urdu book, consider submitting it as a background job rather than a synchronous HTTP request (or add a queue with Redis + RQ on top of this API).

---

## Output format

Each `.txt` file contains the recognized Urdu text for that page, one line per detected text line, top-to-bottom reading order. Files are UTF-8 with BOM (so Windows Notepad renders Urdu correctly).

---

## Notes

- Max PDF upload size: **100 MB**
- Pages are rendered at **200 DPI** — higher DPI improves accuracy on small text but increases processing time
- The `threaded=False` Flask setting ensures only one PDF is processed at a time (avoids memory issues on low-RAM VPS)
- License: UTRNet and YOLOv8-UrduDoc are CC BY-NC-SA 4.0 — non-commercial/research use only
