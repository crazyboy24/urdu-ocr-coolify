FROM python:3.10-slim

# System deps: poppler for PDF→image, OpenCV headless, git, wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the End-To-End webapp (has model.py, read.py, utils.py, UrduGlyphs.txt all in one place)
RUN git clone https://github.com/abdur75648/End-To-End-Urdu-OCR-WebApp.git utrnet

WORKDIR /app/utrnet

# Install CPU-only PyTorch 2.0.1 first to avoid pulling CUDA wheels
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
RUN pip install --no-cache-dir \
    ultralytics==8.1.8 \
    pdf2image==1.17.0 \
    Pillow==10.2.0 \
    numpy==1.23.5 \
    opencv-python-headless==4.9.0.80 \
    six==1.16.0 \
    PyArabic==0.6.15 \
    arabic-reshaper==3.0.0 \
    flask \
    gdown

# Model storage dirs
RUN mkdir -p /app/models

# Copy our API and entrypoint
COPY api.py /app/utrnet/api.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
