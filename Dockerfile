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

# Clone the ORIGINAL UTRNet repo — has the correct HRNet model.py
# that matches the UTRNet-Large weights (best_norm_ED.pth)
RUN git clone https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git utrnet

WORKDIR /app/utrnet

# Install CPU-only PyTorch 1.9.1 — matches the original UTRNet repo
RUN pip install --no-cache-dir \
    torch==1.9.1+cpu \
    torchvision==0.10.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining deps
RUN pip install --no-cache-dir \
    ultralytics==8.1.8 \
    pdf2image==1.17.0 \
    Pillow \
    numpy \
    opencv-python-headless \
    six \
    natsort \
    nltk \
    lmdb \
    flask \
    gdown

# Model storage dir
RUN mkdir -p /app/models

# Copy our API and entrypoint
COPY api.py /app/utrnet/api.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
