# --------------------------------------------------------
# 1. Base image: PyTorch + CUDA 12.1 + cuDNN 8
# --------------------------------------------------------
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# --------------------------------------------------------
# 2. Environment setup
# --------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------
# 3. Install system dependencies
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        ninja-build \
        vim \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# 4. Copy requirements and install Python dependencies
# --------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# --------------------------------------------------------
# 5. Copy project code and assets
# --------------------------------------------------------
COPY inference.py /app/inference.py
COPY data_prepare.py /app/data_prepare.py
COPY noiseCancel.py /app/noiseCancel.py
COPY physical_resolution.py /app/physical_resolution.py
COPY reconstruction.py /app/reconstruction.py
COPY system_matrix.py /app/system_matrix.py
COPY model/ /app/model/
COPY utils/ /app/utils/
COPY ldm/ /app/ldm/
COPY configs/ /app/configs/

# --------------------------------------------------------
# 6. Entrypoint (default inference)
# --------------------------------------------------------
ENTRYPOINT ["python", "/app/inference.py"]
# CMD ["python", "/app/inference.py"]