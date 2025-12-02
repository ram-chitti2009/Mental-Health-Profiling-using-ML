# Use PyTorch base image with CUDA support (for GPU)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (minimal set for d3.py, no version pins)
RUN pip install --no-cache-dir \
    torch \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    seaborn \
    optuna \
    torch-lr-finder

# Copy only what's needed: script + data file
COPY d3.py /workspace/
COPY D3_Academic_processed.csv /workspace/

ENV PYTHONUNBUFFERED=1

# Run d3.py
CMD ["python", "d3.py"]

