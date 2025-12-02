# RunPod Docker Setup for d3.py

## Quick Start

1. **Build Docker Image:**
   ```bash
   docker build -t d3-runner .
   ```

2. **Test Locally (optional):**
   ```bash
   docker run --gpus all d3-runner
   ```

3. **Push to Docker Hub (for RunPod):**
   ```bash
   docker tag d3-runner yourusername/d3-runner:latest
   docker push yourusername/d3-runner:latest
   ```

## RunPod Setup

### Option 1: Use Docker Hub Image
1. Go to RunPod → Create Pod
2. Select "Docker Hub" as image source
3. Enter: `yourusername/d3-runner:latest`
4. Set GPU: A100 or RTX 4090 (recommended)
5. Start Pod

### Option 2: Upload Files Directly
1. Create Pod with PyTorch template
2. Upload files via RunPod file manager:
   - `d3.py`
   - `D3_Academic_processed.csv`
   - `requirements.txt`
3. SSH into pod and run:
   ```bash
   pip install -r requirements.txt
   python d3.py
   ```

## What This Runs

- **Scout Phase**: Optuna optimization (150 trials, 3-fold CV, 15 epochs)
- **Focused Grid**: Narrowed hyperparameter search (10-fold CV, 50-100 epochs)
- **Output**: Best configuration and profiles for D3-Academic dataset

## Expected Runtime

- **Scout Phase**: ~60-90 minutes (on A100)
- **Focused Grid**: ~4-6 hours (on A100)
- **Total**: ~5-7 hours

## Files in Docker Image

- `d3.py` - Main script (copied into image)
- `D3_Academic_processed.csv` - Dataset (can be mounted as volume instead)

**Note:** The Dockerfile only copies `d3.py`. For the CSV file, you have two options:
1. **Copy into image** (current): Add `COPY D3_Academic_processed.csv /workspace/` to Dockerfile
2. **Mount as volume** (flexible): Upload CSV to RunPod and mount it when running

## Notes

- No version pins in requirements.txt (uses latest compatible versions)
- Paths fixed from `/content/` (Colab) to relative paths
- Removed `!pip install` lines (handled by Dockerfile)

