# Digital Intelligence Backend -- VM Setup Guide

Complete step-by-step guide to deploy the DI-BE platform on an Ubuntu/Debian VM
with an NVIDIA RTX 5090 GPU.

**Target VM specs (verified):**

| Resource | Value |
|----------|-------|
| vCPUs | 255 |
| RAM | 503 GB |
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| Disk | 3.7 TB NVMe |
| OS | Ubuntu (Docker/overlay root) |
| Python | 3.12.3 (virtualenv `DI-BE` already created) |
| CUDA Driver | 590.48.01 / CUDA 13.1 |

---

## 1. System-Level Dependencies

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libx11-6 \
    libgif-dev \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    unzip \
    p7zip-full
```

Verify FFmpeg:

```bash
ffmpeg -version
```

---

## 2. Docker Engine

If Docker is not already installed on the VM:

```bash
# Add Docker's official GPG key and repo
sudo apt install -y ca-certificates gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify:

```bash
docker --version
docker compose version
```

If running as root you can skip this, otherwise add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3. Ollama (Local LLM Server)

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start the Ollama service

```bash
# Start Ollama in the background
ollama serve &
```

Wait a few seconds for it to start, then pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull llava:latest
```

Verify both models are available:

```bash
ollama list
```

You should see `llama3.1:8b` and `llava:latest` in the output.

> **Note:** Ollama will automatically use your RTX 5090 GPU for inference.
> The two models together need ~10-12 GB VRAM during inference.

---

## 4. Clone the Project

```bash
cd /workspace
git clone <your-repo-url> DI-BE
cd DI-BE
```

If the project is already on the VM (e.g., uploaded via SCP/rsync), just `cd` into it:

```bash
cd /workspace/DI-BE
```

---

## 5. Activate the Virtual Environment and Install Python Dependencies

```bash
workon DI-BE
```

### Install PyTorch with CUDA support (must be done BEFORE requirements.txt)

The `requirements.txt` lists `torch` without a CUDA index URL, so pip would install
the CPU-only wheel. Install the CUDA build explicitly first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> This installs PyTorch with CUDA 12.8 support which is compatible with the
> CUDA 13.1 driver on the VM (drivers are backward-compatible).

Verify GPU is visible to PyTorch:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

Expected output:

```
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
VRAM: 32.0 GB
```

### Install all remaining Python dependencies

```bash
pip install -r requirements.txt
```

### Install NudeNet (requires separate install from GitHub)

```bash
pip install -U git+https://github.com/platelminto/NudeNet
```

### Install psutil (used by compute_config for hardware detection)

```bash
pip install psutil
```

---

## 6. Download Machine Learning Models

### 6a. Hugging Face Transformer Models (Auto-Download)

The application auto-downloads Hugging Face models on first startup via `setup.py`.
To pre-download them now so the first startup is faster:

```bash
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import os

models = {
    'models/classifier': ('joeddav/xlm-roberta-large-xnli', 'classification'),
    'models/toxic': ('akhooli/xlm-r-large-arabic-toxic', 'classification'),
    'models/emotion': ('cardiffnlp/twitter-xlm-roberta-base-sentiment', 'classification'),
    'models/embeddings': ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'embeddings'),
}

for path, (name, mtype) in models.items():
    if os.path.exists(path):
        print(f'Already exists: {path}')
        continue
    print(f'Downloading {name} -> {path}')
    os.makedirs(path, exist_ok=True)
    if mtype == 'embeddings':
        m = AutoModel.from_pretrained(name)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(name)
    t = AutoTokenizer.from_pretrained(name)
    m.save_pretrained(path)
    t.save_pretrained(path)
    print(f'Saved {name}')

print('All Hugging Face models downloaded.')
"
```

### 6b. DNN Face Detector (OpenCV Caffe model)

```bash
mkdir -p models/dnn-face-detector

wget -O models/dnn-face-detector/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -O models/dnn-face-detector/res10_300x300_ssd_iter_140000.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### 6c. YOLO Object Detection Models

```bash
mkdir -p models/yolo12x-object-detector

wget -O models/yolo12x-object-detector/yolo12x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt
```

Optional (fallback models, downloaded if primary fails):

```bash
wget -O models/yolo12x-object-detector/yolo12l.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt

wget -O models/yolo12x-object-detector/yolo12m.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt

wget -O models/yolo12x-object-detector/yolo12s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt

wget -O models/yolo12x-object-detector/yolo12n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt
```

### 6d. CLIP Model (Sentence Transformers -- auto-cached)

The CLIP model (`clip-ViT-B-32`) is downloaded automatically on first use by
`sentence-transformers`. No manual step needed.

### 6e. Whisper Model (auto-downloaded)

The Whisper `small` model is downloaded automatically by the `openai-whisper`
package on first transcription. No manual step needed. To pre-download:

```bash
python -c "import whisper; whisper.load_model('small', device='cpu'); print('Whisper small downloaded')"
```

---

## 7. Configure the `.env` File

Edit the `.env` file in the project root:

```bash
nano .env
```

Set the following values for the production VM:

```env
# Platform & Device
PLATFORM="linux"
DI_DEVICE="auto"

# Environment
ENVIRONMENT="production"
DEBUG=False

# MongoDB (update with your actual credentials)
MONGO_USERNAME="your_username"
MONGO_PASSWORD="your_password"
MONGO_HOST="your_mongo_host"
MONGO_DATABASE="DigitalIntelligence"

# Qdrant (update with your actual credentials)
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"

# Redis (local via Docker)
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/0"
REDIS_URL="redis://localhost:6379/0"

# AI/ML
LOCAL_MODELS=True
DI_DEVICE="auto"

# Server
HOST="0.0.0.0"
PORT=8000
```

Key settings for the RTX 5090 VM:

| Setting | Value | Reason |
|---------|-------|--------|
| `PLATFORM` | `linux` | Correct platform detection |
| `DI_DEVICE` | `auto` | System auto-selects GPU with VRAM checks per model |
| `ENVIRONMENT` | `production` | Production logging & behavior |

---

## 8. Start Redis and Flower via Docker Compose

```bash
cd /workspace/DI-BE
docker compose up -d
```

Verify the containers are running:

```bash
docker compose ps
```

Expected output:

```
NAME         IMAGE              STATUS
di_redis     redis:7-alpine     Up (healthy)
di_flower    mher/flower:2.0    Up
```

Verify Redis connectivity:

```bash
docker compose exec redis redis-cli ping
```

Expected: `PONG`

---

## 9. Create Database Indexes

```bash
python database_scripts/create_ufdr_indexes.py
```

---

## 10. Create Required Directories

```bash
mkdir -p data logs logo
```

---

## 11. Start the Application

Open three separate terminal sessions (or use `tmux`/`screen`).

### Terminal 1: Ollama (if not already running as a system service)

```bash
workon DI-BE
cd /workspace/DI-BE
ollama serve
```

### Terminal 2: Celery Worker

```bash
workon DI-BE
cd /workspace/DI-BE
chmod +x start_celery_worker.sh
./start_celery_worker.sh
```

On this VM (255 vCPUs), Celery will auto-set concurrency to `32` threads
(the `min(cpu_count * 2, 32)` cap). You can override this with:

```bash
export PARALLEL_MAX_WORKERS=32
```

### Terminal 3: FastAPI Server

```bash
workon DI-BE
cd /workspace/DI-BE
uvicorn server:app --host 0.0.0.0 --port 8000
```

For production with multiple Uvicorn workers:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 12. Verify Everything Is Working

### Check GPU utilization

```bash
nvidia-smi
```

After uploading a case with media files, you should see GPU memory being used by
the Python process (PyTorch models loaded via ModelRegistry).

### Check Celery worker status

Open Flower dashboard in a browser:

```
http://<VM_IP>:5555
```

### Check FastAPI docs

```
http://<VM_IP>:8000/docs
```

### Quick health check

```bash
curl http://localhost:8000/docs
```

---

## 13. Using tmux for Persistent Sessions (Recommended)

Since VM SSH sessions can disconnect, use `tmux` to keep processes running:

```bash
sudo apt install -y tmux
```

```bash
# Create a new session
tmux new-session -s di

# Split into panes (Ctrl+B then %)
# Pane 1: ollama serve
# Pane 2: ./start_celery_worker.sh
# Pane 3: uvicorn server:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t di
```

---

## Quick Reference: Startup Commands

After everything is installed, the daily startup sequence is:

```bash
# 1. Start Docker services (Redis + Flower)
cd /workspace/DI-BE
docker compose up -d

# 2. Start Ollama (if not running as systemd service)
ollama serve &

# 3. Start Celery worker
workon DI-BE
cd /workspace/DI-BE
./start_celery_worker.sh &

# 4. Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

### "Cannot re-initialize CUDA in forked subprocess"

This should not occur with the current setup (threads pool). If it does, verify:

```bash
grep "worker_pool" celery_app.py
# Should show: "worker_pool": "threads"
```

### Models loading on CPU instead of GPU

Check the device detection:

```bash
python -c "
from utils.helpers import get_optimal_device
print(f'Device: {get_optimal_device()}')
"
```

If it returns `cpu`, check:
1. `.env` has `DI_DEVICE=auto` (not `cpu`)
2. `nvidia-smi` shows the GPU
3. PyTorch sees CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### dlib / face_recognition build errors

```bash
sudo apt install -y cmake libopenblas-dev liblapack-dev libx11-dev
pip install dlib --no-cache-dir
pip install face-recognition --no-cache-dir
```

### Ollama connection refused

```bash
# Check if Ollama is running
pgrep -f ollama

# If not, start it
ollama serve &

# Verify models are pulled
ollama list
```

### Redis connection refused

```bash
docker compose ps    # Check if container is running
docker compose up -d # Restart if needed
```
