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

## 2. Docker Engine (Skip on container-based VMs)

> **vast.ai / container-based VMs:** Docker cannot run inside these environments
> because the bridge networking driver needs iptables/NAT kernel access that the
> host does not expose. You will see an error like:
>
> ```
> failed to create NAT chain DOCKER: iptables failed: Permission denied
> ```
>
> **Skip this section entirely.** Redis and Flower will be installed natively in
> Step 9 instead.

If you are on a **bare-metal or full VM** (not a container), install Docker:

```bash
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

## 5. Install virtualenvwrapper and Create the Virtual Environment

### Install pip for the system Python (if not already present)

```bash
sudo apt install -y python3-pip python3-venv
```

### Install virtualenvwrapper

```bash
pip install virtualenv virtualenvwrapper
```

### Find the installed paths

The `~/.bashrc` config needs three paths: the Python binary, the `virtualenv`
binary, and the `virtualenvwrapper.sh` script. Find them:

```bash
which python3
# Example: /usr/bin/python3  or  /venv/main/bin/python3

find / -name "virtualenvwrapper.sh" 2>/dev/null
# Example: /usr/local/bin/virtualenvwrapper.sh  or  /venv/main/bin/virtualenvwrapper.sh

find / -name "virtualenv" -type f 2>/dev/null
# Example: /usr/local/bin/virtualenv  or  /venv/main/bin/virtualenv
```

### Add virtualenvwrapper config to `~/.bashrc`

Open the file:

```bash
nano ~/.bashrc
```

Add the following block at the **end** of the file. **Replace the paths** with
whatever the `find` / `which` commands above returned on your VM:

```bash
# ── virtualenvwrapper ──
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/venv/main/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/venv/main/bin/virtualenv
source /venv/main/bin/virtualenvwrapper.sh
```

> **Important:** Do NOT use `$(which virtualenv)` for `VIRTUALENVWRAPPER_VIRTUALENV`.
> That expression is evaluated when `~/.bashrc` loads, and if the directory
> containing `virtualenv` is not in your `PATH` at that moment, it resolves to
> empty and virtualenvwrapper fails with *"could not find virtualenv in your path"*.
> Always use the full absolute path.

Save and close (`Ctrl+O`, `Enter`, `Ctrl+X` in nano), then reload:

```bash
source ~/.bashrc
```

You should see the virtualenvwrapper startup messages. Verify the commands are
available:

```bash
which mkvirtualenv
which workon
which lsvirtualenv
```

### Create the `DI-BE` virtual environment

```bash
mkvirtualenv DI-BE -p python3
```

This creates `~/.virtualenvs/DI-BE/` and activates it automatically. You'll see
`(DI-BE)` in your prompt.

From now on, activate it any time with:

```bash
workon DI-BE
```

Other useful commands:

| Command | Description |
|---------|-------------|
| `workon DI-BE` | Activate the DI-BE environment |
| `deactivate` | Deactivate the current environment |
| `lsvirtualenv` | List all virtual environments |
| `rmvirtualenv DI-BE` | Delete the DI-BE environment |
| `mkvirtualenv <name>` | Create a new environment |

---

## 6. Install Python Dependencies

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

## 7. Download Machine Learning Models

### 7a. Hugging Face Transformer Models (Auto-Download)

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

### 7b. DNN Face Detector (OpenCV Caffe model)

```bash
mkdir -p models/dnn-face-detector

wget -O models/dnn-face-detector/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -O models/dnn-face-detector/res10_300x300_ssd_iter_140000.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### 7c. YOLO Object Detection Models

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

### 7d. CLIP Model (Sentence Transformers -- auto-cached)

The CLIP model (`clip-ViT-B-32`) is downloaded automatically on first use by
`sentence-transformers`. No manual step needed.

### 7e. Whisper Model (auto-downloaded)

The Whisper `small` model is downloaded automatically by the `openai-whisper`
package on first transcription. No manual step needed. To pre-download:

```bash
python -c "import whisper; whisper.load_model('small', device='cpu'); print('Whisper small downloaded')"
```

---

## 8. Configure the `.env` File

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

## 9. Start Redis and Flower

### Option A: Native install (required on container-based VMs like vast.ai)

```bash
sudo apt install -y redis-server
redis-server --daemonize yes --appendonly yes
```

Verify Redis is running:

```bash
redis-cli ping
```

Expected: `PONG`

Install Flower (Celery task dashboard):

```bash
pip install flower
```

Flower will be started alongside the other services in Step 12.

### Option B: Docker Compose (bare-metal / full VMs only)

If Docker is available (see Step 2), you can use the bundled compose file instead:

```bash
cd /workspace/DI-BE
docker compose up -d
```

Verify:

```bash
docker compose ps          # both di_redis and di_flower should be Up
docker compose exec redis redis-cli ping   # Expected: PONG
```

---

## 10. Create Database Indexes

```bash
python database_scripts/create_ufdr_indexes.py
```

---

## 11. Create Required Directories

```bash
mkdir -p data logs logo
```

---

## 12. Start the Application

Open four separate terminal sessions (or use `tmux`/`screen`).

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

> **Note:** The `start_celery_worker.sh` script flushes Redis via
> `docker compose exec redis redis-cli FLUSHALL`. On container-based VMs where
> Docker is not available, flush Redis natively before starting the worker:
>
> ```bash
> redis-cli FLUSHALL
> ```

On this VM (255 vCPUs), Celery will auto-set concurrency to `32` threads
(the `min(cpu_count * 2, 32)` cap). You can override this with:

```bash
export PARALLEL_MAX_WORKERS=32
```

### Terminal 3: Flower (Celery dashboard)

```bash
workon DI-BE
cd /workspace/DI-BE
celery -A celery_app flower --port=5555
```

### Terminal 4: FastAPI Server

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

## 13. Verify Everything Is Working

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

## 14. Using tmux for Persistent Sessions (Recommended)

Since VM SSH sessions can disconnect, use `tmux` to keep processes running:

```bash
sudo apt install -y tmux
```

```bash
# Create a new session
tmux new-session -s di

# Split into panes (Ctrl+B then % for vertical, Ctrl+B then " for horizontal)
# Pane 1: ollama serve
# Pane 2: redis-cli FLUSHALL && ./start_celery_worker.sh
# Pane 3: celery -A celery_app flower --port=5555
# Pane 4: uvicorn server:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t di
```

---

## Quick Reference: Startup Commands

After everything is installed, the daily startup sequence is:

```bash
# 1. Start Redis (skip if already running — check with: redis-cli ping)
redis-server --daemonize yes --appendonly yes

# 2. Start Ollama (if not running — check with: pgrep -f ollama)
ollama serve &

# 3. Flush stale tasks and start Celery worker
workon DI-BE
cd /workspace/DI-BE
redis-cli FLUSHALL
./start_celery_worker.sh &

# 4. Start Flower (Celery dashboard)
celery -A celery_app flower --port=5555 &

# 5. Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Shutdown and Free GPU Memory

When you're done and want to stop all services and release GPU VRAM:

```bash
# 1. Stop the Celery worker (find and kill the process)
pkill -f "celery -A celery_app worker"

# 2. Stop Flower
pkill -f "celery -A celery_app flower"

# 3. Stop the FastAPI server
pkill -f "uvicorn server:app"

# 4. Stop Ollama (frees LLM models from GPU)
pkill -f "ollama serve"

# 5. Force-release all GPU memory held by Python/PyTorch
python -c "
import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('PyTorch GPU cache cleared')
    used = torch.cuda.memory_allocated(0) / 1024**2
    print(f'Remaining allocated: {used:.0f} MB')
"

# 6. Nuclear option — kill ALL processes using the GPU
#    (only use if nvidia-smi still shows memory in use after steps above)
nvidia-smi | grep 'python\|celery\|ollama\|uvicorn' && \
  echo "Killing remaining GPU processes..." && \
  fuser -v /dev/nvidia* 2>/dev/null | xargs -r kill -9

# 7. Verify GPU is fully free
nvidia-smi
```

After running these commands, `nvidia-smi` should show `0MiB` memory usage.

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
# Check if Redis is running
redis-cli ping

# If not, start it
redis-server --daemonize yes --appendonly yes
```
