# Celery Task Queue Setup Guide

This guide explains how to set up and run the Celery task queue for handling background processing in the Digital Intelligence Platform.

## Prerequisites

1. Docker and Docker Compose (for Redis and Flower)
2. Python environment with required packages installed

## Docker Installation

### Install Docker on Ubuntu/Debian

1. **Update package index:**
```bash
sudo apt update
```

2. **Install required packages:**
```bash
sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release
```

3. **Add Docker's official GPG key:**
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

4. **Add Docker repository:**
```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

5. **Install Docker Engine:**
```bash
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
```

6. **Add your user to docker group (to run without sudo):**
```bash
sudo usermod -aG docker $USER
```

7. **Log out and log back in, then test:**
```bash
docker --version
```

### Install Docker Compose

**Option 1: Using apt (Ubuntu/Debian):**
```bash
sudo apt install docker-compose
```

**Option 2: Using pip:**
```bash
pip install docker-compose
```

**Option 3: Download binary directly:**
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**Verify installation:**
```bash
docker-compose --version
```

### Install Docker Desktop (Alternative - GUI Application)

If you prefer a GUI application:

1. **Download Docker Desktop for Linux:**
   - Visit: https://docs.docker.com/desktop/install/linux-install/
   - Download the `.deb` package for Ubuntu

2. **Install the package:**
```bash
sudo dpkg -i docker-desktop-<version>-<arch>.deb
sudo apt-get install -f  # Fix dependencies if needed
```

3. **Start Docker Desktop:**
```bash
systemctl --user start docker-desktop
```

### Quick Installation (Recommended)

For Ubuntu/Debian systems, you can use this one-liner:
```bash
# Install Docker and Docker Compose
sudo apt update && sudo apt install -y docker.io docker-compose

# Add user to docker group and apply changes
sudo usermod -aG docker $USER && newgrp docker

# Test installation
docker --version && docker-compose --version
```

## Docker Installation for macOS

### Install Docker Desktop for Mac

**Option 1: Download from Docker Website (Recommended)**

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download Docker Desktop for Mac (Intel or Apple Silicon)

2. **Install Docker Desktop:**
   - Open the downloaded `.dmg` file
   - Drag Docker to Applications folder
   - Launch Docker from Applications

3. **Complete setup:**
   - Docker Desktop will start automatically
   - Follow the setup wizard
   - Sign in with Docker Hub account (optional)

4. **Verify installation:**
```bash
docker --version
docker-compose --version
```

**Option 2: Using Homebrew**

1. **Install Homebrew (if not already installed):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Docker Desktop:**
```bash
brew install --cask docker
```

3. **Start Docker Desktop:**
```bash
open /Applications/Docker.app
```

4. **Verify installation:**
```bash
docker --version
docker-compose --version
```

### Install Redis for macOS (Alternative to Docker)

**Using Homebrew:**
```bash
# Install Redis
brew install redis

# Start Redis service
brew services start redis

# Test Redis
redis-cli ping
# Should return: PONG
```

### macOS-Specific Configuration

1. **Allow Docker to access required directories:**
   - Go to System Preferences → Security & Privacy → Privacy → Full Disk Access
   - Add Docker.app if prompted

2. **Configure Docker resources:**
   - Open Docker Desktop
   - Go to Settings → Resources
   - Adjust CPU and Memory limits based on your Mac's specs
   - Recommended: 4 CPUs, 8GB RAM for AI processing

3. **Enable file sharing:**
   - Go to Docker Desktop → Settings → Resources → File Sharing
   - Add your project directory if not already included

## Complete Setup Guide for macOS

### Step 1: Install Prerequisites

1. **Install Homebrew (if not already installed):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python 3.9+ (if not already installed):**
```bash
brew install python@3.11
```

3. **Install Docker Desktop:**
```bash
brew install --cask docker
```

4. **Start Docker Desktop:**
```bash
open /Applications/Docker.app
```
Wait for the whale icon to appear in your menu bar.

### Step 2: Setup Python Environment

1. **Create virtual environment:**
```bash
cd /path/to/your/project
python3 -m venv venv
```

2. **Activate virtual environment:**
```bash
source venv/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Setup Environment Variables

1. **Create .env file:**
```bash
touch .env
```

2. **Add the following to .env file:**
```env
# MongoDB Configuration (use your actual credentials)
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=DigitalIntelligence

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Add your other environment variables
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Step 4: Start Services

1. **Start Redis using Docker:**
```bash
docker-compose up -d redis
```

2. **Verify Redis is running:**
```bash
docker ps
# Should show di_redis container running

docker exec -it di_redis redis-cli ping
# Should return: PONG
```

3. **Start Celery Worker (in a new terminal):**
```bash
# Activate virtual environment first
source venv/bin/activate

# Start worker
./start_celery_worker.sh
```

4. **Start Flower for monitoring (optional, in another terminal):**
```bash
docker-compose up -d flower
```
Access at: http://localhost:5555

5. **Start FastAPI server (in another terminal):**
```bash
# Activate virtual environment first
source venv/bin/activate

# Start server
uvicorn server:app --reload
```

### Step 5: Test the Setup

1. **Test Redis connection:**
```bash
redis-cli -h localhost -p 6379 ping
```

2. **Test Celery worker:**
```bash
celery -A celery_app status
```

3. **Test FastAPI server:**
Open browser and go to: http://localhost:8000/docs

### macOS Terminal Commands Summary

```bash
# Terminal 1: Redis (using Docker)
docker-compose up -d redis

# Terminal 2: Celery Worker
source venv/bin/activate
./start_celery_worker.sh

# Terminal 3: FastAPI Server
source venv/bin/activate
uvicorn server:app --reload

# Terminal 4: Flower (optional)
docker-compose up -d flower
```

### Alternative Setup (Redis without Docker)

If you prefer to run Redis natively on macOS:

1. **Install Redis with Homebrew:**
```bash
brew install redis
```

2. **Start Redis service:**
```bash
brew services start redis
```

3. **Test Redis:**
```bash
redis-cli ping
```

4. **Update .env file:**
```env
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### macOS Performance Tips

1. **For Apple Silicon Macs (M1/M2):**
   - Use native Python installations when possible
   - Specify platform for Docker if needed: `--platform linux/amd64`

2. **Memory allocation:**
   - Allocate at least 8GB RAM to Docker Desktop for AI models
   - Adjust Celery worker concurrency based on CPU cores

3. **File watching issues:**
   - Use `--reload` flag for development
   - Consider using `watchdog` for better file watching on macOS

## Python Package Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Start Redis using Docker Compose:
```bash
docker-compose up -d redis
```

Or install Redis locally:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
```

## Configuration

1. Create a `.env` file with the following variables:
```env
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Running the Application

### 1. Start Redis (if not using Docker)
```bash
redis-server
```

### 2. Start Celery Worker
In a new terminal window:
```bash
./start_celery_worker.sh
```

Or manually:
```bash
celery -A celery_app worker --loglevel=info --concurrency=4
```

### 3. Start Celery Flower (Optional - for monitoring)
```bash
docker-compose up -d flower
```
Then access Flower UI at: http://localhost:5555

### 4. Start FastAPI Server
```bash
uvicorn server:app --reload
```

## How It Works

1. When `/upload-data` endpoint is called, it creates a Celery task instead of using FastAPI background tasks
2. The task is queued in Redis and processed by Celery workers
3. The API immediately returns a response with `task_id`
4. You can check task status using `/task-status/{task_id}` endpoint

## Task Flow

1. **CSV Ingestion** (`ingest_csv_task`)
   - Reads CSV file
   - Inserts data into MongoDB
   - Updates case status

2. **RAG Processing** (`ingest_rag_task`) - Only if `is_rag=true`
   - Generates embeddings for each message
   - Stores in Qdrant vector database

3. **Document Analysis** (`analyze_documents_task`)
   - Processes each document through AI models
   - Performs topic classification, sentiment analysis, etc.
   - Updates MongoDB with analysis results

## Monitoring

### Using Flower
Access http://localhost:5555 to:
- View active tasks
- Monitor task queues
- Check worker status
- View task history

### Using Celery CLI
```bash
# Check worker status
celery -A celery_app status

# Inspect active tasks
celery -A celery_app inspect active

# Inspect scheduled tasks
celery -A celery_app inspect scheduled
```

## Troubleshooting

### Docker Issues

1. **"docker-compose: command not found"**
   ```bash
   # Install docker-compose
   sudo apt install docker-compose
   # OR using pip
   pip install docker-compose
   ```

2. **"permission denied while trying to connect to Docker daemon"**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Apply changes (logout/login or use newgrp)
   newgrp docker
   ```

3. **Docker service not running**
   ```bash
   # Start Docker service
   sudo systemctl start docker
   # Enable auto-start
   sudo systemctl enable docker
   ```

4. **Port 6379 already in use**
   ```bash
   # Check what's using the port
   sudo lsof -i :6379
   # Kill the process or change Redis port in docker-compose.yml
   ```

### Celery Issues

1. **Worker not picking up tasks**
   - Check Redis connection
   - Verify worker is running
   - Check queue names match

2. **Tasks failing**
   - Check worker logs
   - Verify MongoDB connection in workers
   - Check model files are accessible

3. **Memory issues**
   - Adjust worker concurrency
   - Set `--max-tasks-per-child` to prevent memory leaks

### Alternative: Running Redis without Docker

If you prefer not to use Docker, install Redis directly:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Test Redis:**
```bash
redis-cli ping
# Should return: PONG
```

### macOS-Specific Issues

1. **"Cannot connect to the Docker daemon" on macOS**
   ```bash
   # Make sure Docker Desktop is running
   open /Applications/Docker.app
   # Wait for Docker to start (whale icon in menu bar)
   ```

2. **Permission issues with volumes on macOS**
   - Go to Docker Desktop → Settings → Resources → File Sharing
   - Add your project directory: `/Users/yourusername/path/to/project`
   - Restart Docker Desktop

3. **Apple Silicon (M1/M2) compatibility**
   ```bash
   # If you get platform warnings, specify platform
   docker-compose up -d --platform linux/amd64 redis
   ```

4. **Homebrew installation issues**
   ```bash
   # If brew command not found, add to PATH
   echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

5. **Python virtual environment setup on macOS**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   
   # Install requirements
   pip install -r requirements.txt
   ```

## Production Considerations

1. Use a process manager (supervisor, systemd) to manage Celery workers
2. Configure multiple workers for different queues
3. Set up proper logging and monitoring
4. Use Redis persistence for durability
5. Configure task retries and error handling