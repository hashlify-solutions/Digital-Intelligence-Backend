# Digital Intelligence Backend

## Make sure to install the following dependencies to setup the project

### 3rd party dependencies 
1. Ollama (After installing Llama run the below commands in separate terminal tabs to run LLMs locally via Ollama) 
    - ollama run llama3.1:8b
    - ollama run llava:latest
2. Docker CLI
3. Docker Desktop
4. Docker Compose
5. Python (3.12+)
6. MongoDB 
7. MongoDB Compass


## GPU Configuration

### Development Environment
For development on machines with limited VRAM or when debugging:
```bash
export DI_DEVICE=cpu
```

### Production Environment  
For production with dedicated GPUs and sufficient VRAM:
```bash
export DI_DEVICE=cuda
```

### Automatic Selection (Recommended)
Let the system automatically choose the best device based on available GPU memory:
```bash
export DI_DEVICE=auto
```

The system will automatically monitor GPU memory usage and fallback to CPU if needed.

### External Model Download Links
- Download the following files and place them in the models/dnn-face-detector directory (https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector):
    - deploy.prototxt (Download from: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
    - res10_300x300_ssd_iter_140000.caffemodel (Download from: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
- Download the following files and place them in the models/yolo-object-detector directory: 
    - yolo12x.pt: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt
    - yolo12l.pt: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt
    - yolo12m.pt: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt
    - yolo12s.pt: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt
    - yolo12n.pt: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt

### Troubleshooting: dlib / face_recognition Native Library Errors (macOS)

When running Celery workers (or any import that triggers `face_recognition` / `dlib`), you may encounter `ImportError` messages like:

```
ImportError: dlopen(.../_dlib_pybind11.cpython-3xx-darwin.so, ...): Library not loaded: /opt/homebrew/opt/libx11/lib/libX11.6.dylib
```

This happens because `dlib` is compiled against several native C libraries (X11, giflib, jpeg-turbo, etc.) that may not be present on a fresh macOS setup, or may have been removed by a `brew cleanup`, macOS update, or Homebrew auto-update.

**To fix this, install all required native dependencies at once:**

```bash
brew install libx11 libxext libsm libice libxrender libxft giflib jpeg-turbo
```

The full list of Homebrew libraries that `dlib` depends on:

| Library | Homebrew Formula | Provides |
|---|---|---|
| `libX11.6.dylib` | `libx11` | X Window System protocol client library |
| `libXext.6.dylib` | `libxext` | X11 common extensions library |
| `libgif.dylib` | `giflib` | GIF image reading/writing |
| `libpng16.16.dylib` | `libpng` | PNG image reading/writing (installed as dependency of `libxft`) |
| `libjpeg.8.dylib` | `jpeg-turbo` | JPEG image reading/writing |

> **Tip:** If you continue to see similar errors for other libraries, you can inspect all native dependencies of the dlib binary with:
> ```bash
> otool -L $(python -c "import _dlib_pybind11; print(_dlib_pybind11.__file__)")
> ```

> **Alternative:** Install XQuartz to get all X11 libraries bundled together:
> ```bash
> brew install --cask xquartz
> ```

---

### Project setup guide (For mac)
1. Clone the project
2. Setup a virtual environment with "mkvirtualenv", to install the virtual environment run the following command:
    - pip3 install virtualenvwrapper
3. Create a new virtual environment by running the following command:
    - mkvirtualenv di-backend
    - workon di-backend
4. Install the dependencies by running the following command inside the virtual environment:
    - pip3 install -r requirements.txt
5. Download the Nudenet model by running the following command inside the virtual environment:
    - pip3 install -U git+https://github.com/platelminto/NudeNet
6. Create the database indexes by running the script: 
    - python3 database_scripts/create_ufdr_indexes.py
7. Install FFmpeg by running the following command: 
    - brew install ffmpeg
8. Install wget utility if not already installed by running the following command:
    - brew install wget
9. Run the docker engine via docker desktop and run the following in the project's terminal to run redis and flower services:
    - docker-compose up -d
10. Configure GPU/CPU usage by setting the DI_DEVICE environment variable in the .env file:
    - For development (CPU): export DI_DEVICE=cpu
    - For production (GPU): export DI_DEVICE=cuda  
    - For automatic selection: export DI_DEVICE=auto (default)
11. Configure the platform by setting the PLATFORM environment variable in the .env file:
    - For linux: PLATFORM=linux
    - For mac: PLATFORM=mac
    - For windows: PLATFORM=windows
12. Open two terminal sessions and and run the following commands to run the project:
    - Terminal 1 (Command to run the celery workers: /.start_celery_workers.sh)
    - Terminal 2 (Command to run the uvicorn server: uvicorn server:app --reload)


### Project setup guide (For linux)
1. Clone the project
2. Setup a virtual environment with "mkvirtualenv", to install the virtual environment run the following command:
    - pip install virtualenvwrapper
3. Create a new virtual environment by running the following command:
    - mkvirtualenv di-backend
    - workon di-backend
4. Install the dependencies by running the following command inside the virtual environment:
    - pip install -r requirements.txt
5. Download the Nudenet model by running the following command inside the virtual environment:
    - pip install -U git+https://github.com/platelminto/NudeNet
6. Create the database indexes by running the script: 
    - python database_scripts/create_ufdr_indexes.py
7. Install FFmpeg by running the following command: 
    - sudo apt update && sudo apt install -y ffmpeg
8. Install wget utility if not already installed by running the following command:
    - sudo apt update && sudo apt install -y wget
9. Run the docker engine via docker desktop and run the following in the project's terminal to run redis and flower services:
    - docker-compose up -d
10. Configure GPU/CPU usage by setting the DI_DEVICE environment variable in the .env file:
    - For development (CPU): export DI_DEVICE=cpu
    - For production (GPU): export DI_DEVICE=cuda  
    - For automatic selection: export DI_DEVICE=auto (default)
11. Configure the platform by setting the PLATFORM environment variable in the .env file:
    - For linux: PLATFORM=linux
    - For mac: PLATFORM=mac
    - For windows: PLATFORM=windows
12. Open two terminal sessions and and run the following commands to run the project:
    - Terminal 1 (Command to run the celery workers: /.start_celery_workers.sh)
    - Terminal 2 (Command to run the uvicorn server: uvicorn server:app --reload)