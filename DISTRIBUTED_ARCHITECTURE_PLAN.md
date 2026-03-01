# Digital Intelligence: Distributed Architecture Transformation Plan

## Executive Summary

This document outlines the transformation of the Digital Intelligence (DI) system from a monolithic FastAPI application to a distributed, scalable architecture that can efficiently utilize multiple host machines with GPUs for processing large-scale digital forensic data.

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Production Environment Constraints](#production-environment-constraints)
3. [Proposed Distributed Architecture](#proposed-distributed-architecture)
4. [Component Breakdown](#component-breakdown)
5. [GPU Resource Management](#gpu-resource-management)
6. [Implementation Plan](#implementation-plan)
7. [Deployment Strategy](#deployment-strategy)
8. [Monitoring & Scaling](#monitoring-scaling)
9. [Migration Roadmap](#migration-roadmap)

## 1. Current Architecture Analysis <a id="current-architecture-analysis"></a>

### Current State
- **Monolithic FastAPI application** running on a single machine
- **Celery** for asynchronous task processing
- **MongoDB** for data storage
- **Qdrant** for vector database
- **Single GPU** utilization with dynamic device selection
- Processing of large UFDR files (200-300 GB)

### Key Challenges
- Single point of failure
- Limited scalability
- GPU underutilization when multiple machines available
- Memory constraints for large file processing
- No load balancing for concurrent requests

## 2. Production Environment Constraints <a id="production-environment-constraints"></a>

### Average Specifications
- **Machines**: 4 hosts (scalable from 1 to N)
- **GPUs**: 1x NVIDIA RTX A4000 16GB per machine
- **OS**: Windows (with Linux support)
- **File Sizes**: 200-300 GB UFDR files
- **Variability**: System must adapt to 1-N machines and GPUs

### Requirements
- Dynamic scaling based on available resources
- Cross-platform compatibility (Windows/Linux)
- Efficient GPU utilization across machines
- High availability and fault tolerance
- Support for concurrent large file processing

## 3. Proposed Distributed Architecture <a id="proposed-distributed-architecture"></a>

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX - Self-hosted)              │
│                       Distributed across all nodes                   │
└────────────────────┬───────────────────────┬───────────────────────┘
                     │                       │
     ┌───────────────▼──────────┐  ┌────────▼───────────────┐
     │  Hybrid Node 1 (Host 1)  │  │  Hybrid Node 2 (Host 2) │
     │  ┌────────────────────┐  │  │  ┌────────────────────┐ │
     │  │ FastAPI Gateway    │  │  │  │ FastAPI Gateway    │ │
     │  │ - Request routing  │  │  │  │ - Request routing  │ │
     │  │ - File handling    │  │  │  │ - File handling    │ │
     │  │ - Light ML tasks   │  │  │  │ - Light ML tasks   │ │
     │  └────────────────────┘  │  │  └────────────────────┘ │
     │  ┌────────────────────┐  │  │  ┌────────────────────┐ │
     │  │ Celery GPU Worker  │  │  │  │ Celery GPU Worker  │ │
     │  │ - Face Detection   │  │  │  │ - Object Detection │ │
     │  │ - Transcription    │  │  │  │ - NSFW Detection   │ │
     │  │ - Embeddings       │  │  │  │ - Llava Vision     │ │
     │  └────────────────────┘  │  │  └────────────────────┘ │
     │   GPU: RTX A4000 16GB    │  │   GPU: RTX A4000 16GB   │
     │   Local Storage: 2TB     │  │   Local Storage: 2TB    │
     └──────────────────────────┘  └──────────────────────────┘
                    │                        │
     ┌──────────────▼────────────────────────▼──────────┐
     │      Distributed File System (GlusterFS/NFS)      │
     │         Replicated across all nodes               │
     └───────────────────────────────────────────────────┘
                    │                        │
     ┌──────────────▼─────────┐   ┌─────────▼──────────────┐
     │  MongoDB (Self-hosted)  │   │  Qdrant (Self-hosted)  │
     │   Replica Set Mode      │   │  Distributed Mode      │
     │  Node 1: Primary        │   │  Node 1: Leader        │
     │  Node 2-4: Secondary    │   │  Node 2-4: Followers   │
     └─────────────────────────┘   └────────────────────────┘
```

### Key Architecture Changes:

1. **Hybrid Nodes**: Each physical host runs both API Gateway and GPU Worker services
2. **GPU Utilization**: All nodes can process both API requests and GPU-intensive tasks
3. **Local Storage**: Files stored locally with distributed file system for sharing
4. **Self-hosted Everything**: All components run on-premises without cloud dependencies
5. **Load Distribution**: Tasks distributed based on GPU availability across all nodes

## 4. Component Breakdown <a id="component-breakdown"></a>

### 4.1 Hybrid Node Architecture
**Purpose**: Maximize GPU utilization by running both API and Worker services on each node

**Components per Node**:
- **FastAPI instance** with GPU access for light ML tasks
- **Celery Worker** for heavy GPU tasks
- **Local file storage** with distributed sync
- **Resource manager** for GPU allocation

**Implementation**:
```python
# hybrid_node/main.py
from fastapi import FastAPI, UploadFile
from celery import Celery
import aiofiles
import uuid
import torch
from pathlib import Path
from gpu_resource_manager import GPUResourceManager

app = FastAPI()
celery_app = Celery('tasks', broker='redis://localhost:6379')  # Local Redis
gpu_manager = GPUResourceManager()

# Local storage configuration
LOCAL_STORAGE_ROOT = Path("/data/di-storage")
SHARED_STORAGE_ROOT = Path("/mnt/glusterfs/di-shared")

@app.post("/upload-ufdr-data")
async def upload_ufdr_data(file: UploadFile, case_id: str, ...):
    # Check if this node should handle the upload based on load
    if gpu_manager.can_handle_upload():
        # Stream file to local storage first
        file_id = str(uuid.uuid4())
        local_path = LOCAL_STORAGE_ROOT / case_id / file_id / file.filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(local_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
        
        # Sync to distributed storage in background
        await sync_to_distributed_storage(local_path, SHARED_STORAGE_ROOT / case_id / file_id)
        
        # Decide whether to process locally or distribute
        if gpu_manager.has_capacity_for_task('process_ufdr_upload'):
            # Process on this node
            task = celery_app.send_task(
                'process_ufdr_upload',
                kwargs={
                    'local_path': str(local_path),
                    'case_id': case_id,
                    'node_id': gpu_manager.node_id,
                    # ... other parameters
                },
                queue=f'gpu_tasks_node_{gpu_manager.node_id}'
            )
        else:
            # Distribute to least loaded node
            target_node = gpu_manager.get_least_loaded_node()
            task = celery_app.send_task(
                'process_ufdr_upload',
                kwargs={
                    'shared_path': str(SHARED_STORAGE_ROOT / case_id / file_id / file.filename),
                    'case_id': case_id,
                    'target_node': target_node,
                    # ... other parameters
                },
                queue=f'gpu_tasks_node_{target_node}'
            )
    else:
        # Redirect to least loaded node
        return {"redirect": gpu_manager.get_least_loaded_node_url()}
    
    return {"task_id": task.id, "status": "processing", "node": gpu_manager.node_id}

@app.post("/process-light-ml")
async def process_light_ml(text: str):
    """Handle light ML tasks directly in API layer"""
    if gpu_manager.acquire_gpu_slot(estimated_memory_gb=2.0, timeout=5):
        try:
            # Quick inference tasks that don't need heavy models
            result = await quick_sentiment_analysis(text)
            return {"result": result}
        finally:
            gpu_manager.release_gpu_slot()
    else:
        # Fallback to CPU or queue for later
        return {"status": "queued", "reason": "gpu_busy"}
```

### 4.2 Task Distribution Layer
**Purpose**: Intelligent task routing based on GPU availability

**Components**:
- **Redis/RabbitMQ** for message queuing
- **Celery** with custom routers
- **GPU resource monitor**

**Task Routing Strategy**:
```python
# celery_config.py
from kombu import Queue

class GPUTaskRouter:
    def route_for_task(self, task, args=None, kwargs=None):
        gpu_intensive_tasks = [
            'detect_faces_task',
            'detect_objects_task',
            'analyze_audio_task',
            'generate_image_description_llava_task',
            'detect_nsfw_images_task'
        ]
        
        cpu_tasks = [
            'extract_ufdr_file_task',
            'save_extracted_ufdr_data_task'
        ]
        
        if task in gpu_intensive_tasks:
            return {
                'queue': 'gpu_tasks',
                'routing_key': 'gpu.tasks',
                'priority': kwargs.get('priority', 5)
            }
        elif task in cpu_tasks:
            return {
                'queue': 'cpu_tasks',
                'routing_key': 'cpu.tasks'
            }
        
        return None

CELERY_QUEUES = (
    Queue('gpu_tasks', routing_key='gpu.tasks'),
    Queue('cpu_tasks', routing_key='cpu.tasks'),
    Queue('default', routing_key='default.tasks'),
)
```

### 4.3 GPU Resource Management
**Purpose**: Efficiently manage GPU resources across hybrid nodes

**Components**:
- **GPU Resource Manager** for allocation tracking
- **Model Registry** with node-aware caching
- **Dynamic task routing** based on GPU availability

**GPU Resource Manager Implementation**:
```python
# gpu_resource_manager.py
import torch
import psutil
import redis
import json
from threading import Lock
from datetime import datetime
import pynvml

class GPUResourceManager:
    def __init__(self, node_id=None):
        self.node_id = node_id or self._get_node_id()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.gpu_lock = Lock()
        self.local_gpu_slots = {}
        
        # Initialize NVML for GPU monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU per node
        
        # Register this node
        self._register_node()
        
    def _get_node_id(self):
        """Generate unique node ID based on hostname and MAC"""
        import socket
        import hashlib
        hostname = socket.gethostname()
        return hashlib.md5(hostname.encode()).hexdigest()[:8]
    
    def _register_node(self):
        """Register this node in the cluster"""
        node_info = {
            'node_id': self.node_id,
            'hostname': socket.gethostname(),
            'gpu_name': pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8'),
            'total_memory_gb': pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'total_ram_gb': psutil.virtual_memory().total / (1024**3),
            'last_heartbeat': datetime.now().isoformat()
        }
        self.redis_client.hset('di:nodes', self.node_id, json.dumps(node_info))
        self.redis_client.expire('di:nodes', 3600)  # Expire after 1 hour if no heartbeat
    
    def get_gpu_status(self):
        """Get current GPU utilization and memory"""
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        
        return {
            'node_id': self.node_id,
            'gpu_utilization': utilization.gpu,
            'memory_used_gb': memory_info.used / (1024**3),
            'memory_free_gb': memory_info.free / (1024**3),
            'memory_total_gb': memory_info.total / (1024**3),
            'active_tasks': len(self.local_gpu_slots),
            'timestamp': datetime.now().isoformat()
        }
    
    def can_handle_task(self, task_type, estimated_memory_gb):
        """Check if this node can handle a specific task"""
        gpu_status = self.get_gpu_status()
        
        # Define task priorities and memory requirements
        task_priorities = {
            'face_detection': {'priority': 1, 'memory_gb': 2.0},
            'object_detection': {'priority': 1, 'memory_gb': 3.0},
            'transcription': {'priority': 2, 'memory_gb': 4.0},
            'llava_vision': {'priority': 3, 'memory_gb': 8.0},
            'embedding_generation': {'priority': 1, 'memory_gb': 1.5}
        }
        
        task_info = task_priorities.get(task_type, {'priority': 5, 'memory_gb': estimated_memory_gb})
        required_memory = task_info['memory_gb']
        
        # Check if we have enough free memory (with 20% buffer)
        if gpu_status['memory_free_gb'] < required_memory * 1.2:
            return False
        
        # Check GPU utilization (prefer < 80%)
        if gpu_status['gpu_utilization'] > 80:
            return False
        
        return True
    
    def acquire_gpu_slot(self, task_id, task_type, estimated_memory_gb, timeout=30):
        """Acquire a GPU slot for a task"""
        with self.gpu_lock:
            if self.can_handle_task(task_type, estimated_memory_gb):
                self.local_gpu_slots[task_id] = {
                    'task_type': task_type,
                    'memory_gb': estimated_memory_gb,
                    'start_time': datetime.now().isoformat(),
                    'pid': os.getpid()
                }
                
                # Update cluster state
                self._update_cluster_state()
                return True
        return False
    
    def release_gpu_slot(self, task_id):
        """Release a GPU slot after task completion"""
        with self.gpu_lock:
            if task_id in self.local_gpu_slots:
                del self.local_gpu_slots[task_id]
                self._update_cluster_state()
                
                # Force GPU memory cleanup
                torch.cuda.empty_cache()
    
    def get_least_loaded_node(self):
        """Find the least loaded node in the cluster"""
        nodes = self.redis_client.hgetall('di:nodes')
        least_loaded = None
        min_load = float('inf')
        
        for node_id, node_data in nodes.items():
            node_info = json.loads(node_data)
            
            # Get node GPU status
            gpu_status_key = f'di:gpu_status:{node_id}'
            gpu_status_data = self.redis_client.get(gpu_status_key)
            
            if gpu_status_data:
                gpu_status = json.loads(gpu_status_data)
                # Calculate load score (lower is better)
                load_score = (
                    gpu_status['gpu_utilization'] * 0.4 +
                    (gpu_status['memory_used_gb'] / gpu_status['memory_total_gb']) * 100 * 0.4 +
                    gpu_status['active_tasks'] * 10 * 0.2
                )
                
                if load_score < min_load:
                    min_load = load_score
                    least_loaded = node_id
        
        return least_loaded or self.node_id
    
    def _update_cluster_state(self):
        """Update this node's state in the cluster"""
        gpu_status = self.get_gpu_status()
        self.redis_client.setex(
            f'di:gpu_status:{self.node_id}',
            60,  # Expire after 60 seconds
            json.dumps(gpu_status)
        )

# Celery Worker with GPU Management
from celery import Celery, Task
from model_registry import ModelRegistry

class GPUTask(Task):
    """Base task class with GPU resource management"""
    _gpu_manager = None
    _model_registry = None
    
    @property
    def gpu_manager(self):
        if self._gpu_manager is None:
            self._gpu_manager = GPUResourceManager()
        return self._gpu_manager
    
    @property
    def model_registry(self):
        if self._model_registry is None:
            self._model_registry = ModelRegistry(
                device='cuda:0',
                cache_dir='/data/model_cache',
                node_id=self.gpu_manager.node_id
            )
        return self._model_registry

app = Celery('di_tasks')
app.config_from_object('celeryconfig')

@app.task(base=GPUTask, bind=True)
def detect_faces_task(self, file_path, case_id, **kwargs):
    """Face detection task with GPU resource management"""
    task_id = self.request.id
    
    # Acquire GPU slot
    if not self.gpu_manager.acquire_gpu_slot(task_id, 'face_detection', 2.0):
        # Retry on another node
        target_node = self.gpu_manager.get_least_loaded_node()
        if target_node != self.gpu_manager.node_id:
            return self.retry(queue=f'gpu_tasks_node_{target_node}', countdown=5)
        else:
            # All nodes busy, retry later
            return self.retry(countdown=30)
    
    try:
        # Load model from registry
        face_model = self.model_registry.get_or_load(
            'face_detector',
            '/data/models/dnn-face-detector',
            estimated_size_gb=2.0
        )
        
        # Check if file is local or needs to be fetched
        if not Path(file_path).exists():
            # Fetch from distributed storage
            file_path = fetch_from_distributed_storage(file_path)
        
        # Process with GPU
        with torch.cuda.device('cuda:0'):
            results = face_model.process(file_path)
        
        # Save results locally and sync
        results_path = save_and_sync_results(results, case_id)
        
        return {
            'status': 'completed',
            'results_path': results_path,
            'processed_by': self.gpu_manager.node_id
        }
        
    finally:
        # Always release GPU slot
        self.gpu_manager.release_gpu_slot(task_id)
```

### 4.4 Model Registry Service
**Purpose**: Centralized model management and caching

```python
# model_registry.py
import torch
from threading import Lock
from collections import OrderedDict

class ModelRegistry:
    def __init__(self, device='cuda:0', max_models=3):
        self.device = device
        self.max_models = max_models
        self.models = OrderedDict()
        self.locks = {}
        
    def get_or_load(self, model_name, model_path, estimated_size_gb):
        if model_name in self.models:
            # Move to end (LRU)
            self.models.move_to_end(model_name)
            return self.models[model_name]
        
        # Check if we need to evict models
        if len(self.models) >= self.max_models:
            self._evict_lru_model()
        
        # Load model
        with self._get_lock(model_name):
            if model_name not in self.models:
                model = self._load_model(model_name, model_path)
                self.models[model_name] = model
        
        return self.models[model_name]
    
    def _evict_lru_model(self):
        """Evict least recently used model"""
        if self.models:
            evicted = self.models.popitem(last=False)
            del evicted[1]  # Delete model
            torch.cuda.empty_cache()
```

### 4.5 Distributed Local Storage
**Purpose**: High-performance local storage with distributed replication

**Architecture**:
1. **Primary**: Local NVMe/SSD storage on each node for fast I/O
2. **Replication**: GlusterFS for distributed file sharing
3. **Caching**: Redis for metadata and small file caching
4. **Backup**: Periodic snapshots to dedicated storage node

**Implementation**:
```python
# distributed_storage.py
import shutil
import asyncio
import aiofiles
from pathlib import Path
import hashlib
import json
from datetime import datetime
import redis

class DistributedLocalStorage:
    def __init__(self, node_id):
        self.node_id = node_id
        self.local_root = Path(f"/data/local/{node_id}")
        self.shared_root = Path("/mnt/glusterfs/di-shared")
        self.cache_root = Path("/data/cache")
        self.redis_client = redis.Redis(host='localhost', port=6379)
        
        # Create directories
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
    async def store_file(self, file_data, case_id, file_name):
        """Store file with intelligent placement"""
        file_hash = hashlib.sha256(file_name.encode()).hexdigest()[:16]
        
        # Determine storage strategy based on file size
        file_size = len(file_data) if isinstance(file_data, bytes) else file_data.stat().st_size
        
        if file_size < 100 * 1024 * 1024:  # < 100MB - cache it
            return await self._store_small_file(file_data, case_id, file_name, file_hash)
        else:
            return await self._store_large_file(file_data, case_id, file_name, file_hash)
    
    async def _store_small_file(self, file_data, case_id, file_name, file_hash):
        """Store small files in cache and replicate immediately"""
        cache_path = self.cache_root / case_id / file_hash / file_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to cache
        async with aiofiles.open(cache_path, 'wb') as f:
            if isinstance(file_data, bytes):
                await f.write(file_data)
            else:
                await f.write(await file_data.read())
        
        # Replicate to shared storage immediately
        shared_path = self.shared_root / case_id / file_hash / file_name
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.create_subprocess_exec(
            'cp', str(cache_path), str(shared_path)
        )
        
        # Store metadata in Redis
        metadata = {
            'case_id': case_id,
            'file_name': file_name,
            'file_hash': file_hash,
            'size': cache_path.stat().st_size,
            'node_id': self.node_id,
            'local_path': str(cache_path),
            'shared_path': str(shared_path),
            'created_at': datetime.now().isoformat()
        }
        self.redis_client.hset(
            f'di:files:{case_id}',
            file_hash,
            json.dumps(metadata)
        )
        
        return str(cache_path)
    
    async def _store_large_file(self, file_path, case_id, file_name, file_hash):
        """Store large files locally first, replicate async"""
        local_path = self.local_root / case_id / file_hash / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Stream large file to local storage
        if isinstance(file_path, Path):
            # Move file if it's already on disk
            shutil.move(str(file_path), str(local_path))
        else:
            # Stream from upload
            async with aiofiles.open(local_path, 'wb') as f:
                while chunk := await file_path.read(10 * 1024 * 1024):  # 10MB chunks
                    await f.write(chunk)
        
        # Schedule async replication
        await self._schedule_replication(local_path, case_id, file_hash, file_name)
        
        return str(local_path)
    
    async def _schedule_replication(self, local_path, case_id, file_hash, file_name):
        """Schedule background replication to GlusterFS"""
        replication_task = {
            'local_path': str(local_path),
            'shared_path': str(self.shared_root / case_id / file_hash / file_name),
            'priority': 'low' if local_path.stat().st_size > 1024**3 else 'normal',
            'scheduled_at': datetime.now().isoformat()
        }
        
        # Add to replication queue
        self.redis_client.lpush(
            'di:replication_queue',
            json.dumps(replication_task)
        )
    
    async def get_file(self, case_id, file_hash):
        """Retrieve file with cache awareness"""
        # Check metadata
        metadata = self.redis_client.hget(f'di:files:{case_id}', file_hash)
        if not metadata:
            return None
        
        file_info = json.loads(metadata)
        
        # Try local paths first
        for path_key in ['local_path', 'cache_path']:
            if path_key in file_info:
                path = Path(file_info[path_key])
                if path.exists():
                    return str(path)
        
        # Fallback to shared storage
        shared_path = Path(file_info['shared_path'])
        if shared_path.exists():
            # Cache it locally for next time
            if file_info['size'] < 100 * 1024 * 1024:
                cache_path = self.cache_root / case_id / file_hash / file_info['file_name']
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(shared_path), str(cache_path))
                return str(cache_path)
            return str(shared_path)
        
        return None

# Background replication worker
async def replication_worker(storage_client):
    """Background worker for file replication"""
    redis_client = redis.Redis(host='localhost', port=6379)
    
    while True:
        # Get replication task
        task_data = redis_client.brpop('di:replication_queue', timeout=10)
        if not task_data:
            await asyncio.sleep(1)
            continue
        
        task = json.loads(task_data[1])
        local_path = Path(task['local_path'])
        shared_path = Path(task['shared_path'])
        
        if local_path.exists():
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rsync for efficient replication
            proc = await asyncio.create_subprocess_exec(
                'rsync', '-av', '--inplace',
                str(local_path), str(shared_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                print(f"Replicated {local_path} to {shared_path}")
            else:
                print(f"Replication failed: {stderr.decode()}")
                # Re-queue for retry
                redis_client.lpush('di:replication_queue', json.dumps(task))
```

## 5. GPU Resource Management <a id="gpu-resource-management"></a>

### 5.1 Dynamic GPU Discovery
```python
# gpu_discovery.py
import pynvml
import platform

class GPUDiscovery:
    def __init__(self):
        if platform.system() != "Darwin":  # Not macOS
            pynvml.nvmlInit()
    
    def discover_gpus(self):
        gpus = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpus.append({
                    'id': i,
                    'name': name,
                    'total_memory_gb': memory.total / (1024**3),
                    'free_memory_gb': memory.free / (1024**3),
                    'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                })
        except:
            pass
        
        return gpus
```

### 5.2 GPU Task Scheduler
```python
# gpu_scheduler.py
from celery import Celery
import redis

class GPUTaskScheduler:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)
        self.celery = Celery('tasks', broker=redis_url)
        
    def schedule_task(self, task_name, task_args, memory_required_gb):
        # Get available GPUs from all workers
        available_gpus = self._get_available_gpus()
        
        # Find best GPU for task
        best_gpu = None
        for gpu in available_gpus:
            if gpu['free_memory_gb'] > memory_required_gb * 1.2:
                if not best_gpu or gpu['utilization'] < best_gpu['utilization']:
                    best_gpu = gpu
        
        if best_gpu:
            # Route to specific worker
            return self.celery.send_task(
                task_name,
                args=task_args,
                queue=f"gpu_{best_gpu['worker_id']}_{best_gpu['gpu_id']}"
            )
        else:
            # Queue for later when GPU available
            return self.celery.send_task(
                task_name,
                args=task_args,
                queue='gpu_queue_pending'
            )
```

## 6. Implementation Plan <a id="implementation-plan"></a>

### Phase 1: Infrastructure Setup (2-3 weeks)
1. **Set up self-hosted infrastructure**
   - Install Docker on all nodes
   - Configure Docker Swarm cluster (simpler than K8s for on-prem)
   - Set up local container registry
   
2. **Configure distributed storage**
   - Install and configure GlusterFS across all nodes
   - Set up local NVMe/SSD partitions
   - Configure Redis cluster for caching
   
3. **Database deployment**
   - Deploy MongoDB replica set (1 primary, 3 secondaries)
   - Deploy Qdrant in distributed mode
   - Configure local backups

4. **Monitoring stack**
   - Deploy Prometheus on each node
   - Central Grafana instance
   - Configure GPU exporters

### Phase 2: Service Decomposition (3-4 weeks)
1. **Extract API Gateway**
   - Separate FastAPI routing logic
   - Implement file streaming service
   
2. **Create GPU Worker Service**
   - Containerize ML models
   - Implement model registry
   
3. **Task Queue Setup**
   - Configure Celery with Redis cluster
   - Implement custom task routing

### Phase 3: Integration & Testing (2-3 weeks)
1. **Service integration**
   - Connect all components
   - Test end-to-end workflows
   
2. **Performance testing**
   - Load testing with large files
   - GPU utilization monitoring
   
3. **Failover testing**
   - Test node failures
   - Verify task redistribution

### Phase 4: Migration & Deployment (1-2 weeks)
1. **Data migration**
   - Migrate existing MongoDB data
   - Transfer vector embeddings
   
2. **Gradual rollout**
   - Deploy to staging environment
   - Phased production deployment

## 7. Deployment Strategy <a id="deployment-strategy"></a>

### 7.1 Container Orchestration

#### Docker Swarm Configuration for Hybrid Nodes
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Hybrid service running on each node
  di_hybrid_node:
    image: di-hybrid-node:latest
    deploy:
      mode: global  # One instance per node
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NODE_ID={{.Node.Hostname}}
      - REDIS_URL=redis://localhost:6379
      - MONGODB_URL=mongodb://localhost:27017/di?replicaSet=di-rs
      - QDRANT_URL=http://localhost:6333
      - GLUSTER_MOUNT=/mnt/glusterfs
      - LOCAL_STORAGE=/data/local
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - /data/local:/data/local
      - /data/models:/data/models
      - /data/cache:/data/cache
      - /mnt/glusterfs:/mnt/glusterfs
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - target: 8000
        published: 8000
        mode: host  # Each node exposes its own port
    networks:
      - di_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MongoDB replica set (deploy on specific nodes)
  mongodb_primary:
    image: mongo:7
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.mongo == primary
    command: mongod --replSet di-rs --bind_ip_all
    volumes:
      - mongo_primary_data:/data/db
      - mongo_primary_config:/data/configdb
    networks:
      - di_network

  mongodb_secondary:
    image: mongo:7
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.labels.mongo == secondary
    command: mongod --replSet di-rs --bind_ip_all
    volumes:
      - mongo_secondary_data:/data/db
      - mongo_secondary_config:/data/configdb
    networks:
      - di_network

  # Qdrant distributed deployment
  qdrant:
    image: qdrant/qdrant:latest
    deploy:
      mode: global
      placement:
        constraints:
          - node.labels.qdrant == true
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - target: 6333
        published: 6333
        mode: host
    networks:
      - di_network

  # Redis on each node for local caching
  redis:
    image: redis:7-alpine
    deploy:
      mode: global
    command: redis-server --appendonly yes --maxmemory 8gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - target: 6379
        published: 6379
        mode: host
    networks:
      - di_network

  # NGINX load balancer (on manager nodes)
  nginx:
    image: nginx:alpine
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager
      update_config:
        parallelism: 1
        delay: 10s
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - di_network
    depends_on:
      - di_hybrid_node

  # Prometheus on each node
  prometheus:
    image: prom/prometheus:latest
    deploy:
      mode: global
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - target: 9090
        published: 9090
        mode: host
    networks:
      - di_network

  # Grafana (single instance)
  grafana:
    image: grafana/grafana:latest
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    ports:
      - "3000:3000"
    networks:
      - di_network

  # NVIDIA GPU exporter for Prometheus
  nvidia_exporter:
    image: nvidia/dcgm-exporter:latest
    deploy:
      mode: global
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DCGM_EXPORTER_LISTEN=:9400
      - DCGM_EXPORTER_KUBERNETES=false
    ports:
      - target: 9400
        published: 9400
        mode: host
    networks:
      - di_network

volumes:
  mongo_primary_data:
    driver: local
  mongo_primary_config:
    driver: local
  mongo_secondary_data:
    driver: local
  mongo_secondary_config:
    driver: local
  qdrant_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  di_network:
    driver: overlay
    attachable: true
```

#### Kubernetes Configuration
```yaml
# gpu-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-worker
spec:
  replicas: 4
  selector:
    matchLabels:
      app: gpu-worker
  template:
    metadata:
      labels:
        app: gpu-worker
    spec:
      nodeSelector:
        gpu: "true"
      containers:
      - name: worker
        image: di-gpu-worker:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: WORKER_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: GPU_MEMORY_FRACTION
          value: "0.9"
        volumeMounts:
        - name: model-cache
          mountPath: /models
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: temp-storage
        emptyDir:
          sizeLimit: 100Gi
```

### 7.2 Service Discovery & GPU-Aware Load Balancing

```nginx
# nginx.conf
upstream di_nodes {
    # GPU-aware load balancing
    least_conn;
    
    # List all hybrid nodes
    server node1.di.local:8000 weight=1 max_fails=3 fail_timeout=30s;
    server node2.di.local:8000 weight=1 max_fails=3 fail_timeout=30s;
    server node3.di.local:8000 weight=1 max_fails=3 fail_timeout=30s;
    server node4.di.local:8000 weight=1 max_fails=3 fail_timeout=30s;
    
    keepalive 32;
}

# Health check configuration
upstream di_health {
    server node1.di.local:8000;
    server node2.di.local:8000;
    server node3.di.local:8000;
    server node4.di.local:8000;
    
    # Active health checks
    health_check interval=5s fails=3 passes=2 uri=/health match=gpu_available;
}

# Define health check match condition
match gpu_available {
    status 200;
    body ~ "gpu_available.*true";
}

server {
    listen 80;
    server_name di.local;
    
    # Large file upload settings
    client_max_body_size 500G;
    client_body_timeout 3600s;
    client_body_buffer_size 10M;
    
    # Temp file handling for large uploads
    client_body_temp_path /data/nginx/temp 1 2;
    
    # Main API endpoints
    location / {
        proxy_pass http://di_nodes;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Connection settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
        
        # Disable buffering
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # Special handling for UFDR uploads
    location /upload-ufdr-data {
        # Route to least loaded node based on custom header
        proxy_pass http://di_nodes;
        proxy_http_version 1.1;
        
        # Headers for tracking
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header X-File-Size $content_length;
        
        # Extended timeouts for large files
        proxy_connect_timeout 60s;
        proxy_send_timeout 7200s;  # 2 hours
        proxy_read_timeout 7200s;  # 2 hours
        
        # Disable all buffering
        proxy_buffering off;
        proxy_request_buffering off;
        
        # Track upload progress
        track_uploads uploads 60s;
    }
    
    # Upload progress endpoint
    location /upload-progress {
        report_uploads uploads;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://di_health;
        access_log off;
    }
    
    # Metrics endpoint for Prometheus
    location /metrics {
        proxy_pass http://di_nodes;
        access_log off;
    }
    
    # WebSocket support for real-time updates
    location /ws {
        proxy_pass http://di_nodes;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}

# GPU metrics server
server {
    listen 8080;
    server_name gpu-metrics.di.local;
    
    location / {
        # Aggregate GPU metrics from all nodes
        proxy_pass http://localhost:9400;  # NVIDIA DCGM exporter
    }
}
```

## 8. Monitoring & Scaling <a id="monitoring-scaling"></a>

### 8.1 Metrics Collection

```python
# monitoring/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge
import psutil
import GPUtil

# Define metrics
task_counter = Counter('di_tasks_total', 'Total tasks processed', ['task_type', 'status'])
task_duration = Histogram('di_task_duration_seconds', 'Task duration', ['task_type'])
gpu_memory_usage = Gauge('di_gpu_memory_usage_bytes', 'GPU memory usage', ['gpu_id', 'worker_id'])
gpu_utilization = Gauge('di_gpu_utilization_percent', 'GPU utilization', ['gpu_id', 'worker_id'])

class MetricsCollector:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        
    def collect_gpu_metrics(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_memory_usage.labels(
                gpu_id=gpu.id,
                worker_id=self.worker_id
            ).set(gpu.memoryUsed * 1024 * 1024)
            
            gpu_utilization.labels(
                gpu_id=gpu.id,
                worker_id=self.worker_id
            ).set(gpu.load * 100)
```

### 8.2 Auto-scaling Configuration

```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpu-worker
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "70"
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 8.3 Monitoring Dashboard

```python
# monitoring/dashboard_config.py
grafana_dashboard = {
    "dashboard": {
        "title": "Digital Intelligence GPU Cluster",
        "panels": [
            {
                "title": "GPU Utilization by Worker",
                "targets": [{
                    "expr": "di_gpu_utilization_percent"
                }]
            },
            {
                "title": "Task Processing Rate",
                "targets": [{
                    "expr": "rate(di_tasks_total[5m])"
                }]
            },
            {
                "title": "GPU Memory Usage",
                "targets": [{
                    "expr": "di_gpu_memory_usage_bytes / (1024^3)"
                }]
            },
            {
                "title": "Task Queue Length",
                "targets": [{
                    "expr": "celery_queue_length{queue=~'gpu_tasks|cpu_tasks'}"
                }]
            }
        ]
    }
}
```

## 9. Migration Roadmap <a id="migration-roadmap"></a>

### Week 1-2: Environment Preparation
- [ ] Set up development cluster (1-2 machines)
- [ ] Configure container registry
- [ ] Deploy MinIO/storage solution
- [ ] Set up monitoring stack (Prometheus/Grafana)

### Week 3-4: Service Extraction
- [ ] Containerize API Gateway
- [ ] Containerize GPU Workers
- [ ] Extract shared utilities
- [ ] Create model registry service

### Week 5-6: Integration Testing
- [ ] Deploy to staging environment
- [ ] Test with sample UFDR files
- [ ] Performance benchmarking
- [ ] Failover testing

### Week 7-8: Production Deployment
- [ ] Deploy to production (phased rollout)
- [ ] Monitor performance metrics
- [ ] Tune resource allocations
- [ ] Documentation and training

### Post-Deployment (Ongoing)
- [ ] Performance optimization
- [ ] Cost optimization
- [ ] Capacity planning
- [ ] Feature enhancements

## Additional Considerations for Self-Hosted Environment

### Windows Compatibility

For Windows deployment, use:
1. **Docker Desktop** for Windows with WSL2 backend
2. **NVIDIA Container Toolkit** for Windows
3. **Windows Storage Spaces** as alternative to GlusterFS
4. **Windows Admin Center** for cluster management

```powershell
# Windows setup script
# Install Docker Desktop
winget install Docker.DockerDesktop

# Install NVIDIA drivers and container toolkit
winget install NVIDIA.CUDA

# Configure Docker for GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Join Docker Swarm
docker swarm join --token <token> <manager-ip>:2377
```

### Backup Strategy

Self-hosted backup solution:
```yaml
# backup-service.yml
version: '3.8'

services:
  restic_backup:
    image: restic/restic:latest
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.backup == true
    environment:
      - RESTIC_REPOSITORY=/backup/di-backup
      - RESTIC_PASSWORD_FILE=/run/secrets/backup_password
    volumes:
      - /data:/data:ro
      - /mnt/glusterfs:/mnt/glusterfs:ro
      - backup_storage:/backup
    command: |
      sh -c '
      while true; do
        restic backup /data /mnt/glusterfs \
          --tag daily \
          --exclude="*.tmp" \
          --exclude="cache/*"
        restic forget --keep-daily 7 --keep-weekly 4 --keep-monthly 6
        restic prune
        sleep 86400
      done'
    secrets:
      - backup_password

volumes:
  backup_storage:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/backup  # Dedicated backup drive

secrets:
  backup_password:
    external: true
```

### Security Hardening

1. **Network Isolation**:
```yaml
# network-policies.yml
networks:
  di_public:
    driver: overlay
    encrypted: true
    
  di_internal:
    driver: overlay
    encrypted: true
    internal: true  # No external access
    
  di_storage:
    driver: overlay
    encrypted: true
    internal: true
```

2. **TLS Configuration**:
```nginx
# tls-config.conf
server {
    listen 443 ssl http2;
    
    ssl_certificate /etc/nginx/certs/di.crt;
    ssl_certificate_key /etc/nginx/certs/di.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Rest of configuration...
}
```

## Conclusion

This enhanced distributed architecture provides:

1. **Full GPU Utilization**: Every node contributes GPU resources
2. **Complete Self-Hosting**: No external cloud dependencies
3. **Hybrid Node Design**: Efficient resource usage with combined API/Worker nodes
4. **Local Performance**: Fast local storage with distributed replication
5. **Cross-Platform**: Supports both Linux and Windows deployments
6. **Scalability**: From 1 to N nodes with automatic load distribution

Key improvements over original design:
- **GPU Efficiency**: 100% GPU utilization across all nodes
- **Storage Performance**: Local NVMe/SSD with async replication
- **Reduced Complexity**: Single node type instead of separate API/Worker nodes
- **Self-Contained**: All services run on-premises
- **Cost Effective**: No cloud service fees

This architecture maximizes the hardware investment by ensuring every GPU is available for processing while maintaining high availability and scalability for processing large UFDR files in production environments.
