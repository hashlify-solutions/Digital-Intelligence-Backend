import torch
import logging
import subprocess
import time
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Any, Optional
import psutil
import os

logger = logging.getLogger(__name__)

# Track if we've already warned about enhanced optimization failure to avoid noisy logs
_optimization_failure_warned = False

class GPUManager:
    """Manages GPU resources for parallel processing"""
    
    def __init__(self, max_gpus: int = None):
        self.max_gpus = max_gpus or torch.cuda.device_count()
        self.available_gpus = list(range(min(self.max_gpus, torch.cuda.device_count())))
        self.gpu_usage = {gpu: 0 for gpu in self.available_gpus}
        self.gpu_locks = {gpu: asyncio.Lock() for gpu in self.available_gpus}
        
        logger.info(f"GPUManager initialized with {len(self.available_gpus)} GPUs: {self.available_gpus}")
    
    async def get_available_gpu(self, required_memory_mb: int = 2048) -> Optional[int]:
        """Get an available GPU with sufficient memory"""
        for gpu in self.available_gpus:
            async with self.gpu_locks[gpu]:
                if self._check_gpu_memory(gpu, required_memory_mb):
                    self.gpu_usage[gpu] += 1
                    logger.debug(f"Allocated GPU {gpu} for processing")
                    return gpu
        return None
    
    async def release_gpu(self, gpu_id: int):
        """Release a GPU after processing"""
        if gpu_id in self.gpu_usage:
            async with self.gpu_locks[gpu_id]:
                self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
                logger.debug(f"Released GPU {gpu_id}")
    
    def _check_gpu_memory(self, gpu_id: int, required_memory_mb: int) -> bool:
        """Check if GPU has sufficient available memory"""
        try:
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
            available = total - allocated
            return available >= required_memory_mb
        except Exception as e:
            logger.error(f"Error checking GPU {gpu_id} memory: {e}")
            return False
    
    def get_gpu_utilization(self) -> Dict[int, float]:
        """Get current GPU utilization percentages"""
        utilization = {}
        for gpu in self.available_gpus:
            try:
                result = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=utilization.gpu', 
                    '--format=csv,noheader,nounits', f'--id={gpu}'
                ])
                utilization[gpu] = float(result.decode().strip())
            except Exception as e:
                logger.warning(f"Could not get utilization for GPU {gpu}: {e}")
                utilization[gpu] = 0.0
        return utilization

# Global GPU manager instance
gpu_manager = GPUManager()

def setup_model_for_multi_gpu(model, gpu_id: Optional[int] = None):
    """Setup model for multi-GPU processing with specific GPU assignment"""
    if gpu_id is not None and gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_id}')
        model = model.to(device)
        logger.info(f"Model assigned to GPU {gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        else:
            logger.info(f"Using single GPU: {device}")
    else:
        device = torch.device('cpu')
        model = model.to(device)
        logger.info(f"Using CPU device")
    
    return model, device

_last_gpu_log_time = 0
_GPU_LOG_INTERVAL = 300  # 5 minutes in seconds

def log_gpu_utilization():
    """Log GPU utilization with enhanced monitoring"""
    global _last_gpu_log_time
    now = time.time()
    if now - _last_gpu_log_time < _GPU_LOG_INTERVAL:
        return
    _last_gpu_log_time = now
    
    if torch.cuda.is_available():
        logger.info("=== GPU Utilization Report ===")
        for i in range(torch.cuda.device_count()):
            try:
                # Memory information
                mem_alloc = torch.cuda.memory_allocated(i) / 1024**2
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**2
                mem_util = (mem_alloc / total_mem) * 100
                
                # GPU utilization
                result = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=utilization.gpu', 
                    '--format=csv,noheader,nounits', f'--id={i}'
                ])
                utilization = int(result.decode().strip())
                
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  Memory: {mem_alloc:.1f}MB / {total_mem:.1f}MB ({mem_util:.1f}%)")
                logger.info(f"  Utilization: {utilization}%")
                
                # Warning for low utilization
                if utilization < 80 and mem_util > 20:
                    logger.warning(f"GPU {i} has low utilization ({utilization}%) but high memory usage ({mem_util:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"Could not fetch GPU {i} information: {e}")
    else:
        logger.info("No CUDA devices available")

def adjust_batch_size_for_utilization(current_batch_size: int, target_utilization: float = 95, 
                                    check_interval: int = 10) -> int:
    """Dynamically adjust batch size based on GPU utilization"""
    if not torch.cuda.is_available():
        return current_batch_size
    
    try:
        # Get current GPU utilization
        utilization = gpu_manager.get_gpu_utilization()
        avg_utilization = np.mean(list(utilization.values()))
        
        if avg_utilization < target_utilization - 10:
            # Increase batch size if utilization is too low
            new_batch_size = min(current_batch_size * 2, current_batch_size + 16)
            logger.info(f"Low GPU utilization ({avg_utilization:.1f}%), increasing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        elif avg_utilization > target_utilization + 10:
            # Decrease batch size if utilization is too high
            new_batch_size = max(current_batch_size // 2, 1)
            logger.info(f"High GPU utilization ({avg_utilization:.1f}%), decreasing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        else:
            return current_batch_size
            
    except Exception as e:
        logger.error(f"Error adjusting batch size: {e}")
        return current_batch_size

def gpu_logging_decorator(func):
    """Decorator to log GPU utilization before and after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_gpu_utilization()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log_gpu_utilization()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

async def async_gpu_logging_decorator(func):
    """Async decorator to log GPU utilization before and after async function execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        log_gpu_utilization()
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        log_gpu_utilization()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class BatchProcessor:
    """Handles batch processing with GPU optimization"""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_batch_parallel(self, items: List[Any], process_func, 
                                   gpu_required: bool = True) -> List[Any]:
        """Process items in parallel batches with GPU optimization"""
        if not items:
            return []
        
        # Split items into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            if gpu_required:
                task = asyncio.create_task(self._process_batch_with_gpu(batch, process_func))
            else:
                task = asyncio.create_task(self._process_batch_cpu(batch, process_func))
            tasks.append(task)
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        processed_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
            else:
                processed_items.extend(result)
        
        return processed_items
    
    async def _process_batch_with_gpu(self, batch: List[Any], process_func) -> List[Any]:
        """Process a batch using GPU resources"""
        # Get available GPU
        gpu_id = await gpu_manager.get_available_gpu()
        if gpu_id is None:
            logger.warning("No GPU available, falling back to CPU")
            return await self._process_batch_cpu(batch, process_func)
        
        try:
            # Process batch with GPU
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_gpu_processing,
                batch, process_func, gpu_id
            )
            return result
        finally:
            # Always release GPU
            await gpu_manager.release_gpu(gpu_id)
    
    async def _process_batch_cpu(self, batch: List[Any], process_func) -> List[Any]:
        """Process a batch using CPU"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_cpu_processing,
            batch, process_func
        )
    
    def _run_gpu_processing(self, batch: List[Any], process_func, gpu_id: int) -> List[Any]:
        """Run GPU processing in a separate thread"""
        try:
            # Set CUDA device for this thread
            torch.cuda.set_device(gpu_id)
            return [process_func(item) for item in batch]
        except Exception as e:
            logger.error(f"GPU processing error on GPU {gpu_id}: {e}")
            return []
    
    def _run_cpu_processing(self, batch: List[Any], process_func) -> List[Any]:
        """Run CPU processing in a separate thread"""
        try:
            return [process_func(item) for item in batch]
        except Exception as e:
            logger.error(f"CPU processing error: {e}")
            return []

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(i) / 1024**2
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**2
                gpu_info[f"gpu_{i}"] = {
                    "memory_allocated_mb": mem_alloc,
                    "total_memory_mb": total_mem,
                    "memory_utilization_percent": (mem_alloc / total_mem) * 100
                }
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_gb": memory.total / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_gb": memory.used / 1024**3,
                "percent": memory.percent,
                "free_gb": memory.free / 1024**3
            },
            "gpu_info": gpu_info
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        # Return safe defaults instead of empty dict
        return {
            "cpu": {
                "usage_percent": 50,
                "count": 4,
                "count_logical": 4
            },
            "memory": {
                "total_gb": 8.0,
                "available_gb": 4.0,
                "used_gb": 4.0,
                "percent": 50.0,
                "free_gb": 4.0
            },
            "gpu_info": {}
        }

def calculate_optimal_batch_size(
    memory_available_gb: float = 8.0,
    cpu_cores: int = 4,
    gpu_memory_mb: Optional[float] = None,
    document_avg_size_kb: float = 2.0
) -> Dict[str, Any]:
    """
    Calculate optimal batch size with enhanced memory management and GPU optimization
    
    Args:
        memory_available_gb: Available system RAM in GB
        cpu_cores: Number of CPU cores available
        gpu_memory_mb: Available GPU memory in MB
        document_avg_size_kb: Average document size in KB
    
    Returns:
        Dictionary with optimal configuration
    """
    try:
        # ENHANCED: More sophisticated memory calculation
        # Base memory per document (text + embeddings + model overhead)
        base_memory_per_doc_mb = 0.5  # Base text processing
        embedding_memory_per_doc_mb = 0.3  # Embedding vectors
        model_overhead_mb = 0.2  # Model state and gradients
        
        # GPU memory considerations
        gpu_memory_factor = 1.0
        if gpu_memory_mb:
            # Reserve 20% of GPU memory for model weights and intermediate states
            available_gpu_mb = gpu_memory_mb * 0.8
            # Calculate how many documents can fit in GPU memory
            gpu_docs_capacity = int(available_gpu_mb / (base_memory_per_doc_mb + embedding_memory_per_doc_mb))
            gpu_memory_factor = min(2.0, gpu_docs_capacity / 100)  # Cap at 2x multiplier
        
        # ENHANCED: Dynamic batch size calculation based on available resources
        # Start with conservative estimate
        base_batch_size = 16
        
        # Adjust based on available RAM
        if memory_available_gb >= 32:
            base_batch_size = 64
        elif memory_available_gb >= 16:
            base_batch_size = 48
        elif memory_available_gb >= 8:
            base_batch_size = 32
        elif memory_available_gb >= 4:
            base_batch_size = 24
        else:
            base_batch_size = 16
        
        # Apply GPU memory factor
        gpu_adjusted_batch_size = int(base_batch_size * gpu_memory_factor)
        
        # ENHANCED: CPU core optimization - target 75% of available cores
        # Optimal worker count based on CPU cores and batch size
        target_cpu_utilization = 0.75  # Use 75% of available CPU cores
        optimal_workers = int(cpu_cores * target_cpu_utilization)
        
        # Ensure workers don't exceed batch size constraints
        optimal_workers = min(optimal_workers, gpu_adjusted_batch_size // 2)
        optimal_workers = max(4, optimal_workers)  # Minimum 4 workers for better parallelism
        
        # ENHANCED: Final batch size calculation with safety margins
        # Use 70% of available memory to leave room for system processes
        safe_memory_gb = memory_available_gb * 0.7
        
        # Calculate how many documents can fit in safe memory
        memory_per_doc_gb = (base_memory_per_doc_mb + embedding_memory_per_doc_mb + model_overhead_mb) / 1024
        memory_based_batch_size = int(safe_memory_gb / memory_per_doc_gb)
        
        # Take the minimum of GPU-adjusted and memory-based batch sizes
        final_batch_size = min(gpu_adjusted_batch_size, memory_based_batch_size)
        
        # Ensure batch size is within reasonable bounds
        final_batch_size = max(8, min(128, final_batch_size))
        
        # ENHANCED: Memory efficiency calculation
        estimated_memory_usage_mb = final_batch_size * (base_memory_per_doc_mb + embedding_memory_per_doc_mb + model_overhead_mb)
        memory_efficiency = (estimated_memory_usage_mb / 1024) / memory_available_gb
        
        # ENHANCED: Detailed reasoning for configuration choices
        gpu_memory_str = f"{gpu_memory_mb:.0f}MB" if gpu_memory_mb is not None else "No GPU"
        reasoning = {
            "memory_limit": f"Safe memory: {safe_memory_gb:.1f}GB, per_doc: {memory_per_doc_gb:.4f}GB",
            "cpu_limit": f"CPU cores: {cpu_cores}, optimal_workers: {optimal_workers}",
            "gpu_limit": f"GPU memory: {gpu_memory_str}, factor: {gpu_memory_factor:.2f}",
            "batch_calculation": f"Base: {base_batch_size}, GPU-adjusted: {gpu_adjusted_batch_size}, Final: {final_batch_size}"
        }
        
        return {
            "batch_size": final_batch_size,
            "max_workers": optimal_workers,
            "memory_efficiency": memory_efficiency,
            "estimated_memory_usage_mb": estimated_memory_usage_mb,
            "gpu_memory_factor": gpu_memory_factor,
            "reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"Error calculating optimal batch size: {e}")
        # Return safe fallback values
        return {
            "batch_size": 16,
            "max_workers": 4,
            "memory_efficiency": 0.1,
            "estimated_memory_usage_mb": 800,
            "gpu_memory_factor": 1.0,
            "reasoning": {"error": str(e)}
        }

def optimize_for_system_resources() -> Dict[str, Any]:
    """
    Enhanced system optimization with better resource utilization
    """
    try:
        # Get current system resources
        resources = get_system_resources()
        # Safely resolve CPU cores with multiple fallbacks
        resolved_cpu_cores = (
            resources.get("cpu", {}).get("count")
            or (psutil.cpu_count() if 'psutil' in globals() else None)
            or (os.cpu_count() if 'os' in globals() else None)
            or 4
        )
        
        # Calculate optimal configuration with safe memory access
        memory_data = resources.get("memory", {})
        memory_available_gb = memory_data.get("available_gb", 8) if memory_data else 8
        gpu_info = resources.get("gpu_info", {})
        gpu_memory_mb = gpu_info.get("gpu_0", {}).get("total_memory_mb", None) if gpu_info else None
        
        optimal_config = calculate_optimal_batch_size(
            memory_available_gb=memory_available_gb,
            cpu_cores=resolved_cpu_cores,
            gpu_memory_mb=gpu_memory_mb
        )
        
        # ENHANCED: Additional optimizations based on current system state
        cpu_data = resources.get("cpu", {})
        memory_data = resources.get("memory", {})
        cpu_usage = cpu_data.get("usage_percent", 50) if cpu_data else 50
        memory_usage = memory_data.get("percent", 50) if memory_data else 50
        
        # Enable global CUDA optimizations
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Global CUDA optimizations enabled: cudnn.benchmark, tf32, matmul optimizations")
            except Exception as e:
                logger.warning(f"Could not enable global CUDA optimizations: {e}")
        
        # Adjust based on current system load
        if cpu_usage < 30 and memory_usage < 60:
            # System is underutilized, can be more aggressive
            # Ensure we keep integer values (avoid float multiplication results)
            try:
                increased_batch = int(min(128, int(optimal_config.get("batch_size", 0)) * 1.5))
            except Exception:
                increased_batch = int(optimal_config.get("batch_size", 16))
            try:
                increased_workers = int(min(16, max(1, int(optimal_config.get("max_workers", 1))) * 1.5))
            except Exception:
                increased_workers = int(optimal_config.get("max_workers", 4))

            optimal_config["batch_size"] = increased_batch
            optimal_config["max_workers"] = increased_workers
            optimal_config["reasoning"]["load_adjustment"] = "Low system load - increased batch size and workers"
        elif cpu_usage > 80 or memory_usage > 85:
            # System is under pressure, be more conservative
            try:
                optimal_config["batch_size"] = max(8, int(optimal_config.get("batch_size", 16)) // 2)
            except Exception:
                optimal_config["batch_size"] = 8
            try:
                optimal_config["max_workers"] = max(2, int(optimal_config.get("max_workers", 4)) // 2)
            except Exception:
                optimal_config["max_workers"] = 2
            optimal_config["reasoning"]["load_adjustment"] = "High system load - reduced batch size and workers"
        else:
            optimal_config["reasoning"]["load_adjustment"] = "Normal system load - using calculated optimal values"
        
        # ENHANCED: Log optimization details with safe None handling
        logger.info(f"Enhanced system optimization completed:")
        logger.info(f"  Batch size: {optimal_config.get('batch_size', 'unknown')}")
        logger.info(f"  Max workers: {optimal_config.get('max_workers', 'unknown')}")
        logger.info(f"  Memory efficiency: {optimal_config.get('memory_efficiency', 0):.2%}")
        logger.info(f"  GPU memory factor: {optimal_config.get('gpu_memory_factor', 0):.2f}")
        max_workers = optimal_config.get('max_workers', 'unknown')
        logger.info(f"  CPU utilization target: 75% ({max_workers}/{resolved_cpu_cores} cores)")
        reasoning = optimal_config.get('reasoning', {})
        logger.info(f"  Load adjustment: {reasoning.get('load_adjustment', 'unknown')}")
        
        return optimal_config
        
    except Exception as e:
        global _optimization_failure_warned
        if not _optimization_failure_warned:
            logger.warning(f"Enhanced system optimization failed, using safe defaults: {e}")
            _optimization_failure_warned = True
        else:
            logger.debug(f"Enhanced system optimization failed again: {e}")
        # Return safe fallback
        return {
            "batch_size": 16,
            "max_workers": 4,
            "memory_efficiency": 0.1,
            "estimated_memory_usage_mb": 800,
            "gpu_memory_factor": 1.0,
            "reasoning": {"error": "optimization_failed"}
        }