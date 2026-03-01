import os
import logging
import multiprocessing as mp
from typing import Optional, Dict, Any
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic BaseSettings for automatic type conversion and validation.
    """
    
    # Application Settings
    app_name: str = Field(env="APP_NAME")
    app_version: str = Field(env="APP_VERSION")
    debug: bool = Field(env="DEBUG")
    environment: str = Field(env="ENVIRONMENT")
    
    # Platform Settings
    platform: str = Field(env="PLATFORM")
    
    # Server Settings
    host: str = Field(env="HOST")
    port: int = Field(env="PORT")
    
    # Database Settings - MongoDB
    mongo_uri: Optional[str] = Field(env="MONGO_URI")
    mongo_username: str = Field(env="MONGO_USERNAME")
    mongo_password: str = Field(env="MONGO_PASSWORD")
    mongo_host: str = Field(env="MONGO_HOST")
    mongo_database: str = Field(env="MONGO_DATABASE")
    
    # Vector Database Settings - Qdrant
    qdrant_url: str = Field(
        env="QDRANT_URL"
    )
    qdrant_api_key: str = Field(
        env="QDRANT_API_KEY"
    )
    
    # Graph Database Settings - Neo4j
    neo4j_uri: str = Field(env="NEO4J_URI", default="bolt://localhost:7687")
    neo4j_user: str = Field(env="NEO4J_USER", default="neo4j")
    neo4j_password: str = Field(env="NEO4J_PASSWORD", default="password")
    neo4j_database: str = Field(env="NEO4J_DATABASE", default=None)
    
    # Celery Settings
    celery_app_name: str = Field(env="CELERY_APP_NAME")
    celery_broker_url: str = Field(env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(env="CELERY_RESULT_BACKEND")
    
    # Redis Settings
    redis_url: str = Field(env="REDIS_URL")
    
    # AI/ML Model Settings
    local_models: bool = Field(env="LOCAL_MODELS")
    di_device: str = Field(env="DI_DEVICE")
    
    # Security Settings
    secret_key: str = Field(env="SECRET_KEY")
    jwt_secret_key: str = Field(env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_minutes: int = Field(env="REFRESH_TOKEN_EXPIRE_MINUTES")
    
    # File Storage Settings
    upload_dir: str = Field(env="UPLOAD_DIR")
    logo_dir: str = Field(env="LOGO_DIR")
    max_file_size: int = Field(env="MAX_FILE_SIZE")  
    
    # Logging Settings
    log_dir: str = Field(env="LOG_DIR")
    log_level: str = Field(env="LOG_LEVEL")
    log_file: str = Field(env="LOG_FILE")
    
    # CORS Settings
    cors_origins: str = Field(env="CORS_ORIGINS")
    
    # Parallel Processing Settings (optional - auto-detected if not set)
    parallel_max_workers: Optional[int] = Field(default=None, env="PARALLEL_MAX_WORKERS")
    parallel_batch_size: Optional[int] = Field(default=None, env="PARALLEL_BATCH_SIZE")
    parallel_gpu_batch_size: Optional[int] = Field(default=None, env="PARALLEL_GPU_BATCH_SIZE")
    parallel_embedding_batch_size: Optional[int] = Field(default=None, env="PARALLEL_EMBEDDING_BATCH_SIZE")
    parallel_max_concurrent_llm: Optional[int] = Field(default=None, env="PARALLEL_MAX_CONCURRENT_LLM")
    parallel_chunk_size: Optional[int] = Field(default=None, env="PARALLEL_CHUNK_SIZE")
    parallel_use_multiprocessing: Optional[bool] = Field(default=None, env="PARALLEL_USE_MULTIPROCESSING")
    parallel_enable_monitoring: Optional[bool] = Field(default=True, env="PARALLEL_ENABLE_MONITORING")
    parallel_enable_auto_optimization: Optional[bool] = Field(default=True, env="PARALLEL_ENABLE_AUTO_OPTIMIZATION")
    parallel_max_retries: Optional[int] = Field(default=3, env="PARALLEL_MAX_RETRIES")
    parallel_timeout_seconds: Optional[int] = Field(default=300, env="PARALLEL_TIMEOUT_SECONDS")
    
    @property
    def mongo_connection_string(self) -> str:
        """
        Generate MongoDB connection string from individual components.
        Uses MONGO_URI if provided, otherwise constructs from individual components.
        """
        if self.mongo_uri:
            return self.mongo_uri
        # URL encode username and password for safety
        username = quote_plus(self.mongo_username)
        password = quote_plus(self.mongo_password)
        return f"mongodb+srv://{username}:{password}@{self.mongo_host}/{self.mongo_database}"
    
    @property
    def cors_origins_list(self) -> list:
        """Convert CORS origins string to list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if the application is running in production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if the application is running in development."""
        return self.environment.lower() == "development"
    
    @property
    def compute_config(self) -> Dict[str, Any]:
        """
        Auto-detect hardware and return optimal compute configuration.
        
        Returns a dictionary with:
        - Hardware info: cpu_count, ram_gb, gpu_count, gpu_memory_gb
        - Processing settings: max_workers, batch_size, gpu_batch_size, etc.
        
        Values can be overridden via environment variables (PARALLEL_*).
        If not set, optimal values are calculated based on detected hardware.
        """
        try:
            import psutil
            import torch
        except ImportError as e:
            logger.warning(f"Could not import hardware detection libraries: {e}")
            # Return safe defaults if libraries not available
            return self._get_fallback_compute_config()
        
        try:
            # Enable global CUDA optimizations on first access
            if torch.cuda.is_available():
                try:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.set_float32_matmul_precision('medium')
                except Exception as cuda_e:
                    logger.warning(f"Could not enable global CUDA optimizations: {cuda_e}")
            
            # Detect hardware
            cpu_count = mp.cpu_count() or 4
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # GPU detection
            gpu_count = 0
            gpu_memory_gb = 0.0
            gpu_name = "None"
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_name = torch.cuda.get_device_name(0)
            
            # Calculate optimal values based on hardware
            # RTX 5080: ~16GB VRAM, can handle large batches
            # RTX 4090: ~24GB VRAM
            # RTX 3090: ~24GB VRAM
            # RTX 3080: ~10GB VRAM
            if gpu_memory_gb >= 16:
                # High-end GPU (RTX 5080, 4090, 3090, A100, etc.)
                default_batch_size = 256
                default_gpu_batch = 128
                default_embedding_batch = 512
            elif gpu_memory_gb >= 8:
                # Mid-range GPU (RTX 3080, 4070, etc.)
                default_batch_size = 128
                default_gpu_batch = 64
                default_embedding_batch = 256
            elif gpu_memory_gb >= 4:
                # Entry-level GPU
                default_batch_size = 64
                default_gpu_batch = 32
                default_embedding_batch = 128
            else:
                # No GPU or very low VRAM
                default_batch_size = 32
                default_gpu_batch = 16
                default_embedding_batch = 64
            
            # Workers: 2x CPU cores if GPU available, else 1x, capped at 32
            default_workers = min(cpu_count * (2 if gpu_count > 0 else 1), 32)
            
            # LLM concurrency based on RAM (each LLM request uses ~0.5-1GB)
            default_llm_concurrent = min(int(ram_gb / 2), 32)
            
            # Chunk size for streaming file reading (50k records default)
            default_chunk_size = 50000
            
            # Use multiprocessing on Linux, threading on macOS/Windows
            import platform
            default_use_multiprocessing = platform.system().lower() == "linux"
            
            config = {
                # Hardware info
                "cpu_count": cpu_count,
                "ram_gb": round(ram_gb, 1),
                "gpu_count": gpu_count,
                "gpu_memory_gb": round(gpu_memory_gb, 1),
                "gpu_name": gpu_name,
                # Processing settings (with env overrides)
                "max_workers": self.parallel_max_workers or default_workers,
                "batch_size": self.parallel_batch_size or default_batch_size,
                "gpu_batch_size": self.parallel_gpu_batch_size or default_gpu_batch,
                "embedding_batch_size": self.parallel_embedding_batch_size or default_embedding_batch,
                "max_concurrent_llm": self.parallel_max_concurrent_llm or default_llm_concurrent,
                "chunk_size": self.parallel_chunk_size or default_chunk_size,
                "use_multiprocessing": self.parallel_use_multiprocessing if self.parallel_use_multiprocessing is not None else default_use_multiprocessing,
                "enable_monitoring": self.parallel_enable_monitoring,
                "enable_auto_optimization": self.parallel_enable_auto_optimization,
                "max_retries": self.parallel_max_retries,
                "timeout_seconds": self.parallel_timeout_seconds,
            }
            
            logger.info(f"Compute config initialized: {cpu_count} CPUs, {round(ram_gb, 1)}GB RAM, "
                       f"{gpu_count} GPU(s) ({gpu_name}, {round(gpu_memory_gb, 1)}GB VRAM)")
            logger.info(f"Processing config: workers={config['max_workers']}, batch={config['batch_size']}, "
                       f"gpu_batch={config['gpu_batch_size']}, embedding_batch={config['embedding_batch_size']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error detecting hardware configuration: {e}")
            return self._get_fallback_compute_config()
    
    def _get_fallback_compute_config(self) -> Dict[str, Any]:
        """Return safe fallback configuration when hardware detection fails."""
        return {
            "cpu_count": 4,
            "ram_gb": 8.0,
            "gpu_count": 0,
            "gpu_memory_gb": 0.0,
            "gpu_name": "Unknown",
            "max_workers": self.parallel_max_workers or 4,
            "batch_size": self.parallel_batch_size or 32,
            "gpu_batch_size": self.parallel_gpu_batch_size or 16,
            "embedding_batch_size": self.parallel_embedding_batch_size or 64,
            "max_concurrent_llm": self.parallel_max_concurrent_llm or 8,
            "chunk_size": self.parallel_chunk_size or 50000,
            "use_multiprocessing": self.parallel_use_multiprocessing or False,
            "enable_monitoring": self.parallel_enable_monitoring,
            "enable_auto_optimization": self.parallel_enable_auto_optimization,
            "max_retries": self.parallel_max_retries,
            "timeout_seconds": self.parallel_timeout_seconds,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Create a global settings instance
settings = Settings()

# For backward compatibility and easy access
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
