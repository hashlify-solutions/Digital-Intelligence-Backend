"""
Performance monitoring module for parallel processing
"""
import time
import logging
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for processing performance"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    processing_time_seconds: float = 0.0
    documents_per_second: float = 0.0
    batch_count: int = 0
    average_batch_time: float = 0.0
    gpu_utilization: Dict[str, float] = field(default_factory=dict)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    parallel_processing_enabled: bool = True
    model_inference_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    database_operation_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

class PerformanceMonitor:
    """Monitor and track performance metrics for parallel processing"""
    
    def __init__(self, case_id: str, save_metrics: bool = True):
        self.case_id = case_id
        self.save_metrics = save_metrics
        self.metrics = ProcessingMetrics(start_time=datetime.now())
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Create metrics directory if it doesn't exist
        if self.save_metrics:
            os.makedirs("metrics", exist_ok=True)
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitor_system_resources())
        logger.info(f"Performance monitoring started for case {self.case_id}")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Performance monitoring stopped for case {self.case_id}")
    
    async def _monitor_system_resources(self):
        """Monitor system resources in the background"""
        while self.monitoring_active:
            try:
                # Get CPU and memory utilization
                self.metrics.cpu_utilization = psutil.cpu_percent(interval=1)
                self.metrics.memory_utilization = psutil.virtual_memory().percent
                
                # Get GPU utilization if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_name = f"gpu_{i}"
                            mem_alloc = torch.cuda.memory_allocated(i) / 1024**2
                            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**2
                            self.metrics.gpu_utilization[gpu_name] = (mem_alloc / total_mem) * 100
                except Exception as e:
                    logger.debug(f"Could not get GPU utilization: {e}")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logger.error(f"Error in system resource monitoring: {e}")
                await asyncio.sleep(5)
    
    def record_model_inference(self, model_type: str, inference_time: float):
        """Record model inference time"""
        self.metrics.model_inference_times[model_type].append(inference_time)
    
    def record_database_operation(self, operation_type: str, operation_time: float):
        """Record database operation time"""
        self.metrics.database_operation_times[operation_type].append(operation_time)
    
    def update_processing_stats(self, 
                              total_documents: int = None,
                              processed_documents: int = None,
                              failed_documents: int = None,
                              batch_count: int = None):
        """Update processing statistics"""
        if total_documents is not None:
            self.metrics.total_documents = total_documents
        if processed_documents is not None:
            self.metrics.processed_documents = processed_documents
        if failed_documents is not None:
            self.metrics.failed_documents = failed_documents
        if batch_count is not None:
            self.metrics.batch_count = batch_count
    
    def finalize_metrics(self):
        """Finalize metrics and calculate performance statistics"""
        self.metrics.end_time = datetime.now()
        self.metrics.processing_time_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()
        
        if self.metrics.processing_time_seconds > 0:
            self.metrics.documents_per_second = self.metrics.processed_documents / self.metrics.processing_time_seconds
        
        if self.metrics.batch_count > 0:
            self.metrics.average_batch_time = self.metrics.processing_time_seconds / self.metrics.batch_count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        self.finalize_metrics()
        
        # Calculate average model inference times
        avg_model_times = {}
        for model_type, times in self.metrics.model_inference_times.items():
            if times:
                avg_model_times[model_type] = {
                    "average_time": sum(times) / len(times),
                    "total_calls": len(times),
                    "total_time": sum(times)
                }
        
        # Calculate average database operation times
        avg_db_times = {}
        for op_type, times in self.metrics.database_operation_times.items():
            if times:
                avg_db_times[op_type] = {
                    "average_time": sum(times) / len(times),
                    "total_operations": len(times),
                    "total_time": sum(times)
                }
        
        summary = {
            "case_id": self.case_id,
            "processing_time_seconds": self.metrics.processing_time_seconds,
            "total_documents": self.metrics.total_documents,
            "processed_documents": self.metrics.processed_documents,
            "failed_documents": self.metrics.failed_documents,
            "documents_per_second": self.metrics.documents_per_second,
            "batch_count": self.metrics.batch_count,
            "average_batch_time": self.metrics.average_batch_time,
            "parallel_processing_enabled": self.metrics.parallel_processing_enabled,
            "system_resources": {
                "cpu_utilization": self.metrics.cpu_utilization,
                "memory_utilization": self.metrics.memory_utilization,
                "gpu_utilization": dict(self.metrics.gpu_utilization)
            },
            "model_performance": avg_model_times,
            "database_performance": avg_db_times,
            "start_time": self.metrics.start_time.isoformat(),
            "end_time": self.metrics.end_time.isoformat() if self.metrics.end_time else None
        }
        
        return summary
    
    def save_metrics_to_file(self, filename: str = None):
        """Save metrics to a JSON file"""
        if not self.save_metrics:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics/case_{self.case_id}_{timestamp}.json"
        
        summary = self.get_performance_summary()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Performance metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
    
    def print_performance_summary(self):
        """Print a formatted performance summary to console"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print(f"PERFORMANCE SUMMARY - Case {self.case_id}")
        print("="*60)
        print(f"Processing Time: {summary['processing_time_seconds']:.2f} seconds")
        print(f"Total Documents: {summary['total_documents']}")
        print(f"Processed Documents: {summary['processed_documents']}")
        print(f"Failed Documents: {summary['failed_documents']}")
        print(f"Documents per Second: {summary['documents_per_second']:.2f}")
        print(f"Batch Count: {summary['batch_count']}")
        print(f"Average Batch Time: {summary['average_batch_time']:.2f} seconds")
        print(f"Parallel Processing: {'Enabled' if summary['parallel_processing_enabled'] else 'Disabled'}")
        
        print(f"\nSystem Resources:")
        print(f"  CPU Utilization: {summary['system_resources']['cpu_utilization']:.1f}%")
        print(f"  Memory Utilization: {summary['system_resources']['memory_utilization']:.1f}%")
        
        if summary['system_resources']['gpu_utilization']:
            print(f"  GPU Utilization:")
            for gpu, util in summary['system_resources']['gpu_utilization'].items():
                print(f"    {gpu}: {util:.1f}%")
        
        if summary['model_performance']:
            print(f"\nModel Performance:")
            for model, perf in summary['model_performance'].items():
                print(f"  {model}: {perf['average_time']:.3f}s avg ({perf['total_calls']} calls)")
        
        if summary['database_performance']:
            print(f"\nDatabase Performance:")
            for op, perf in summary['database_performance'].items():
                print(f"  {op}: {perf['average_time']:.3f}s avg ({perf['total_operations']} operations)")
        
        print("="*60)

class PerformanceTracker:
    """Context manager for tracking performance of specific operations"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_database_operation(self.operation_name, duration)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_database_operation(self.operation_name, duration)

def create_performance_monitor(case_id: str, save_metrics: bool = True) -> PerformanceMonitor:
    """Factory function to create a performance monitor"""
    return PerformanceMonitor(case_id, save_metrics)

def track_operation(monitor: PerformanceMonitor, operation_name: str) -> PerformanceTracker:
    """Create a performance tracker for a specific operation"""
    return PerformanceTracker(monitor, operation_name) 