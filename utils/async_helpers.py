"""
Async helpers for Celery tasks to handle event loops properly.
All platforms use threads pool, so each Celery worker thread maintains
its own event loop via thread-local storage.
"""
import asyncio
import threading
from typing import Any, Coroutine
import logging

logger = logging.getLogger(__name__)

# Thread-local storage for event loops (one loop per Celery worker thread)
_thread_loops = threading.local()


def get_or_create_event_loop():
    """
    Get or create an event loop for the current thread.
    Each Celery worker thread maintains its own loop.
    """
    try:
        if hasattr(_thread_loops, 'loop') and _thread_loops.loop and not _thread_loops.loop.is_closed():
            return _thread_loops.loop
    except Exception:
        pass
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _thread_loops.loop = loop
    return loop


def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Run an async coroutine using the current thread's event loop.
    """
    loop = get_or_create_event_loop()
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"Error running async task in thread: {str(e)}")
            raise


def run_async_task(coro: Coroutine) -> Any:
    """Run an async coroutine from a sync Celery task context."""
    return run_async_in_thread(coro)


def cleanup_thread_loop():
    """Clean up the event loop for the current thread."""
    try:
        if hasattr(_thread_loops, 'loop') and _thread_loops.loop:
            loop = _thread_loops.loop
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            delattr(_thread_loops, 'loop')
    except Exception as e:
        logger.error(f"Error cleaning up event loop: {str(e)}")
