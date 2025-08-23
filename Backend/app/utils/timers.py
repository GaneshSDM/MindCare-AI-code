# path: mindcare-backend/app/utils/timers.py
import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution and log the duration.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Add timing info to logger record
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.latency_ms = duration_ms
            return record
        
        logging.setLogRecordFactory(record_factory)
        logger.info(f"Function {func.__name__} executed in {duration_ms:.2f}ms")
        logging.setLogRecordFactory(old_factory)
        
        return result
    
    return wrapper