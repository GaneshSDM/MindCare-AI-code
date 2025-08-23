# path: mindcare-backend/app/logging_conf.py
import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path

from app import config

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'latency_ms'):
            log_record['latency_ms'] = record.latency_ms
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)

def setup_logging():
    """Setup logging configuration"""
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": log_dir / "mindcare.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)