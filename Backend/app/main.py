# path: mindcare-backend/app/main.py
import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routers import ingest, features, model, risk, career, rag
from app.logging_conf import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MindCare AI Backend",
    description="HR Copilot for Decision Minds - Offline Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(features.router, prefix="/api/v1", tags=["features"])
app.include_router(model.router, prefix="/api/v1", tags=["model"])
app.include_router(risk.router, prefix="/api/v1", tags=["risk"])
app.include_router(career.router, prefix="/api/v1", tags=["career"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# Health check endpoint
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "message": "MindCare AI Backend is running"}

# Initialize data and models
@app.on_event("startup")
async def startup_event():
    logger.info("Starting MindCare AI Backend")
    # Ensure directories exist
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    Path(config.ARTIFACTS_DIR).mkdir(exist_ok=True)
    Path(config.VECTOR_INDEX_DIR).mkdir(exist_ok=True)
    logger.info("Directories checked/created")

# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="MindCare AI Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--init", action="store_true", help="Initialize data and models")
    
    args = parser.parse_args()
    
    if args.init:
        logger.info("Initializing data and models...")
        from app.routers.ingest import run_ingest
        from app.routers.features import build_features
        from app.routers.model import train_model
        from app.routers.rag import build_index
        
        # Run initialization steps
        run_ingest()
        build_features()
        train_model()
        build_index()
        logger.info("Initialization complete")
        return
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.LOG_LEVEL.lower(),
    )

if __name__ == "__main__":
    main()