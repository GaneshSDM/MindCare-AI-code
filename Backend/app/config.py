# path: mindcare-backend/app/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_BACKEND = os.getenv("DB_BACKEND", "duckdb")
DB_PATH = os.getenv("DB_PATH", "./models/mindcare.duckdb" if DB_BACKEND == "duckdb" else "./models/mindcare.sqlite")

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Embedding model configuration
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "")
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")

# PII masking
PII_SALT = os.getenv("PII_SALT", "mindcare_salt_2023")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Paths
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
VECTOR_INDEX_DIR = os.getenv("VECTOR_INDEX_DIR", "./models/vector_index")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./models/artifacts")