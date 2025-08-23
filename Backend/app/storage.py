# path: mindcare-backend/app/storage.py
import logging
import os
from pathlib import Path

import pandas as pd
from app import config

logger = logging.getLogger(__name__)

# Try to import DuckDB, fallback to SQLite
try:
    import duckdb
    DUCKDB_AVAILABLE = True
    logger.info("Using DuckDB for storage")
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available, falling back to SQLite")
    try:
        import sqlite3
        SQLITE_AVAILABLE = True
        logger.info("Using SQLite for storage")
    except ImportError:
        logger.error("Neither DuckDB nor SQLite is available")
        raise RuntimeError("No compatible database library available")

class Storage:
    def __init__(self):
        self.db_path = config.DB_PATH
        self.conn = None
        
        # Ensure the models directory exists
        Path(config.MODELS_DIR).mkdir(exist_ok=True)
        
        # Initialize the database
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database with required tables"""
        if DUCKDB_AVAILABLE:
            self.conn = duckdb.connect(self.db_path)
        else:
            self.conn = sqlite3.connect(self.db_path)
            # Enable SQLite foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_connection(self):
        """Get a database connection"""
        if self.conn is None:
            self._initialize_db()
        return self.conn
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return the results"""
        conn = self.get_connection()
        if DUCKDB_AVAILABLE:
            if params:
                return conn.execute(query, params).fetchall()
            return conn.execute(query).fetchall()
        else:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
    
    def execute_update(self, query, params=None):
        """Execute a SQL update/insert/delete query"""
        conn = self.get_connection()
        if DUCKDB_AVAILABLE:
            if params:
                conn.execute(query, params)
            else:
                conn.execute(query)
            conn.commit()
        else:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            cursor.close()
    
    def create_table_from_csv(self, table_name, csv_path, primary_key=None):
        """Create a table from a CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Create the table
        if DUCKDB_AVAILABLE:
            # DuckDB can directly create a table from a CSV
            self.execute_update(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
        else:
            # SQLite requires creating the table first
            columns = []
            for col, dtype in df.dtypes.items():
                if dtype == 'object':
                    columns.append(f"{col} TEXT")
                elif dtype == 'int64':
                    columns.append(f"{col} INTEGER")
                elif dtype == 'float64':
                    columns.append(f"{col} REAL")
                else:
                    columns.append(f"{col} TEXT")
            
            create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            if primary_key:
                create_query += f", PRIMARY KEY ({primary_key})"
            
            self.execute_update(create_query)
            
            # Insert the data
            for _, row in df.iterrows():
                placeholders = ', '.join(['?'] * len(row))
                insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                self.execute_update(insert_query, tuple(row))
        
        logger.info(f"Table {table_name} created from {csv_path}")
    
    def table_exists(self, table_name):
        """Check if a table exists"""
        if DUCKDB_AVAILABLE:
            result = self.execute_query(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'")
            return result[0][0] > 0
        else:
            result = self.execute_query(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            return result[0][0] > 0
    
    def get_table_schema(self, table_name):
        """Get the schema of a table"""
        if DUCKDB_AVAILABLE:
            return self.execute_query(f"DESCRIBE {table_name}")
        else:
            return self.execute_query(f"PRAGMA table_info({table_name})")
    
    def read_table(self, table_name):
        """Read a table into a pandas DataFrame"""
        conn = self.get_connection()
        if DUCKDB_AVAILABLE:
            return conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        else:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Global storage instance
storage = Storage()