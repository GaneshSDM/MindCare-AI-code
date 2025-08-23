# path: mindcare-backend/app/rag/indexer.py
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from app import config
from app.utils.timers import timed

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, fallback to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence transformers available for embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available, falling back to TF-IDF")

# Try to import PyPDF2
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
    logger.info("PyPDF2 available for PDF processing")
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.error("PyPDF2 not available, cannot process PDFs")

# Try to import FAISS and ChromaDB
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available for vector storage")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
    logger.info("ChromaDB available for vector storage")
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    if not PYPDF2_AVAILABLE:
        logger.error("PyPDF2 not available for PDF extraction")
        return ""
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        
        logger.info(f"Extracted text from PDF: {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum size of each chunk in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        List of text chunks
    """
    # Simple word-based chunking
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

class DocumentIndexer:
    """Document indexer for RAG"""
    
    def __init__(self):
        """Initialize the document indexer"""
        self.embedding_model = None
        self.vectorizer = None
        self.vector_store = None
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        
        # Initialize embedding model or vectorizer
        self._init_embedding()
        
        # Initialize vector store
        self._init_vector_store()
    
    def _init_embedding(self):
        """Initialize embedding model or vectorizer"""
        if SENTENCE_TRANSFORMERS_AVAILABLE and config.EMBED_MODEL_PATH:
            try:
                # Try to load from local path
                self.embedding_model = SentenceTransformer(config.EMBED_MODEL_PATH)
                logger.info(f"Loaded sentence transformer from {config.EMBED_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading sentence transformer: {str(e)}")
                self.embedding_model = None
        
        # Fall back to TF-IDF if sentence transformers not available
        if self.embedding_model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=1000)
            logger.info("Using TF-IDF vectorizer for embeddings")
    
    def _init_vector_store(self):
        """Initialize vector store"""
        vector_db = config.VECTOR_DB.lower()
        
        if vector_db == "faiss" and FAISS_AVAILABLE:
            self.vector_store = FAISSVectorStore()
            logger.info("Using FAISS for vector storage")
        elif vector_db == "chroma" and CHROMADB_AVAILABLE:
            self.vector_store = ChromaVectorStore()
            logger.info("Using ChromaDB for vector storage")
        else:
            logger.warning(f"Vector store {vector_db} not available, using in-memory storage")
            self.vector_store = InMemoryVectorStore()
    
    def index_documents(self, documents_dir: str) -> bool:
        """
        Index documents from a directory.
        
        Args:
            documents_dir: Path to directory containing documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing data
            self.documents = []
            self.chunks = []
            self.chunk_metadata = []
            
            # Get all PDF files in the directory
            pdf_files = list(Path(documents_dir).glob("*.pdf"))
            
            if len(pdf_files) == 0:
                logger.warning(f"No PDF files found in {documents_dir}")
                return False
            
            # Process each PDF
            for pdf_path in pdf_files:
                # Extract text
                text = extract_text_from_pdf(str(pdf_path))
                
                if not text:
                    logger.warning(f"No text extracted from {pdf_path}")
                    continue
                
                # Store document
                doc_id = len(self.documents)
                self.documents.append({
                    "id": doc_id,
                    "source": str(pdf_path),
                    "title": pdf_path.stem
                })
                
                # Chunk text
                doc_chunks = chunk_text(text)
                
                # Store chunks with metadata
                for i, chunk in enumerate(doc_chunks):
                    chunk_id = len(self.chunks)
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "source": str(pdf_path),
                        "title": pdf_path.stem
                    })
            
            logger.info(f"Processed {len(self.documents)} documents into {len(self.chunks)} chunks")
            
            # Create embeddings
            embeddings = self._create_embeddings(self.chunks)
            
            # Add to vector store
            self.vector_store.add_embeddings(embeddings, self.chunk_metadata)
            
            logger.info("Document indexing completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if self.embedding_model:
            # Use sentence transformers
            embeddings = self.embedding_model.encode(texts)
            logger.info(f"Created embeddings using sentence transformer: {embeddings.shape}")
            return embeddings
        else:
            # Use TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            embeddings = tfidf_matrix.toarray()
            logger.info(f"Created embeddings using TF-IDF: {embeddings.shape}")
            return embeddings
    
    def save_index(self, index_path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(index_path).mkdir(exist_ok=True)
            
            # Save documents and chunks
            import pickle
            with open(os.path.join(index_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)
            
            with open(os.path.join(index_path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.chunk_metadata, f)
            
            # Save vectorizer if using TF-IDF
            if self.vectorizer:
                import joblib
                joblib.dump(self.vectorizer, os.path.join(index_path, "vectorizer.pkl"))
            
            # Save vector store
            self.vector_store.save(index_path)
            
            logger.info(f"Index saved to {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, index_path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            index_path: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load documents and chunks
            import pickle
            with open(os.path.join(index_path, "documents.pkl"), "rb") as f:
                self.documents = pickle.load(f)
            
            with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
                self.chunks = pickle.load(f)
            
            with open(os.path.join(index_path, "metadata.pkl"), "rb") as f:
                self.chunk_metadata = pickle.load(f)
            
            # Load vectorizer if using TF-IDF
            if os.path.exists(os.path.join(index_path, "vectorizer.pkl")):
                import joblib
                self.vectorizer = joblib.load(os.path.join(index_path, "vectorizer.pkl"))
            
            # Load vector store
            self.vector_store.load(index_path)
            
            logger.info(f"Index loaded from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False

class FAISSVectorStore:
    """FAISS vector store"""
    
    def __init__(self):
        """Initialize the FAISS vector store"""
        self.index = None
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dictionaries
        """
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype(np.float32))
            self.metadata = metadata
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[dict], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (metadata, scores)
        """
        if self.index is None:
            logger.error("FAISS index not initialized")
            return [], []
        
        try:
            # Reshape query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Get metadata and scores
            results = []
            scores = []
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    results.append(self.metadata[idx])
                    scores.append(float(distances[0][i]))
            
            return results, scores
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            return [], []
    
    def save(self, index_path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save the vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.index is None:
                logger.warning("No FAISS index to save")
                return False
            
            # Save index
            faiss.write_index(self.index, os.path.join(index_path, "faiss.index"))
            
            # Save metadata
            import pickle
            with open(os.path.join(index_path, "faiss_metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"FAISS index saved to {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            return False
    
    def load(self, index_path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to load the vector store from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load index
            index_file = os.path.join(index_path, "faiss.index")
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                
                # Load metadata
                metadata_file = os.path.join(index_path, "faiss_metadata.pkl")
                if os.path.exists(metadata_file):
                    import pickle
                    with open(metadata_file, "rb") as f:
                        self.metadata = pickle.load(f)
                
                logger.info(f"FAISS index loaded from {index_path}")
                return True
            else:
                logger.warning(f"FAISS index file not found: {index_file}")
                return False
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False

class ChromaVectorStore:
    """ChromaDB vector store"""
    
    def __init__(self):
        """Initialize the ChromaDB vector store"""
        self.client = None
        self.collection = None
        self.collection_name = "mindcare_policies"
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dictionaries
        """
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=config.VECTOR_INDEX_DIR)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            
            # Prepare data for ChromaDB
            ids = [str(m["chunk_id"]) for m in metadata]
            documents = [m.get("text", "") for m in metadata]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadata
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding embeddings to ChromaDB: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[dict], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (metadata, scores)
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return [], []
        
        try:
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            # Extract metadata and distances
            metadata = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            # Convert distances to scores (lower distance = higher score)
            scores = [1.0 / (1.0 + d) for d in distances] if distances else []
            
            return metadata, scores
        except Exception as e:
            logger.error(f"Error searching ChromaDB collection: {str(e)}")
            return [], []
    
    def save(self, index_path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save the vector store
            
        Returns:
            True if successful, False otherwise
        """
        # ChromaDB saves automatically to the persistent path
        logger.info("ChromaDB collection is persisted automatically")
        return True
    
    def load(self, index_path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to load the vector store from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=config.VECTOR_INDEX_DIR)
            
            # Get collection
            self.collection = self.client.get_collection(name=self.collection_name)
            
            logger.info(f"ChromaDB collection loaded from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading ChromaDB collection: {str(e)}")
            return False

class InMemoryVectorStore:
    """In-memory vector store for fallback"""
    
    def __init__(self):
        """Initialize the in-memory vector store"""
        self.embeddings = None
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dictionaries
        """
        self.embeddings = embeddings
        self.metadata = metadata
        logger.info(f"Added {len(embeddings)} embeddings to in-memory vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[dict], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (metadata, scores)
        """
        if self.embeddings is None:
            logger.error("In-memory vector store not initialized")
            return [], []
        
        try:
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                self.embeddings
            ).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            # Get metadata and scores
            results = [self.metadata[i] for i in top_indices]
            scores = [float(similarities[i]) for i in top_indices]
            
            return results, scores
        except Exception as e:
            logger.error(f"Error searching in-memory vector store: {str(e)}")
            return [], []
    
    def save(self, index_path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save the vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(index_path).mkdir(exist_ok=True)
            
            # Save embeddings and metadata
            import pickle
            with open(os.path.join(index_path, "in_memory_embeddings.pkl"), "wb") as f:
                pickle.dump(self.embeddings, f)
            
            with open(os.path.join(index_path, "in_memory_metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"In-memory vector store saved to {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving in-memory vector store: {str(e)}")
            return False
    
    def load(self, index_path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to load the vector store from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load embeddings and metadata
            import pickle
            with open(os.path.join(index_path, "in_memory_embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
            
            with open(os.path.join(index_path, "in_memory_metadata.pkl"), "rb") as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"In-memory vector store loaded from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading in-memory vector store: {str(e)}")
            return False

@timed
def build_index(documents_dir: str = None) -> bool:
    """
    Build document index.
    
    Args:
        documents_dir: Path to directory containing documents
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if documents_dir is None:
            documents_dir = os.path.join(config.DATA_DIR, "policies")
        
        # Initialize indexer
        indexer = DocumentIndexer()
        
        # Index documents
        success = indexer.index_documents(documents_dir)
        
        if not success:
            logger.error("Failed to index documents")
            return False
        
        # Save index
        indexer.save_index(config.VECTOR_INDEX_DIR)
        
        logger.info("Document index built successfully")
        return True
    except Exception as e:
        logger.error(f"Error building document index: {str(e)}")
        return False

@timed
def load_index() -> DocumentIndexer:
    """
    Load document index.
    
    Returns:
        DocumentIndexer instance
    """
    try:
        # Initialize indexer
        indexer = DocumentIndexer()
        
        # Load index
        success = indexer.load_index(config.VECTOR_INDEX_DIR)
        
        if not success:
            logger.warning("Failed to load existing index, returning empty indexer")
        
        return indexer
    except Exception as e:
        logger.error(f"Error loading document index: {str(e)}")
        return DocumentIndexer()