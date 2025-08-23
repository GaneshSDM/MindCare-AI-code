# path: mindcare-backend/app/rag/qa.py
import json
import logging
import os
import subprocess
from typing import Dict, List, Tuple

from app import config
from app.rag.indexer import DocumentIndexer, load_index
from app.utils.timers import timed

logger = logging.getLogger(__name__)

class QASystem:
    """Question-Answering system using RAG and Ollama"""
    
    def __init__(self):
        """Initialize the QA system"""
        self.indexer = load_index()
        self.model_name = config.OLLAMA_MODEL
        self.ollama_host = config.OLLAMA_HOST
    
    def _create_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        if self.indexer.embedding_model:
            # Use sentence transformers
            import numpy as np
            embedding = self.indexer.embedding_model.encode(query)
            return embedding.tolist()
        else:
            # Use TF-IDF
            import numpy as np
            query_tfidf = self.indexer.vectorizer.transform([query])
            embedding = query_tfidf.toarray()[0]
            return embedding.tolist()
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 4) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (chunk metadata, scores)
        """
        # Create query embedding
        query_embedding = self._create_query_embedding(query)
        
        # Search vector store
        return self.indexer.vector_store.search(query_embedding, k=top_k)
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context for the model.
        
        Args:
            chunks: List of chunk metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", i)
            chunk_text = self.indexer.chunks[chunk_id] if chunk_id < len(self.indexer.chunks) else ""
            source = chunk.get("source", "Unknown")
            title = chunk.get("title", "Unknown")
            
            context_part = f"""
Document [{i+1}]
Title: {title}
Source: {source}
Content: {chunk_text}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the model.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""
You are an HR policy assistant for MindCare AI. Answer the user's question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question based only on the provided context.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Cite the source documents using [1], [2], etc. notation.
4. Be concise and direct in your answer.

Answer:
"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama model with a prompt.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Model response
        """
        try:
            # Prepare command
            cmd = ["ollama", "run", self.model_name]
            
            # Run command
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=60  # Timeout after 60 seconds
            )
            
            if result.returncode != 0:
                logger.error(f"Ollama error: {result.stderr}")
                return f"Error: Failed to get response from Ollama. {result.stderr}"
            
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error("Ollama request timed out")
            return "Error: Request to Ollama timed out."
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return f"Error: {str(e)}"
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """
        Extract citations from the answer.
        
        Args:
            answer: Model answer
            chunks: Retrieved chunks
            
        Returns:
            List of citations
        """
        citations = []
        
        # Find all citation references in the answer
        import re
        citation_pattern = r'\[(\d+)\]'
        citation_matches = re.findall(citation_pattern, answer)
        
        # Create citation objects
        for match in set(citation_matches):
            try:
                idx = int(match) - 1  # Convert to 0-based index
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    chunk_id = chunk.get("chunk_id", idx)
                    chunk_text = self.indexer.chunks[chunk_id] if chunk_id < len(self.indexer.chunks) else ""
                    
                    # Create preview (first 100 characters)
                    preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                    
                    citations.append({
                        "id": int(match),
                        "source": chunk.get("source", "Unknown"),
                        "preview": preview
                    })
            except (ValueError, IndexError):
                continue
        
        return citations
    
    @timed
    def answer_question(self, query: str, top_k: int = 4) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and citations
        """
        try:
            # Retrieve relevant chunks
            chunks, scores = self._retrieve_relevant_chunks(query, top_k)
            
            if not chunks:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "citations": [],
                    "model": self.model_name,
                    "latency_ms": 0
                }
            
            # Format context
            context = self._format_context(chunks)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Call model
            answer = self._call_ollama(prompt)
            
            # Extract citations
            citations = self._extract_citations(answer, chunks)
            
            return {
                "answer": answer,
                "citations": citations,
                "model": self.model_name,
                "latency_ms": 0  # Will be set by the decorator
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "citations": [],
                "model": self.model_name,
                "latency_ms": 0
            }

@timed
def answer_question(query: str, top_k: int = 4) -> Dict:
    """
    Answer a question using RAG.
    
    Args:
        query: User query
        top_k: Number of chunks to retrieve
        
    Returns:
        Dictionary with answer and citations
    """
    qa_system = QASystem()
    return qa_system.answer_question(query, top_k)