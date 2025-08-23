# path: mindcare-backend/app/routers/rag.py
import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.indexer import build_index, load_index
from app.rag.qa import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()

class Citation(BaseModel):
    id: int
    source: str
    preview: str

class AskRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    top_k: int = Field(default=4, description="Number of documents to retrieve")

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model: str
    latency_ms: float

class IndexResponse(BaseModel):
    success: bool
    message: str
    documents_indexed: int = 0

@router.post("/rag/index", response_model=IndexResponse)
async def build_rag_index():
    """
    Build or rebuild the RAG index from policy documents.
    
    Returns:
        IndexResponse with results
    """
    try:
        # Build index
        success = build_index()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to build index")
        
        # Load index to get document count
        indexer = load_index()
        documents_indexed = len(indexer.documents)
        
        return IndexResponse(
            success=True,
            message=f"Successfully indexed {documents_indexed} documents",
            documents_indexed=documents_indexed
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building RAG index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building RAG index: {str(e)}")

@router.post("/rag/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question using the RAG system.
    
    Args:
        request: AskRequest with question and top_k
        
    Returns:
        AskResponse with answer and citations
    """
    try:
        # Answer question
        result = answer_question(request.question, request.top_k)
        
        # Format citations
        citations = [
            Citation(
                id=citation['id'],
                source=citation['source'],
                preview=citation['preview']
            )
            for citation in result['citations']
        ]
        
        return AskResponse(
            answer=result['answer'],
            citations=citations,
            model=result['model'],
            latency_ms=result['latency_ms']
        )
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")