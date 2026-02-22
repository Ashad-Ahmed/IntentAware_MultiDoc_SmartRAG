import logging
import time
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from config import settings
from database import init_db, get_db, SessionLocal, Document, Chunk, QueryLog
from document_processor import DocumentProcessor
from rag_core import rag_system

# Setup logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===================== STARTUP / SHUTDOWN =====================


def rebuild_indices(db: Session):
    """Rebuild RAG indices from database on startup"""
    logger.info("Rebuilding RAG indices from database...")

    chunks = []
    metadatas = []

    # Load all chunks from database
    db_chunks = db.query(Chunk).all()

    logger.info(f"Loading {len(db_chunks)} chunks from database...")

    for chunk in db_chunks:
        chunks.append(chunk.text)
        doc = db.query(Document).filter(Document.id == chunk.document_id).first()

        metadatas.append(
            {
                "DocName": doc.filename if doc else "Unknown",
                "StartWord": chunk.start_word,
                "ChunkIndex": chunk.chunk_index,
                "PageNumber": chunk.page_number,
            }
        )

    if chunks:
        rag_system.build_indices(chunks)
        rag_system.build_doc_graph(metadatas)
        logger.info(f"RAG indices rebuilt with {len(chunks)} chunks")
    else:
        logger.info("No chunks found in database")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    # Startup
    logger.info("Starting Smart Multi-Doc RAG Server...")
    init_db()
    db = SessionLocal()
    try:
        rebuild_indices(db)
    finally:
        db.close()
    logger.info("Server ready to receive requests")

    yield

    # Shutdown
    logger.info("Shutting down server...")


# ===================== FASTAPI APP =====================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


# ===================== UTILITIES =====================


def get_document_type(filename: str) -> str:
    """Extract file type from filename"""
    return filename.split(".")[-1].lower()


# ===================== ENDPOINTS =====================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a document (txt, pdf, or docx)

    Args:
        file: Document file to upload
        db: Database session

    Returns:
        Document info and processing status
    """
    try:
        # Get file type and validate extension
        file_type = get_document_type(file.filename)

        if file_type not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # Save file to disk
        file_path = settings.DOCUMENTS_DIR / file.filename

        with open(file_path, "wb") as f:
            f.write(content)

        # Check if document already exists
        existing = db.query(Document).filter(Document.filename == file.filename).first()
        if existing:
            logger.info(f"Updating existing document: {file.filename}")
            doc = existing
            # Delete old chunks
            db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        else:
            doc = Document(
                filename=file.filename,
                file_type=file_type,
                file_size=file_size,
            )
            db.add(doc)
            db.flush()

        # Extract text
        logger.info(f"Extracting text from {file_type} file: {file.filename}")
        text = DocumentProcessor.extract_text(str(file_path), file_type)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no extractable text")

        # Chunk text
        logger.info(f"Chunking text...")
        chunks = DocumentProcessor.chunk_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create chunks from document")

        # Store chunks in database
        logger.info(f"Storing {len(chunks)} chunks in database...")
        for chunk_idx, (start_word, chunk_text) in enumerate(chunks):
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=chunk_idx,
                text=chunk_text,
                start_word=start_word,
            )
            db.add(chunk)

        doc.total_chunks = len(chunks)
        doc.processed = True
        doc.error_message = None

        db.commit()

        # Rebuild indices
        logger.info("Rebuilding RAG indices...")
        rebuild_indices(db)

        logger.info(f"Document uploaded and indexed successfully: {file.filename}")

        return {
            "status": "success",
            "filename": file.filename,
            "file_type": file_type,
            "file_size": file_size,
            "chunks": len(chunks),
            "message": f"Document processed successfully with {len(chunks)} chunks",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/query")
async def query(
    query_text: str,
    db: Session = Depends(get_db),
):
    """
    Query the RAG system

    Args:
        query_text: Query text
        db: Database session

    Returns:
        Retrieved context and metadata
    """
    try:
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not rag_system.chunks:
            raise HTTPException(status_code=400, detail="No documents uploaded yet")

        start_time = time.time()

        # Run retrieval
        context, intent, num_chunks = rag_system.retrieve(query_text)

        response_time = (time.time() - start_time) * 1000  # Convert to ms

        # Log query
        query_log = QueryLog(
            query_text=query_text,
            intent=intent,
            retrieved_chunks=num_chunks,
            response_time_ms=response_time,
        )
        db.add(query_log)
        db.commit()

        logger.info(
            f"Query processed - Intent: {intent}, Chunks: {num_chunks}, Time: {response_time:.1f}ms"
        )

        return {
            "status": "success",
            "query": query_text,
            "intent": intent,
            "context": context,
            "retrieved_chunks": num_chunks,
            "response_time_ms": round(response_time, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """
    List all uploaded documents

    Returns:
        List of documents with metadata
    """
    try:
        docs = db.query(Document).all()

        documents = [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "total_chunks": doc.total_chunks,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "processed": doc.processed,
            }
            for doc in docs
        ]

        return {
            "status": "success",
            "count": len(documents),
            "documents": documents,
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and its chunks

    Args:
        doc_id: Document ID
        db: Database session

    Returns:
        Deletion status
    """
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete chunks
        db.query(Chunk).filter(Chunk.document_id == doc_id).delete()

        # Delete document
        db.delete(doc)
        db.commit()

        # Rebuild indices
        rebuild_indices(db)

        logger.info(f"Document deleted: {doc.filename}")

        return {
            "status": "success",
            "message": f"Document '{doc.filename}' deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Get system statistics

    Returns:
        System stats and metrics
    """
    try:
        total_docs = db.query(Document).count()
        total_chunks = db.query(Chunk).count()
        total_queries = db.query(QueryLog).count()

        # Get average response time
        avg_response_time = 0
        if total_queries > 0:
            from sqlalchemy import func

            result = db.query(func.avg(QueryLog.response_time_ms)).first()
            avg_response_time = result[0] or 0

        return {
            "status": "success",
            "statistics": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_queries": total_queries,
                "chunks_in_memory": len(rag_system.chunks),
                "average_response_time_ms": round(avg_response_time, 2),
                "config": {
                    "chunk_size": settings.CHUNK_WORDS,
                    "overlap": settings.CHUNK_OVERLAP,
                    "max_final_chunks": settings.TOP_K_FINAL_CHUNKS,
                },
            },
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /upload (file upload form)",
            "query": "POST /query (query_text parameter)",
            "documents": "GET /documents",
            "delete_document": "DELETE /documents/{doc_id}",
            "stats": "GET /stats",
            "docs": "GET /docs (Swagger UI)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
