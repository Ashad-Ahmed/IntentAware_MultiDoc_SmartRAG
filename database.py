from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings

Base = declarative_base()


class Document(Base):
    """Document metadata storage"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    file_type = Column(String(20), nullable=False)  # txt, pdf, docx
    file_size = Column(Integer, nullable=False)  # bytes
    total_chunks = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, type={self.file_type})>"


class Chunk(Base):
    """Document chunks with metadata"""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    start_word = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)  # For PDFs
    section = Column(String(255), nullable=True)  # For structured documents
    embedding_id = Column(String(36), nullable=True)  # Chroma UUID

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, idx={self.chunk_index})>"


class QueryLog(Base):
    """Query history and analytics"""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    intent = Column(String(50), nullable=False)
    retrieved_chunks = Column(Integer, default=0)
    response_time_ms = Column(Float, default=0.0)
    executed_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<QueryLog(id={self.id}, intent={self.intent}, time={self.response_time_ms}ms)>"


# Database initialization
engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
