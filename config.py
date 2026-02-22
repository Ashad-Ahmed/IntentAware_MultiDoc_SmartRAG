import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file"""

    # App Configuration
    APP_NAME: str = "Smart Multi-Doc RAG"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    DB_PATH: Path = DATA_DIR / "rag_system.db"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # RAG Core Configuration
    CHUNK_WORDS: int = 140
    CHUNK_OVERLAP: int = 40

    TOP_K_VECTOR: int = 80
    TOP_K_BM25: int = 80
    TOP_K_FUSED: int = 150
    TOP_K_FINAL_CHUNKS: int = 6

    RRF_K: int = 60

    # Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # File Upload Configuration
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: list[str] = ["txt", "pdf", "docx"]

    # Intent-based configuration
    INTENT_CONFIG: dict = {
        "definition": {"rerank_depth": 30, "doc_limit": 2, "diversity": False},
        "reasoning": {"rerank_depth": 60, "doc_limit": 3, "diversity": False},
        "comparison": {"rerank_depth": 70, "doc_limit": 4, "diversity": True},
        "procedural": {"rerank_depth": 50, "doc_limit": 3, "diversity": False},
        "troubleshooting": {"rerank_depth": 80, "doc_limit": 4, "diversity": False},
        "general": {"rerank_depth": 50, "doc_limit": 3, "diversity": False},
    }

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./rag_system.db"

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "rag_system.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
