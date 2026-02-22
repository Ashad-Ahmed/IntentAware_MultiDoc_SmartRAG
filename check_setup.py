#!/usr/bin/env python
"""
Test script to verify Smart RAG system setup
Run this before starting the server to check all dependencies
"""

import sys
from pathlib import Path


def check_python_version():
    """Verify Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"[FAIL] Python 3.8+ required, found {version.major}.{version.minor}")
        return False


def check_imports():
    """Check all required imports"""
    packages = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
        "sqlalchemy": "SQLAlchemy",
        "sentence_transformers": "Sentence Transformers",
        "rank_bm25": "BM25",
        "faiss": "FAISS",
        "pdfplumber": "PDF Plumber",
        "docx": "python-docx",
    }

    failed = []
    for package, display_name in packages.items():
        try:
            __import__(package)
            print(f"[OK] {display_name}")
        except ImportError:
            print(f"[FAIL] {display_name} (run: pip install -r requirements.txt)")
            failed.append(package)

    return len(failed) == 0


def check_directories():
    """Verify directory structure"""
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "data",
        base_dir / "logs",
    ]

    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"[OK] {dir_path.name}/ directory exists")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Created {dir_path.name}/ directory")

    return all_ok


def check_config():
    """Test config loading"""
    try:
        from config import settings

        print(f"[OK] Config loaded")
        print(f"    - Chunk size: {settings.CHUNK_WORDS} words")
        print(f"    - Max file size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f} MB")
        print(f"    - Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}")
        return True
    except Exception as e:
        print(f"[FAIL] Config error: {e}")
        return False


def check_database():
    """Test database setup"""
    try:
        from database import init_db, engine, Base

        init_db()
        print("[OK] Database initialized")
        return True
    except Exception as e:
        print(f"[FAIL] Database error: {e}")
        return False


def check_modules():
    """Test core modules"""
    modules = {
        "document_processor": "DocumentProcessor",
        "rag_core": "RAGCore",
    }

    all_ok = True
    for module, class_name in modules.items():
        try:
            mod = __import__(module)
            print(f"[OK] {class_name} module loaded")
        except Exception as e:
            print(f"[FAIL] {class_name} error: {e}")
            all_ok = False

    return all_ok


def main():
    """Run all checks"""
    print("=" * 60)
    print("Smart Multi-Doc RAG - System Check")
    print("=" * 60)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("Directory Structure", check_directories),
        ("Configuration", check_config),
        ("Database", check_database),
        ("Core Modules", check_modules),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\nChecking: {check_name}")
        print("-" * 40)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"[ERROR] {e}")
            results.append((check_name, False))

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        all_pass = all_pass and result
        print(f"[{status}] {check_name}")

    print()
    if all_pass:
        print("All checks passed! Ready to start the server.")
        print()
        print("Run this command to start:")
        print("  python main.py")
        print()
        print("Then visit: http://localhost:8000/docs")
        return 0
    else:
        print("Some checks failed. Please fix the issues above.")
        print()
        print("Install dependencies with:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
