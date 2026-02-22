# Quick Start Guide - Smart Multi-Doc RAG v2.0

Get your production-grade RAG system running in 5 minutes!

## ⚡ Quick Setup (Windows)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
⏱️ Takes 2-3 minutes (downloads ML models on first run)

### 2. Start the Server
```bash
python main.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Try It Out!

#### Option A: Using the Web Interface
- Open browser → `http://localhost:8000/docs`
- Interactive Swagger UI for all endpoints

#### Option B: Using Command Line

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/upload" -F "file=@path/to/your/document.pdf"
```

**Query the documents:**
```bash
curl -X POST "http://localhost:8000/query?query_text=What%20is%20main%20topic"
```

**View documents:**
```bash
curl "http://localhost:8000/documents"
```

#### Option C: Using Python
```python
import requests

# Upload
files = {"file": open("guide.pdf", "rb")}
r = requests.post("http://localhost:8000/upload", files=files)
print(r.json())

# Query
r = requests.post(
    "http://localhost:8000/query",
    params={"query_text": "Your question here"}
)
print(r.json()["context"])
```

## 📁 Supported File Formats

| Format | Status | Notes |
|--------|--------|-------|
| **PDF** | ✅ Supported | Text-based PDFs (not scanned) |
| **DOCX** | ✅ Supported | Word documents with tables |
| **TXT** | ✅ Supported | Plain text files |
| **CSV** | 🔄 Coming | Phase 3 |
| **Markdown** | 🔄 Coming | Phase 3 |

## 🎯 Key Endpoints

### 1. Upload (`POST /upload`)
Upload any PDF, DOCX, or TXT file
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@myfile.pdf"
```

### 2. Query (`POST /query`)
Ask a question about your documents
```bash
curl -X POST "http://localhost:8000/query" \
  -d "query_text=How does it work?"
```

### 3. List Documents (`GET /documents`)
See all uploaded documents
```bash
curl "http://localhost:8000/documents"
```

### 4. Delete Document (`DELETE /documents/{id}`)
Remove a document by ID
```bash
curl -X DELETE "http://localhost:8000/documents/1"
```

### 5. Statistics (`GET /stats`)
View system metrics
```bash
curl "http://localhost:8000/stats"
```

## 🚀 What's Working Now (Phase 1 & 2)

✅ PDF extraction and chunking
✅ DOCX document parsing
✅ Multi-document querying
✅ Intent-aware retrieval
✅ Hybrid search (semantic + keyword)
✅ Cross-encoder reranking
✅ SQLite persistence
✅ REST API with full documentation
✅ Query analytics and logging
✅ Error handling and validation

## 📊 Example Query Flow

1. **Upload documents**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@machine_learning_guide.pdf"
```

2. **Query the system**
```bash
curl -X POST "http://localhost:8000/query" \
  -d "query_text=Explain neural networks"
```

3. **Get intelligent context**
```json
{
  "status": "success",
  "query": "Explain neural networks",
  "intent": "definition",
  "context": "[machine_learning_guide]\nNeural networks are...",
  "retrieved_chunks": 6,
  "response_time_ms": 234.5
}
```

## 🔧 Configuration

Most settings in `.env` work out of the box. Common tweaks:

**For better accuracy:**
- Keep defaults (CHUNK_WORDS=140)

**For faster responses on large collections:**
- Reduce `TOP_K_VECTOR` from 80 to 40
- Reduce `TOP_K_FINAL_CHUNKS` from 6 to 3

**For handling large files:**
- Increase `MAX_FILE_SIZE` in `.env`
- Default is 50MB

## 📁 Project Layout

```
SmartMultiDocRAG/
├── main.py              ← Start here!
├── config.py            ← Configuration
├── database.py          ← Data persistence
├── document_processor.py ← PDF/DOCX extraction
├── rag_core.py         ← Retrieval logic
├── requirements.txt     ← Dependencies
├── .env                ← Environment settings
├── README.md           ← Full documentation
└── data/               ← Auto-created
    ├── documents/      ← Uploaded files
    ├── vector_db/      ← Embeddings
    └── rag_system.db  ← Database
```

## 🐛 Troubleshooting

**Can't start server?**
```bash
# Make sure port 8000 is available
# Or change PORT in .env to 8080, etc.
```

**No documents found?**
```bash
# Check what's uploaded
curl "http://localhost:8000/documents"

# Upload a test file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test.txt"
```

**Models downloading slowly?**
- First run downloads models (~500MB)
- Models cached in `~/.cache/huggingface/`
- Allow 1-2 minutes on first startup

**Query returns empty context?**
- Ensure documents are processed: `GET /documents`
- Try simpler, more specific queries
- Check `retrieved_chunks > 0` in response

## 📚 Next Steps

1. **Upload your documents** via `/docs` interface
2. **Try different queries** to test retrieval
3. **Check statistics** at `/stats`
4. **View API docs** at `/docs` for more details

## 📖 Full Documentation

See `README.md` for:
- Detailed API documentation
- Architecture overview
- Configuration reference
- Advanced usage examples
- Performance tuning guide

## 🤔 Common Questions

**Q: How many documents can it handle?**
A: Tested up to 100+ documents. Performance depends on document size and system RAM.

**Q: Can I use it in production?**
A: Yes! Phase 1 & 2 are production-ready. See Phase 3 for enterprise features (auth, monitoring).

**Q: How accurate is the retrieval?**
A: Varies by query quality. Intent-aware ranking helps. Rerank more documents for complex queries.

**Q: Can I use my own embedding model?**
A: Yes, edit `EMBEDDING_MODEL` in `.env`. Any HuggingFace sentence transformer works.

---

**Ready to go?** Start with:
```bash
python main.py
# Then visit http://localhost:8000/docs
```

Happy querying! 🚀
