# Smart Multi-Document RAG System v2.0

A production-ready Retrieval Augmented Generation (RAG) system that processes multiple documents (TXT, PDF, DOCX) and answers queries using intelligent retrieval and ranking.

## Features

### Core Intelligence
- **Intent-Aware Retrieval**: Automatically detects query type (definition, reasoning, comparison, procedural, troubleshooting) and adjusts retrieval strategy
- **Hybrid Search**: Combines semantic (vector) and lexical (BM25) retrieval using Reciprocal Rank Fusion
- **Multi-Stage Ranking**:
  - Dense vector embeddings for semantic understanding
  - BM25 for keyword matching
  - Cross-encoder reranking for relevance
  - Graph-based context expansion for document continuity

### Multi-Format Support
- ✅ **Plain Text (.txt)**
- ✅ **PDF Files** - Extracts text with page tracking
- ✅ **Word Documents (.docx)** - Preserves structure and tables
- 🔄 Future: Excel, CSV, markdown, and more

### Production Ready
- **REST API** - FastAPI with automatic documentation
- **Persistent Storage** - SQLite database + Vector store
- **Error Handling** - Comprehensive validation and error messages
- **Logging** - Structured logging for debugging
- **Monitoring** - Query analytics and system statistics

## Architecture

```
┌─────────────────┐
│  User Uploads   │
│  (TXT/PDF/DOCX) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Document Processor          │
│ - Extract text              │
│ - Intelligent chunking      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ SQLite Database             │
│ - Document metadata         │
│ - Chunk storage             │
│ - Query history             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ RAG Core Services           │
│ - FAISS Vector Index        │
│ - BM25 Keyword Index        │
│ - Document Graph            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Retrieval Pipeline          │
│ - Intent Classification     │
│ - Vector Retrieval          │
│ - BM25 Retrieval            │
│ - RRF Fusion                │
│ - Cross-Encoder Reranking   │
│ - Graph Expansion           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│ Context Result  │
└─────────────────┘
```

## How It Works: Complete Pipeline Explanation

### Upload Pipeline (When You Upload a Document)

```
1. FILE VALIDATION
   ├─ Check file extension (txt, pdf, docx)
   ├─ Validate file size (max 50MB)
   └─ Verify file is not empty

2. TEXT EXTRACTION
   ├─ PDF: pdfplumber extracts text from each page
   │        (preserves page numbers for context)
   ├─ DOCX: python-docx extracts paragraphs & tables
   │        (preserves document structure)
   └─ TXT:  Direct file read (UTF-8 encoded)

3. TEXT CHUNKING
   ├─ Split text into 140-word chunks
   ├─ Add 40-word overlap between chunks
   │  (ensures context continuity)
   └─ Track word position in original document

4. DATABASE STORAGE
   ├─ Save document metadata to SQLite
   │  (filename, type, size, created date)
   └─ Save all chunks to SQLite
      (text, position, page number, document_id)

5. INDEX BUILDING
   ├─ Load all chunks from database
   ├─ FAISS Index (Vector Search)
   │  ├─ Encode chunks using sentence-transformer
   │  │  "hello world" → [0.23, -0.51, 0.88, ...]
   │  ├─ Normalize vectors for cosine similarity
   │  └─ Build FAISS index for O(1) similarity search
   │
   ├─ BM25 Index (Keyword Search)
   │  ├─ Tokenize all chunks
   │  └─ Build BM25 scoring for keyword matching
   │
   └─ Document Graph
      ├─ Connect chunks from same document
      └─ Enable context expansion (get nearby chunks)

RESULT: Document is indexed and searchable ✓
```

### Query Pipeline (When You Ask a Question)

```
1. INTENT CLASSIFICATION
   ├─ Analyze query keywords
   ├─ Detect type: definition, reasoning, comparison,
   │               procedural, troubleshooting, or general
   └─ Adjust retrieval parameters based on intent
      (e.g., "why" questions get deeper reranking)

2. SEMANTIC RETRIEVAL (Vector Search)
   ├─ Encode query using same sentence-transformer
   │  "Why is caching important?" → [0.15, 0.42, -0.33, ...]
   ├─ Search FAISS index for top 80 similar chunks
   │  (uses cosine similarity with normalized vectors)
   └─ Return: [(chunk_id_5, score_0.95), (chunk_id_12, score_0.87), ...]

3. LEXICAL RETRIEVAL (BM25 Search)
   ├─ Extract keywords from query
   │  ["why", "caching", "important"]
   ├─ Search BM25 index for matching chunks
   └─ Return: [(chunk_id_8, score_2.3), (chunk_id_15, score_1.8), ...]

4. FUSION (Combine Both Results)
   ├─ Use Reciprocal Rank Fusion (RRF)
   │  ├─ Vector result rank=1 → score = 1/(60+1) = 0.0164
   │  ├─ BM25 result rank=1 → score = 1/(60+1) = 0.0164
   │  └─ Sum scores for each chunk
   │
   ├─ Sort by combined score (semantic + keyword)
   └─ Return top 150 fused results

5. CROSS-ENCODER RERANKING
   ├─ Take top 150 (or rerank_depth based on intent)
   ├─ Create pairs: (query, chunk_text)
   │  [("Why is caching important?", "Caching stores..."),
   │   ("Why is caching important?", "Cache hit ratio..."), ...]
   │
   ├─ Score each pair with cross-encoder model
   │  (more accurate relevance than vector embeddings)
   ├─ Sort by relevance score (descending)
   └─ Return: [(chunk_id_5, score_0.95), (chunk_id_8, score_0.88), ...]

6. CONTEXT EXPANSION (Document Graph)
   ├─ Take top 6 reranked chunks
   ├─ For each chunk, check relevance score:
   │  ├─ If score > 0.8 → expand window by 2 chunks
   │  ├─ If score > 0.5 → expand window by 1 chunk
   │  └─ If score ≤ 0.5 → no expansion
   │
   ├─ Find neighboring chunks in document graph
   │  (chunks within expansion window distance)
   ├─ Add neighbors with slightly lower score
   │  (neighbor_score = parent_score × 0.95)
   └─ Result: Same document may contribute multiple chunks

7. OVERLAP SUPPRESSION
   ├─ Remove chunks that are too close together
   │  (within 40 words = chunk overlap size)
   ├─ Preserve chunks from different documents
   └─ Maintain relevance order

8. FINAL CONTEXT ASSEMBLY
   ├─ Take top 6 chunks (after filtering)
   ├─ Format with document names and relevance scores
   │
   │  [document_name] (relevance: 0.95)
   │  Chunk text here...
   │
   │  [document_name] (relevance: 0.88)
   │  Another chunk here...
   │
   └─ Return context + query metadata

RESULT: User gets most relevant chunks in order ✓
```

### Storage & Retrieval

```
Upload → SQLite Database (Persistent)
      │
      └─→ FAISS Index (In-Memory, Fast)
          BM25 Index (In-Memory, Fast)
          Document Graph (In-Memory, Fast)

Query → FAISS + BM25 → RRF Fusion → Rerank → Expand → Result

On Server Restart:
  └─→ Load chunks from SQLite → Rebuild FAISS/BM25/Graph (~30-60 sec)
```

### Data Flow Example

```
INPUT: PDF about "Machine Learning"
       (12,000 words, 85 chunks)

1. Extract: "A neural network is a computing system..."
2. Chunk: [chunk_0, chunk_1, ..., chunk_84]
3. Store: Document_1 with 85 rows in chunks table
4. Encode: 85 vectors (each 384-dimensional)
5. Index: One FAISS entry per chunk

QUERY: "What is a neural network?"
       ↓
1. Classify Intent: "definition" (contains "what is")
2. Vector Search: Find 80 similar chunks
   (query vector is closest to chunk_3, chunk_7, chunk_12...)
3. BM25 Search: Match keywords
   (chunks containing "neural" and "network")
4. RRF: Merge both rankings
5. Rerank: Cross-encoder scores all candidates
6. Expand: Add neighboring chunks for context
7. Filter: Remove overlaps, keep top 6
8. Return:
   [PDF_Name] (relevance: 0.95)
   "A neural network is a computing system..."

   [PDF_Name] (relevance: 0.88)
   "Neural networks learn by adjusting weights..."

LATENCY: ~200-400ms (depending on document count)
ACCURACY: Highest relevant chunks appear first
```

## Installation

### Requirements
- Python 3.8+
- pip package manager

### Setup

1. **Clone/Navigate to project directory**
   ```bash
   cd "c:\Ashad\ML Project\ExperimentAI\SmartMultiDocRAG"
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: First run will download ML models (~500MB for embeddings + reranking)

4. **Run the server**
   ```bash
   python main.py
   ```

   Server starts at: `http://localhost:8000`

## API Usage

### 1. Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 123456,
  "chunks": 42,
  "message": "Document processed successfully with 42 chunks"
}
```

### 2. Query Documents
```bash
curl -X POST "http://localhost:8000/query?query_text=How%20does%20caching%20improve%20performance"
```

**Response:**
```json
{
  "status": "success",
  "query": "How does caching improve performance",
  "intent": "reasoning",
  "context": "[Document Name]\nRelevant text chunks...",
  "retrieved_chunks": 6,
  "response_time_ms": 245.3
}
```

### 3. List Documents
```bash
curl "http://localhost:8000/documents"
```

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "documents": [
    {
      "id": 1,
      "filename": "guide.pdf",
      "file_type": "pdf",
      "file_size": 123456,
      "total_chunks": 42,
      "uploaded_at": "2024-02-22T10:30:00",
      "processed": true
    }
  ]
}
```

### 4. Delete Document
```bash
curl -X DELETE "http://localhost:8000/documents/1"
```

### 5. View Statistics
```bash
curl "http://localhost:8000/stats"
```

**Response:**
```json
{
  "status": "success",
  "statistics": {
    "total_documents": 2,
    "total_chunks": 84,
    "total_queries": 15,
    "chunks_in_memory": 84,
    "average_response_time_ms": 245.3,
    "config": {
      "chunk_size": 140,
      "overlap": 40,
      "max_final_chunks": 6
    }
  }
}
```

### 6. Health Check
```bash
curl "http://localhost:8000/health"
```

## Interactive API Documentation

Swagger UI: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

Try out endpoints directly in the browser!

## Configuration

Edit `.env` file or environment variables:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# RAG Parameters
CHUNK_WORDS=140          # Words per chunk (increase = longer context)
CHUNK_OVERLAP=40         # Overlap between chunks
TOP_K_FINAL_CHUNKS=6     # Return top N chunks in context

# Retrieval Tuning
TOP_K_VECTOR=80          # Vector search candidates
TOP_K_BM25=80            # BM25 search candidates
TOP_K_FUSED=150          # After RRF fusion
RRF_K=60                 # RRF normalization

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Upload
MAX_FILE_SIZE=52428800   # 50 MB
```

## Understanding Query Intent

The system automatically detects query intent and adjusts retrieval:

| Intent | Triggers | Behavior |
|--------|----------|----------|
| **Definition** | "what is", "define", "explain" | Lower rerank depth, single document |
| **Reasoning** | "why", "reason", "cause" | Deep rerank, multiple sources |
| **Comparison** | "compare", "vs", "difference" | Highest diversity, multiple docs |
| **Procedural** | "how to", "steps", "procedure" | Medium depth, focused retrieval |
| **Troubleshooting** | "error", "problem", "issue" | Maximum depth, comprehensive search |
| **General** | Anything else | Balanced approach |

## Database Structure

### Documents Table
- `id` - Document ID
- `filename` - Original filename
- `file_type` - txt, pdf, or docx
- `file_size` - Size in bytes
- `total_chunks` - Number of chunks created
- `uploaded_at` - Upload timestamp
- `processed` - Processing status
- `error_message` - If processing failed

### Chunks Table
- `id` - Chunk ID
- `document_id` - Parent document
- `chunk_index` - Position in document
- `text` - Chunk content
- `start_word` - Word offset in original
- `page_number` - For PDFs
- `section` - For structured docs

### Query Logs Table
- `id` - Query ID
- `query_text` - User query
- `intent` - Detected intent type
- `retrieved_chunks` - Number of chunks returned
- `response_time_ms` - Latency
- `executed_at` - Query timestamp

## Performance Tips

1. **Adjust chunk size** for your domain:
   - Smaller chunks (80-100 words): Better precision, more chunks
   - Larger chunks (200+ words): More context, fewer results

2. **Tune TOP_K values** based on document count:
   - 5-10 docs: Keep defaults
   - 50+ docs: Increase TOP_K_VECTOR to 100+

3. **Model selection**:
   - Default `all-MiniLM-L6-v2`: Fast, lightweight (~500MB)
   - Alternative `all-mpnet-base-v2`: Slower, better quality (~500MB)

4. **Database cleanup**:
   - Monitor `data/rag_system.db` size
   - Periodically delete old documents

## Troubleshooting

### No context retrieved
- Check if documents are uploaded: `GET /documents`
- Query might be too specific
- Try simpler query with more keywords

### Slow response time
- Reduce `TOP_K_*` values in config
- Use smaller embedding model
- Check system memory

### Memory issues
- Reduce `TOP_K_FINAL_CHUNKS`
- Use fewer documents
- Increase field `CHUNK_OVERLAP`

### PDF extraction issues
- Ensure PDF is text-based (not scanned image)
- Try simpler PDFs first
- Check file is not corrupted

## Project Structure

```
SmartMultiDocRAG/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── database.py            # SQLite database models
├── document_processor.py  # PDF/DOCX/TXT extraction
├── rag_core.py           # Core RAG logic
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── data/
│   ├── documents/        # Uploaded files
│   ├── vector_db/        # Vector store
│   └── rag_system.db    # SQLite database
└── logs/
    └── rag_system.log   # Application logs
```

## Advanced Usage

### Python Client Example
```python
import requests

BASE_URL = "http://localhost:8000"

# Upload document
with open("guide.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print(response.json())

# Query
response = requests.post(
    f"{BASE_URL}/query",
    params={"query_text": "How to optimize performance?"}
)
print(response.json())
```

### Docker Deployment (Phase 3)
See `docker-compose.yml` and `Dockerfile` (coming soon)

## Roadmap

- [x] Phase 1: Core API & Persistence
  - [x] FastAPI application
  - [x] SQLite database
  - [x] File upload
  - [x] Error handling

- [x] Phase 2: Multi-Format Support
  - [x] PDF extraction
  - [x] DOCX support
  - [x] Better chunking

- [ ] Phase 3: Production Hardening
  - [ ] Authentication & authorization
  - [ ] Rate limiting
  - [ ] Comprehensive logging
  - [ ] Docker/Compose setup

- [ ] Phase 4: Performance & Scaling
  - [ ] Caching layer
  - [ ] Async processing
  - [ ] Query optimization

- [ ] Phase 5: Advanced Features
  - [ ] Document versioning
  - [ ] Advanced analytics
  - [ ] Admin dashboard
  - [ ] Web UI

## License

MIT License

## Support

For issues and feature requests: Create an issue or contact support.

---

**Last Updated**: February 2024
**Version**: 2.0.0
