# smart_rag_advanced.py
# Document-Agnostic Smart RAG with Advanced Context Expansion

import os
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

# ---------------- CONFIG ---------------- #

DOC_FOLDER = "Documents"

CHUNK_WORDS = 140
CHUNK_OVERLAP = 40

TOP_K_VECTOR = 80
TOP_K_BM25 = 80
TOP_K_FUSED = 150

TOP_K_FINAL_CHUNKS = 6

RRF_K = 60

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------- MODELS ---------------- #

print("Loading models...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# ---------------- INTENT ---------------- #

def classify_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["what is", "define", "definition", "explain"]):
        return "definition"
    if any(x in q for x in ["why", "reason", "cause"]):
        return "reasoning"
    if any(x in q for x in ["compare", "difference", "vs", "versus"]):
        return "comparison"
    if any(x in q for x in ["how to", "steps", "procedure"]):
        return "procedural"
    if any(x in q for x in ["error", "issue", "problem", "failure"]):
        return "troubleshooting"

    return "general"

INTENT_CONFIG = {
    "definition": {"rerank_depth": 30, "doc_limit": 2, "diversity": False},
    "reasoning": {"rerank_depth": 60, "doc_limit": 3, "diversity": False},
    "comparison": {"rerank_depth": 70, "doc_limit": 4, "diversity": True},
    "procedural": {"rerank_depth": 50, "doc_limit": 3, "diversity": False},
    "troubleshooting": {"rerank_depth": 80, "doc_limit": 4, "diversity": False},
    "general": {"rerank_depth": 50, "doc_limit": 3, "diversity": False},
}

# ---------------- CHUNKING ---------------- #

def adaptive_chunk(text):
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP

    for i in range(0, len(words), step):
        yield i, " ".join(words[i:i + CHUNK_WORDS])

# ---------------- LOAD ---------------- #

def load_documents(folder):
    chunks = []
    metadatas = []

    print("Loading documents...")

    for file in os.listdir(folder):
        if not file.endswith(".txt"):
            continue

        doc_name = file.replace(".txt", "")
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            text = f.read()

        for start_idx, chunk in adaptive_chunk(text):
            chunks.append(chunk)
            metadatas.append({
                "DocName": doc_name,
                "StartWord": start_idx
            })

    return chunks, metadatas

# ---------------- INDICES ---------------- #

def build_indices(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    bm25 = BM25Okapi([c.split() for c in chunks])

    return embeddings, index, bm25

# ---------------- RETRIEVAL ---------------- #

def vector_retrieval(query, index):
    q_vec = embedder.encode([query])
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    _, ids = index.search(q_vec, TOP_K_VECTOR)
    return ids[0]

def bm25_retrieval(query, bm25):
    scores = bm25.get_scores(query.split())
    return np.argsort(scores)[::-1][:TOP_K_BM25]

# ---------------- RRF ---------------- #

def reciprocal_rank_fusion(vec_ids, bm25_ids):
    scores = defaultdict(float)

    for rank, idx in enumerate(vec_ids):
        scores[idx] += 1 / (RRF_K + rank)

    for rank, idx in enumerate(bm25_ids):
        scores[idx] += 1 / (RRF_K + rank)

    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_FUSED]]

# ---------------- RERANK ---------------- #

def rerank(query, candidate_ids, chunks):
    pairs = [[query, chunks[i]] for i in candidate_ids]
    scores = cross_encoder.predict(pairs)

    return sorted(zip(scores, candidate_ids), reverse=True)

# ---------------- GRAPH + EXPANSION ---------------- #

def build_doc_graph(metadatas):
    graph = defaultdict(list)

    for i, meta_i in enumerate(metadatas):
        for j, meta_j in enumerate(metadatas):
            if meta_i["DocName"] != meta_j["DocName"]:
                continue

            if abs(meta_i["StartWord"] - meta_j["StartWord"]) <= CHUNK_WORDS:
                graph[i].append(j)

    return graph

def adaptive_window(score):
    if score > 0.8:
        return 2
    if score > 0.5:
        return 1
    return 0

def expand_with_graph(reranked, graph, metadatas):
    expanded = set()

    for score, cid in reranked[:TOP_K_FINAL_CHUNKS]:
        window = adaptive_window(score)

        expanded.add(cid)

        if window == 0:
            continue

        for neighbor in graph[cid]:
            distance = abs(metadatas[neighbor]["StartWord"] - metadatas[cid]["StartWord"])

            if distance <= CHUNK_WORDS * window:
                expanded.add(neighbor)

    return sorted(expanded)

def suppress_overlaps(chunk_ids, metadatas):
    filtered = []

    for cid in chunk_ids:
        keep = True

        for existing in filtered:
            if metadatas[cid]["DocName"] != metadatas[existing]["DocName"]:
                continue

            if abs(metadatas[cid]["StartWord"] - metadatas[existing]["StartWord"]) < CHUNK_OVERLAP:
                keep = False
                break

        if keep:
            filtered.append(cid)

    return filtered

# ---------------- CONTEXT ---------------- #

def build_context(chunk_ids, chunks, metadatas):
    blocks = []
    for cid in chunk_ids[:TOP_K_FINAL_CHUNKS]:
        blocks.append(f"[{metadatas[cid]['DocName']}]\n{chunks[cid]}")

    return "\n\n".join(blocks)

# ---------------- PIPELINE ---------------- #

def run_query(query, chunks, metadatas, index, bm25, graph):
    intent = classify_intent(query)
    cfg = INTENT_CONFIG[intent]

    print("\nIntent:", intent)

    vec_ids = vector_retrieval(query, index)
    bm25_ids = bm25_retrieval(query, bm25)

    fused_ids = reciprocal_rank_fusion(vec_ids, bm25_ids)

    reranked = rerank(query, fused_ids[:cfg["rerank_depth"]], chunks)

    expanded_ids = expand_with_graph(reranked, graph, metadatas)

    non_overlapping = suppress_overlaps(expanded_ids, metadatas)

    context = build_context(non_overlapping, chunks, metadatas)

    print("\n---------------- CONTEXT ----------------\n")
    print(context)
    print("\n-----------------------------------------\n")

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    chunks, metadatas = load_documents(DOC_FOLDER)
    embeddings, index, bm25 = build_indices(chunks)

    graph = build_doc_graph(metadatas)

    while True:
        q = input("🔎 Query (exit to quit): ")
        if q.lower() == "exit":
            break

        run_query(q, chunks, metadatas, index, bm25, graph)