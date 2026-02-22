import logging
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

from config import settings

logger = logging.getLogger(__name__)


class RAGCore:
    """Core RAG retrieval and ranking logic"""

    def __init__(self):
        """Initialize RAG system with models and indices"""
        logger.info("Initializing RAG models...")

        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)

        # Indices will be built when documents are added
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.metadatas = []
        self.doc_graph = {}

        logger.info("RAG models loaded successfully")

    def classify_intent(self, query: str) -> str:
        """
        Classify query intent for adaptive retrieval

        Args:
            query: User query text

        Returns:
            Intent type (definition, reasoning, comparison, procedural, troubleshooting, general)
        """
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

    def build_indices(self, chunks: List[str]) -> None:
        """
        Build FAISS vector index and BM25 index

        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Building indices for {len(chunks)} chunks...")

        # Build FAISS index
        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        # Build BM25 index
        self.bm25_index = BM25Okapi([c.split() for c in chunks])
        self.chunks = chunks

        logger.info("Indices built successfully")

    def build_doc_graph(self, metadatas: List[Dict]) -> None:
        """
        Build adjacency graph of chunks within same document

        Args:
            metadatas: List of chunk metadata dicts
        """
        self.metadatas = metadatas
        graph = defaultdict(list)

        for i, meta_i in enumerate(metadatas):
            for j, meta_j in enumerate(metadatas):
                # Only connect chunks from same document
                if meta_i.get("DocName") != meta_j.get("DocName"):
                    continue

                # Connect if word positions are within chunk distance
                if abs(meta_i.get("StartWord", 0) - meta_j.get("StartWord", 0)) <= settings.CHUNK_WORDS:
                    graph[i].append(j)

        self.doc_graph = graph
        logger.info(f"Document graph built with {len(graph)} nodes")

    def vector_retrieval(self, query: str) -> np.ndarray:
        """
        Retrieve chunks using semantic similarity

        Args:
            query: Query text

        Returns:
            Array of chunk indices
        """
        if self.faiss_index is None:
            return np.array([])

        q_vec = self.embedder.encode([query])
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        _, ids = self.faiss_index.search(q_vec, settings.TOP_K_VECTOR)
        return ids[0]

    def bm25_retrieval(self, query: str) -> np.ndarray:
        """
        Retrieve chunks using BM25 keyword matching

        Args:
            query: Query text

        Returns:
            Array of chunk indices
        """
        if self.bm25_index is None:
            return np.array([])

        scores = self.bm25_index.get_scores(query.split())
        return np.argsort(scores)[::-1][: settings.TOP_K_BM25]

    def reciprocal_rank_fusion(self, vec_ids: np.ndarray, bm25_ids: np.ndarray) -> List[int]:
        """
        Fuse vector and BM25 results using RRF

        Args:
            vec_ids: Vector search result indices
            bm25_ids: BM25 search result indices

        Returns:
            Fused list of chunk indices
        """
        scores = defaultdict(float)

        for rank, idx in enumerate(vec_ids):
            scores[idx] += 1 / (settings.RRF_K + rank)

        for rank, idx in enumerate(bm25_ids):
            scores[idx] += 1 / (settings.RRF_K + rank)

        return [
            idx
            for idx, _ in sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )[: settings.TOP_K_FUSED]
        ]

    def rerank(self, query: str, candidate_ids: List[int]) -> List[Tuple[float, int]]:
        """
        Rerank candidates using cross-encoder

        Args:
            query: Query text
            candidate_ids: Candidate chunk indices

        Returns:
            List of (score, chunk_id) tuples sorted by score
        """
        if not candidate_ids or not self.chunks:
            return []

        pairs = [[query, self.chunks[i]] for i in candidate_ids if i < len(self.chunks)]
        if not pairs:
            return []

        scores = self.cross_encoder.predict(pairs)
        return sorted(zip(scores, candidate_ids[: len(scores)]), reverse=True)

    def adaptive_window(self, score: float) -> int:
        """
        Determine expansion window size based on relevance score

        Args:
            score: Cross-encoder relevance score

        Returns:
            Window expansion level (0, 1, or 2)
        """
        if score > 0.8:
            return 2
        if score > 0.5:
            return 1
        return 0

    def expand_with_graph(self, reranked: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """
        Expand selected chunks with nearby chunks from document graph

        Args:
            reranked: List of (score, chunk_id) tuples

        Returns:
            Expanded list of (score, chunk_id) tuples, preserving relevance order
        """
        expanded_dict = {}  # chunk_id -> (score, is_primary)

        for rank, (score, cid) in enumerate(reranked[: settings.TOP_K_FINAL_CHUNKS]):
            window = self.adaptive_window(score)

            # Add primary chunk with original score
            if cid not in expanded_dict:
                expanded_dict[cid] = (score, True)  # primary=True

            if window == 0 or cid not in self.doc_graph:
                continue

            # Add neighboring chunks with slightly reduced score (secondary)
            for neighbor in self.doc_graph[cid]:
                if neighbor >= len(self.metadatas) or neighbor in expanded_dict:
                    continue

                distance = abs(
                    self.metadatas[neighbor].get("StartWord", 0)
                    - self.metadatas[cid].get("StartWord", 0)
                )

                if distance <= settings.CHUNK_WORDS * window:
                    # Neighbor gets slightly lower score than primary
                    neighbor_score = score * 0.95
                    expanded_dict[neighbor] = (neighbor_score, False)  # primary=False

        # Return sorted by score (descending), then by chunk_id for stability
        return sorted(
            [(score, cid) for cid, (score, _) in expanded_dict.items()],
            key=lambda x: (-x[0], x[1])  # Sort by score desc, then ID asc
        )

    def suppress_overlaps(self, chunk_tuples: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """
        Remove overlapping chunks from same document, preserving relevance order

        Args:
            chunk_tuples: List of (score, chunk_id) tuples

        Returns:
            Filtered list without excessive overlaps, maintaining order
        """
        filtered = []

        for score, cid in chunk_tuples:
            if cid >= len(self.metadatas):
                continue

            keep = True

            # Check against already-kept chunks
            for existing_score, existing_cid in filtered:
                if existing_cid >= len(self.metadatas):
                    continue

                # Only compare chunks from same document
                if (
                    self.metadatas[cid].get("DocName")
                    != self.metadatas[existing_cid].get("DocName")
                ):
                    continue

                # Skip if too close (overlapping)
                if (
                    abs(
                        self.metadatas[cid].get("StartWord", 0)
                        - self.metadatas[existing_cid].get("StartWord", 0)
                    )
                    < settings.CHUNK_OVERLAP
                ):
                    keep = False
                    break

            if keep:
                filtered.append((score, cid))

        return filtered

    def build_context(self, chunk_tuples: List[Tuple[float, int]]) -> str:
        """
        Build final context from selected chunks in relevance order

        Args:
            chunk_tuples: List of (score, chunk_id) tuples, ordered by relevance

        Returns:
            Formatted context string with most relevant chunks first
        """
        blocks = []

        for score, cid in chunk_tuples[: settings.TOP_K_FINAL_CHUNKS]:
            if cid >= len(self.chunks) or cid >= len(self.metadatas):
                continue

            meta = self.metadatas[cid]
            doc_name = meta.get("DocName", "Unknown")
            chunk_text = self.chunks[cid]

            # Include relevance score as comment (useful for debugging)
            blocks.append(f"[{doc_name}] (relevance: {score:.2f})\n{chunk_text}")

        return "\n\n".join(blocks)

    def retrieve(self, query: str) -> Tuple[str, str, int]:
        """
        Execute full retrieval pipeline (main entry point)

        Args:
            query: User query text

        Returns:
            (context_text, intent, num_chunks)
        """
        if not self.chunks:
            return "", "general", 0

        # Classify intent
        intent = self.classify_intent(query)
        cfg = settings.INTENT_CONFIG.get(intent, settings.INTENT_CONFIG["general"])

        # Retrieve from both indices
        vec_ids = self.vector_retrieval(query)
        bm25_ids = self.bm25_retrieval(query)

        # Fuse results
        fused_ids = self.reciprocal_rank_fusion(vec_ids, bm25_ids)

        # Rerank top candidates
        rerank_depth = min(cfg["rerank_depth"], len(fused_ids))
        reranked = self.rerank(query, fused_ids[:rerank_depth])

        # Expand with graph neighbors
        expanded_ids = self.expand_with_graph(reranked)

        # Suppress overlaps
        final_ids = self.suppress_overlaps(expanded_ids)

        # Build context
        context = self.build_context(final_ids)

        return context, intent, len(final_ids)


# Global RAG instance
rag_system = RAGCore()
