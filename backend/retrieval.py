"""
retrieval.py — Hybrid retrieval pipeline.

Pipeline:
  1. dense_search  : Cosine similarity over the numpy embedding matrix.
  2. bm25_search   : Keyword-based retrieval using the BM25 index.
  3. reciprocal_rank_fusion : Merge both ranked lists into a single score.
  4. retrieve      : Orchestrator — runs both searches, fuses, filters, returns top-k.
"""

import re
import numpy as np
from store import DocumentStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenizer — must match store.py's tokenizer."""
    return re.findall(r'\w+', text.lower())


def _cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and every row
    of `matrix` in one vectorised operation.

    Both inputs are assumed to be L2-normalised (which embed_chunks guarantees
    via normalize_embeddings=True), so cosine similarity reduces to a dot product.

    Args:
        query_vec: Shape (D,) — a single, L2-normalised query embedding.
        matrix:    Shape (N, D) — L2-normalised chunk embeddings.

    Returns:
        scores: Shape (N,) — cosine similarity ∈ [-1, 1] for each chunk.
    """
    # Ensure query is 1-D
    q = query_vec.flatten().astype(np.float32)

    # If embeddings are already unit-normalised, dot product == cosine similarity.
    # We still normalise defensively to handle un-normalised inputs gracefully.
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for zero-norm rows
    m_norms = np.where(m_norms == 0, 1.0, m_norms)
    normalised_matrix = matrix / m_norms

    return normalised_matrix @ q  # shape (N,)


# ---------------------------------------------------------------------------
# 1. dense_search
# ---------------------------------------------------------------------------

def dense_search(
    query_embedding: list[float] | np.ndarray,
    store: DocumentStore,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve the top-k chunks by cosine similarity to the query embedding.

    Args:
        query_embedding: The query's embedding vector (list or numpy array).
        store:           A populated DocumentStore instance.
        top_k:           Number of results to return.

    Returns:
        List of result dicts (descending score order):
          {
            "chunk":      dict,   # the original chunk dict from the store
            "score":      float,  # cosine similarity ∈ [-1, 1]
            "rank_type":  "dense",
          }
    """
    matrix = store.get_embedding_matrix()
    chunks = store.get_all_chunks()

    if matrix is None or len(chunks) == 0:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    scores = _cosine_similarity_matrix(q, matrix)

    # Descending sort, take top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "chunk": chunks[i],
            "score": float(scores[i]),
            "rank_type": "dense",
        }
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# 2. bm25_search
# ---------------------------------------------------------------------------

def bm25_search(
    query_text: str,
    store: DocumentStore,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve the top-k chunks by BM25 score.

    Args:
        query_text: Raw query string (will be tokenized internally).
        store:      A populated DocumentStore instance.
        top_k:      Number of results to return.

    Returns:
        List of result dicts (descending score order):
          {
            "chunk":      dict,
            "score":      float,  # raw BM25 score (unbounded)
            "rank_type":  "bm25",
          }
    """
    bm25_index = store.get_bm25()
    chunks = store.get_all_chunks()

    if bm25_index is None or len(chunks) == 0:
        return []

    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    scores = bm25_index.get_scores(query_tokens)  # shape (N,)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "chunk": chunks[i],
            "score": float(scores[i]),
            "rank_type": "bm25",
        }
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# 3. reciprocal_rank_fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine two ranked result lists into one via Reciprocal Rank Fusion.

    RRF formula (Cormack et al., 2009):
        RRF_score(d) = Σ  1 / (k + rank(d, list_i))

    where rank is 1-based and k is a smoothing constant (typically 60).
    Documents absent from a list receive no contribution from that list.

    Args:
        dense_results: Output of dense_search().
        bm25_results:  Output of bm25_search().
        k:             Smoothing constant (default 60).

    Returns:
        Fused list of result dicts (descending RRF score order):
          {
            "chunk":     dict,
            "rrf_score": float,
          }
    """
    # Map chunk id → RRF score accumulator
    rrf_scores: dict[str, float] = {}
    # Map chunk id → chunk dict (for reconstruction)
    id_to_chunk: dict[str, dict] = {}

    def _accumulate(results: list[dict]) -> None:
        for rank_0, result in enumerate(results):
            chunk = result["chunk"]
            chunk_id = chunk["id"]
            rrf_contribution = 1.0 / (k + rank_0 + 1)  # 1-based rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution
            id_to_chunk[chunk_id] = chunk

    _accumulate(dense_results)
    _accumulate(bm25_results)

    # Sort by descending RRF score
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    return [
        {
            "chunk": id_to_chunk[cid],
            "rrf_score": rrf_scores[cid],
        }
        for cid in sorted_ids
    ]


# ---------------------------------------------------------------------------
# 4. retrieve  (orchestrator)
# ---------------------------------------------------------------------------

def retrieve(
    query_text: str,
    query_embedding: list[float] | np.ndarray,
    store: DocumentStore,
    top_k: int = 5,
    confidence_threshold: float = 0.30,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline with out-of-scope filtering.

    Steps:
      1. Run dense_search and bm25_search (each fetches top_k * 2 candidates).
      2. Fuse ranked lists via RRF.
      3. Take the top `top_k` fused results.
      4. Compute cosine similarity for every result to use as a confidence score.
      5. If the highest confidence score < `confidence_threshold`, return []
         (signals that no document in the store is relevant to the query).

    Args:
        query_text:            Raw query string.
        query_embedding:       Embedded query vector.
        store:                 Populated DocumentStore.
        top_k:                 Number of final results to return (default 5).
        confidence_threshold:  Minimum cosine similarity for the best result
                               before the query is considered out-of-scope (default 0.30).

    Returns:
        List of result dicts (descending confidence order), or [] if out-of-scope:
          {
            "chunk":      dict,   # original chunk with doc_name, page_num, text, etc.
            "confidence": float,  # cosine similarity ∈ [0, 1] (clipped at 0)
            "rrf_score":  float,  # raw RRF score used for ranking
          }
    """
    if store.get_embedding_matrix() is None:
        return []

    # Use a wider candidate pool for better fusion coverage.
    candidate_k = max(top_k * 2, 10)

    dense_results = dense_search(query_embedding, store, top_k=candidate_k)
    bm25_results = bm25_search(query_text, store, top_k=candidate_k)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)

    # Trim to the requested top_k
    top_fused = fused[:top_k]

    if not top_fused:
        return []

    # Compute cosine similarity for each fused result to serve as confidence.
    q = np.array(query_embedding, dtype=np.float32)
    matrix = store.get_embedding_matrix()
    chunks = store.get_all_chunks()

    # Build a quick id→matrix-row-index lookup
    id_to_index: dict[str, int] = {c["id"]: i for i, c in enumerate(chunks)}

    results = []
    for item in top_fused:
        chunk = item["chunk"]
        idx = id_to_index.get(chunk["id"])
        if idx is None:
            continue  # chunk was removed mid-flight (shouldn't happen in-memory)

        chunk_vec = matrix[idx].astype(np.float32)
        # Cosine similarity (clip negative values to 0 for a [0, 1] confidence range)
        similarity = float(np.clip(_cosine_similarity_matrix(q, chunk_vec[np.newaxis, :]), 0, 1))

        results.append({
            "chunk": chunk,
            "confidence": similarity,
            "rrf_score": item["rrf_score"],
        })

    # Out-of-scope guard: if the best confidence is below the threshold, reject all.
    if not results or results[0]["confidence"] < confidence_threshold:
        return []

    return results
