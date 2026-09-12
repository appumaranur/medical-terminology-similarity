"""FAISS-based semantic similarity search."""

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an inner-product index for already-normalized embeddings."""
    vector_size = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_size)
    index.add(embeddings.astype("float32"))
    return index


def search_similar_terms(
    index: faiss.Index,
    query_embedding: np.ndarray,
    terms: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return terms and cosine-like scores in descending order."""
    number_to_return = min(top_k, len(terms))
    scores, positions = index.search(query_embedding.astype("float32"), number_to_return)
    return [
        (terms[position], float(score))
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    ]
