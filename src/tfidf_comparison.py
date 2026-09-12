"""Traditional TF-IDF and cosine-similarity baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_tfidf_matches(
    terms: list[str], query: str, top_k: int = 5
) -> list[tuple[str, float]]:
    """Compare the query with terms using word-level TF-IDF."""
    vectorizer = TfidfVectorizer(lowercase=True)
    term_vectors = vectorizer.fit_transform(terms + [query])
    scores = cosine_similarity(term_vectors[-1], term_vectors[:-1]).ravel()
    ranked_positions = scores.argsort()[::-1][:top_k]
    return [(terms[position], float(scores[position])) for position in ranked_positions]
