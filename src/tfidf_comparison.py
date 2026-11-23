"""Traditional TF-IDF and cosine-similarity baseline."""

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache(maxsize=1)
def _build_vectorizer(terms_tuple: tuple[str, ...]) -> tuple:
    """Fit and cache a vectorizer over the full term list."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),   # capture bigrams like "heart attack"
        sublinear_tf=True,    # log-scale TF dampens common terms
    )
    term_vectors = vectorizer.fit_transform(list(terms_tuple))
    return vectorizer, term_vectors


def find_tfidf_matches(
    terms: list[str], query: str, top_k: int = 5
) -> list[tuple[str, float]]:
    """Compare the query with terms using word-level TF-IDF with bigrams."""
    terms_tuple = tuple(terms)
    vectorizer, term_vectors = _build_vectorizer(terms_tuple)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, term_vectors).ravel()
    ranked_positions = scores.argsort()[::-1][:top_k]
    return [(terms[pos], float(scores[pos])) for pos in ranked_positions]
