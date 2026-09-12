"""Semantic similarity search for a small medical terminology dataset."""

from pathlib import Path

from src.data_loader import load_medical_terms
from src.embedding_model import create_embeddings, load_embedding_model
from src.similarity_engine import build_faiss_index, search_similar_terms
from src.tfidf_comparison import find_tfidf_matches


DATASET_PATH = Path(__file__).parent / "data" / "medical_terms.csv"


def print_results(title: str, results: list[tuple[str, float]]) -> None:
    """Print ranked search results in a readable format."""
    print(f"\n{title}")
    for rank, (term, score) in enumerate(results, start=1):
        print(f"{rank}. {term} - Similarity: {score:.4f}")


def main() -> None:
    """Load data, search with Sentence-BERT, then compare with TF-IDF."""
    try:
        medical_data = load_medical_terms(DATASET_PATH)
        terms = medical_data["term"].tolist()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading dataset: {error}")
        return

    print(f"Loaded {len(terms)} medical terms.")
    print("Loading Sentence-BERT model. The first run may download the model...")
    model = load_embedding_model()
    term_embeddings = create_embeddings(model, terms)
    faiss_index = build_faiss_index(term_embeddings)

    query = input("\nEnter a medical term: ").strip()
    if not query:
        print("Please enter a non-empty medical term.")
        return

    query_embedding = create_embeddings(model, [query])
    semantic_results = search_similar_terms(faiss_index, query_embedding, terms)
    print_results("Top Similar Medical Terms (Sentence-BERT + FAISS):", semantic_results)

    tfidf_results = find_tfidf_matches(terms, query)
    print_results("Traditional Baseline (TF-IDF + cosine similarity):", tfidf_results)


if __name__ == "__main__":
    main()
