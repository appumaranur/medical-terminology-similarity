"""Create Sentence-BERT embeddings for medical terms."""

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """Load the pretrained model from the local cache or Hugging Face."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model: SentenceTransformer, terms: list[str]):
    """Encode terms and normalize vectors for cosine similarity."""
    return model.encode(
        terms,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
