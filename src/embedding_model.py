"""Create memory-conscious Sentence-BERT embeddings for medical terms."""

import os

import torch
from sentence_transformers import SentenceTransformer


# This smaller Sentence-BERT model fits Render's free memory limit more reliably.
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load_embedding_model() -> SentenceTransformer:
    """Load the pretrained model from the local cache or Hugging Face."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model: SentenceTransformer, terms: list[str]):
    """Encode terms and normalize vectors for cosine similarity."""
    return model.encode(
        terms,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
