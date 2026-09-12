"""Flask web application for medical terminology similarity matching."""

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.data_loader import load_medical_terms
from src.embedding_model import create_embeddings, load_embedding_model
from src.similarity_engine import build_faiss_index, search_similar_terms
from src.tfidf_comparison import find_tfidf_matches


DATASET_PATH = Path(__file__).parent / "data" / "medical_terms.csv"


def normalize_query(value: object) -> str:
    """Collapse extra whitespace and return a clean search phrase."""
    return " ".join(str(value or "").split())


def score_as_percentage(score: float) -> float:
    """Convert cosine-like scores into a user-friendly 0-100 range."""
    return round(max(0.0, min(1.0, score)) * 100, 1)


def similarity_badge(percentage: float) -> str:
    """Choose a simple label for displaying a match strength."""
    if percentage >= 80:
        return "Highly Similar"
    if percentage >= 55:
        return "Related"
    return "Moderately Similar"


def create_app() -> Flask:
    """Create the Flask app and load reusable search resources once."""
    app = Flask(__name__)
    state = {
        "model": None,
        "index": None,
        "terms": [],
        "metadata": {},
        "error": None,
    }

    try:
        medical_data = load_medical_terms(DATASET_PATH)
        terms = medical_data["term"].tolist()
        state["model"] = load_embedding_model()
        embeddings = create_embeddings(state["model"], terms)
        state["index"] = build_faiss_index(embeddings)
        state["terms"] = terms
        state["metadata"] = {
            row["term"].casefold(): row for row in medical_data.to_dict("records")
        }
    except Exception as error:
        state["error"] = str(error)

    def make_result(term: str, score: float, query: str) -> dict:
        metadata = state["metadata"].get(term.casefold(), {})
        percentage = score_as_percentage(score)
        relation = metadata.get("relation_type") or "Related term"
        if term.casefold() == query.casefold():
            relation = "Exact match"
        return {
            "term": term,
            "category": metadata.get("category", "Medical terminology"),
            "description": metadata.get(
                "description", "A related term from the project dataset."
            ),
            "related_terms": [
                item.strip()
                for item in metadata.get("related_terms", "").split("|")
                if item.strip()
            ],
            "relation": relation,
            "score": round(score, 4),
            "percentage": percentage,
            "badge": similarity_badge(percentage),
        }

    @app.get("/")
    def home():
        return render_template("index.html", total_terms=len(state["terms"]))

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "ready": state["error"] is None and state["model"] is not None,
                "total_terms": len(state["terms"]),
                "error": "Model or dataset could not be loaded." if state["error"] else None,
            }
        )

    @app.post("/api/search")
    def search():
        if state["error"] or state["model"] is None or state["index"] is None:
            return jsonify({"error": "The search model is not ready yet."}), 503

        payload = request.get_json(silent=True) or {}
        query = normalize_query(payload.get("query"))
        if not query:
            return jsonify({"error": "Please enter a medical term to search."}), 400
        if len(query) < 2:
            return jsonify({"error": "Please enter at least two characters."}), 400

        try:
            query_embedding = create_embeddings(state["model"], [query])
            semantic_matches = search_similar_terms(
                state["index"], query_embedding, state["terms"], top_k=5
            )
            semantic_results = [
                make_result(term, score, query) for term, score in semantic_matches
            ]

            tfidf_matches = find_tfidf_matches(state["terms"], query, top_k=5)
            tfidf_results = [
                make_result(term, score, query) for term, score in tfidf_matches
            ]

            percentages = [item["percentage"] for item in semantic_results]
            best_match = semantic_results[0] if semantic_results else None
            return jsonify(
                {
                    "query": query,
                    "results": semantic_results,
                    "stats": {
                        "total_terms": len(state["terms"]),
                        "results_found": len(semantic_results),
                        "highest_similarity": best_match["percentage"] if best_match else 0,
                        "average_similarity": round(sum(percentages) / len(percentages), 1)
                        if percentages
                        else 0,
                    },
                    "comparison": {
                        "sentence_bert": semantic_results[0] if semantic_results else None,
                        "tfidf": tfidf_results[0] if tfidf_results else None,
                    },
                    "tfidf_results": tfidf_results,
                }
            )
        except Exception:
            return jsonify({"error": "The search could not be completed. Please try again."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run()
