import os
import gc
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.data_loader import load_medical_terms
from src.embedding_model import create_embeddings, load_embedding_model
from src.similarity_engine import build_faiss_index, search_similar_terms
from src.tfidf_comparison import find_tfidf_matches


DATASET_PATH = Path(__file__).parent / "data" / "medical_terms.csv"


def normalize_query(value):
    return " ".join(str(value or "").split())


def score_as_percentage(score):
    return round(max(0.0, min(1.0, score)) * 100, 1)


def similarity_badge(percentage):
    if percentage >= 80:
        return "Highly Similar"
    if percentage >= 55:
        return "Related"
    return "Moderately Similar"


def create_app():
    app = Flask(__name__)

    # Allow GitHub Pages frontend to call Render API
    CORS(app)

    state = {
        "model": None,
        "index": None,
        "terms": [],
        "metadata": {},
        "error": None,
    }

    try:
        print("Loading medical dataset...")

        medical_data = load_medical_terms(DATASET_PATH)

        terms = medical_data["term"].tolist()

        # Store metadata before deleting dataframe
        metadata = {
            row["term"].casefold(): row
            for row in medical_data.to_dict("records")
        }

        state["terms"] = terms
        state["metadata"] = metadata

        # Free pandas memory
        del medical_data
        gc.collect()

        print("Loading Sentence-BERT model...")

        state["model"] = load_embedding_model()

        print("Creating embeddings...")

        embeddings = create_embeddings(
            state["model"],
            terms
        )

        print("Building FAISS index...")

        state["index"] = build_faiss_index(embeddings)

        # Free temporary embedding memory
        del embeddings
        gc.collect()

        print("Model and FAISS index ready.")

    except Exception as error:
        print("Startup error:", error)
        state["error"] = str(error)

    def make_result(term, score, query):

        metadata = state["metadata"].get(
            term.casefold(),
            {}
        )

        percentage = score_as_percentage(score)

        relation = metadata.get(
            "relation_type",
            "Related term"
        )

        if term.casefold() == query.casefold():
            relation = "Exact match"

        return {
            "term": term,
            "category": metadata.get(
                "category",
                "Medical terminology"
            ),
            "description": metadata.get(
                "description",
                "A related term from the project dataset."
            ),
            "related_terms": [
                item.strip()
                for item in metadata.get(
                    "related_terms",
                    ""
                ).split("|")
                if item.strip()
            ],
            "relation": relation,
            "score": round(score, 4),
            "percentage": percentage,
            "badge": similarity_badge(percentage),
        }

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            total_terms=len(state["terms"])
        )

    @app.get("/api/health")
    def health():

        ready = (
            state["error"] is None
            and state["model"] is not None
            and state["index"] is not None
        )

        return jsonify({
            "ready": ready,
            "total_terms": len(state["terms"]),
            "error": state["error"]
        })

    @app.post("/api/search")
    def search():

        if (
            state["error"]
            or state["model"] is None
            or state["index"] is None
        ):
            return jsonify({
                "error": "The search model is not ready yet."
            }), 503

        payload = request.get_json(
            silent=True
        ) or {}

        query = normalize_query(
            payload.get("query")
        )

        if not query:
            return jsonify({
                "error": "Please enter a medical term to search."
            }), 400

        if len(query) < 2:
            return jsonify({
                "error": "Please enter at least two characters."
            }), 400

        try:

            # Create embedding only for current query
            query_embedding = create_embeddings(
                state["model"],
                [query]
            )

            semantic_matches = search_similar_terms(
                state["index"],
                query_embedding,
                state["terms"],
                top_k=5
            )

            semantic_results = [
                make_result(
                    term,
                    score,
                    query
                )
                for term, score in semantic_matches
            ]

            # Free query embedding after search
            del query_embedding
            gc.collect()

            tfidf_matches = find_tfidf_matches(
                state["terms"],
                query,
                top_k=5
            )

            tfidf_results = [
                make_result(
                    term,
                    score,
                    query
                )
                for term, score in tfidf_matches
            ]

            percentages = [
                item["percentage"]
                for item in semantic_results
            ]

            best_match = (
                semantic_results[0]
                if semantic_results
                else None
            )

            return jsonify({
                "query": query,

                "results": semantic_results,

                "stats": {
                    "total_terms": len(state["terms"]),
                    "results_found": len(semantic_results),
                    "highest_similarity": (
                        best_match["percentage"]
                        if best_match
                        else 0
                    ),
                    "average_similarity": (
                        round(
                            sum(percentages)
                            / len(percentages),
                            1
                        )
                        if percentages
                        else 0
                    )
                },

                "comparison": {
                    "sentence_bert": (
                        semantic_results[0]
                        if semantic_results
                        else None
                    ),
                    "tfidf": (
                        tfidf_results[0]
                        if tfidf_results
                        else None
                    )
                },

                "tfidf_results": tfidf_results
            })

        except Exception as error:

            print("Search error:", error)

            return jsonify({
                "error": "The search could not be completed."
            }), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )