import os
import gc
import csv
import io
import secrets
from pathlib import Path
from functools import wraps
from cachetools import LRUCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS
from dotenv import load_dotenv

from src.data_loader import load_medical_terms
from src.embedding_model import create_embeddings, load_embedding_model
from src.similarity_engine import build_faiss_index, search_similar_terms
from src.tfidf_comparison import find_tfidf_matches

load_dotenv()

DATASET_PATH = Path(__file__).parent / "data" / "medical_terms.csv"

# LRU cache for search results keyed by query string
_search_cache: LRUCache = LRUCache(maxsize=512)


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


def require_api_key(f):
    """Optional API key guard. Skipped when API_KEY env var is not set."""
    @wraps(f)
    def decorated(*args, **kwargs):
        expected = os.environ.get("API_KEY")
        if not expected:
            return f(*args, **kwargs)
        provided = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not provided or not secrets.compare_digest(provided, expected):
            return jsonify({"error": "Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated


def create_app():
    app = Flask(__name__)
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
        metadata = {
            row["term"].casefold(): row
            for row in medical_data.to_dict("records")
        }
        state["terms"] = terms
        state["metadata"] = metadata
        del medical_data
        gc.collect()

        print("Loading Sentence-BERT model...")
        state["model"] = load_embedding_model()

        print("Creating embeddings...")
        embeddings = create_embeddings(state["model"], terms)

        print("Building FAISS index...")
        state["index"] = build_faiss_index(embeddings)
        del embeddings
        gc.collect()

        print("Model and FAISS index ready.")
    except Exception as error:
        print("Startup error:", error)
        state["error"] = str(error)

    def make_result(term, score, query):
        meta = state["metadata"].get(term.casefold(), {})
        percentage = score_as_percentage(score)
        relation = meta.get("relation_type", "Related term")
        if term.casefold() == query.casefold():
            relation = "Exact match"
        return {
            "term": term,
            "category": meta.get("category", "Medical terminology"),
            "description": meta.get("description", "A related term from the project dataset."),
            "related_terms": [
                item.strip()
                for item in meta.get("related_terms", "").split("|")
                if item.strip()
            ],
            "relation": relation,
            "score": round(score, 4),
            "percentage": percentage,
            "badge": similarity_badge(percentage),
        }

    def _run_search(query: str):
        """Core search logic, used by both single and batch endpoints."""
        # Check cache first
        cache_key = query.casefold()
        if cache_key in _search_cache:
            return _search_cache[cache_key]

        query_embedding = create_embeddings(state["model"], [query])
        semantic_matches = search_similar_terms(
            state["index"], query_embedding, state["terms"], top_k=5
        )
        semantic_results = [make_result(t, s, query) for t, s in semantic_matches]
        del query_embedding
        gc.collect()

        tfidf_matches = find_tfidf_matches(state["terms"], query, top_k=5)
        tfidf_results = [make_result(t, s, query) for t, s in tfidf_matches]

        percentages = [item["percentage"] for item in semantic_results]
        best_match = semantic_results[0] if semantic_results else None

        result = {
            "query": query,
            "results": semantic_results,
            "stats": {
                "total_terms": len(state["terms"]),
                "results_found": len(semantic_results),
                "highest_similarity": best_match["percentage"] if best_match else 0,
                "average_similarity": (
                    round(sum(percentages) / len(percentages), 1)
                    if percentages else 0
                ),
            },
            "comparison": {
                "sentence_bert": semantic_results[0] if semantic_results else None,
                "tfidf": tfidf_results[0] if tfidf_results else None,
            },
            "tfidf_results": tfidf_results,
        }
        _search_cache[cache_key] = result
        return result

    def _model_ready():
        return (
            state["error"] is None
            and state["model"] is not None
            and state["index"] is not None
        )

    @app.get("/")
    def home():
        return render_template("index.html", total_terms=len(state["terms"]))

    @app.get("/api/health")
    def health():
        return jsonify({
            "ready": _model_ready(),
            "total_terms": len(state["terms"]),
            "cached_queries": len(_search_cache),
            "error": state["error"],
        })

    @app.post("/api/search")
    @require_api_key
    def search():
        if not _model_ready():
            return jsonify({"error": "The search model is not ready yet."}), 503

        payload = request.get_json(silent=True) or {}
        query = normalize_query(payload.get("query"))

        if not query:
            return jsonify({"error": "Please enter a medical term to search."}), 400
        if len(query) < 2:
            return jsonify({"error": "Please enter at least two characters."}), 400

        try:
            return jsonify(_run_search(query))
        except Exception as error:
            print("Search error:", error)
            return jsonify({"error": "The search could not be completed."}), 500

    @app.post("/api/batch_search")
    @require_api_key
    def batch_search():
        """Accept up to 20 queries at once and return results for each."""
        if not _model_ready():
            return jsonify({"error": "The search model is not ready yet."}), 503

        payload = request.get_json(silent=True) or {}
        raw_queries = payload.get("queries", [])

        if not isinstance(raw_queries, list):
            return jsonify({"error": "'queries' must be a JSON array."}), 400

        queries = [normalize_query(q) for q in raw_queries if normalize_query(q)]
        queries = [q for q in queries if len(q) >= 2][:20]

        if not queries:
            return jsonify({"error": "No valid queries provided."}), 400

        try:
            results = [_run_search(q) for q in queries]
            return jsonify({"batch_results": results, "count": len(results)})
        except Exception as error:
            print("Batch search error:", error)
            return jsonify({"error": "Batch search could not be completed."}), 500

    @app.post("/api/export")
    @require_api_key
    def export():
        """Export the last search results as JSON, CSV, or TXT."""
        if not _model_ready():
            return jsonify({"error": "The search model is not ready yet."}), 503

        payload = request.get_json(silent=True) or {}
        query = normalize_query(payload.get("query"))
        fmt = str(payload.get("format", "json")).lower()

        if not query or len(query) < 2:
            return jsonify({"error": "Provide a valid query."}), 400
        if fmt not in {"json", "csv", "txt"}:
            return jsonify({"error": "Format must be json, csv, or txt."}), 400

        try:
            data = _run_search(query)
            results = data["results"]

            if fmt == "json":
                return Response(
                    __import__("json").dumps(data, indent=2),
                    mimetype="application/json",
                    headers={"Content-Disposition": f'attachment; filename="medmatch_{query[:30]}.json"'},
                )

            if fmt == "csv":
                buf = io.StringIO()
                writer = csv.DictWriter(
                    buf,
                    fieldnames=["rank", "term", "category", "description", "relation", "score", "percentage", "badge"],
                )
                writer.writeheader()
                for i, r in enumerate(results, 1):
                    writer.writerow({
                        "rank": i, "term": r["term"], "category": r["category"],
                        "description": r["description"], "relation": r["relation"],
                        "score": r["score"], "percentage": r["percentage"], "badge": r["badge"],
                    })
                return Response(
                    buf.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="medmatch_{query[:30]}.csv"'},
                )

            # txt
            lines = [f"MedMatch Export — Query: {query}\n{'='*50}"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"\n#{i}  {r['term']}  ({r['percentage']}%)\n"
                    f"    Category : {r['category']}\n"
                    f"    Relation : {r['relation']}\n"
                    f"    Badge    : {r['badge']}\n"
                    f"    Desc     : {r['description']}"
                )
            return Response(
                "\n".join(lines),
                mimetype="text/plain",
                headers={"Content-Disposition": f'attachment; filename="medmatch_{query[:30]}.txt"'},
            )

        except Exception as error:
            print("Export error:", error)
            return jsonify({"error": "Export failed."}), 500

    @app.get("/api/suggest")
    def suggest():
        """Return up to 8 term suggestions matching a prefix (for autocomplete)."""
        prefix = normalize_query(request.args.get("q", "")).casefold()
        if len(prefix) < 2:
            return jsonify({"suggestions": []})
        matches = [t for t in state["terms"] if t.casefold().startswith(prefix)][:8]
        return jsonify({"suggestions": matches})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
