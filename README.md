# Semantic Similarity-Based Medical Terminology Matching Using Transformer Models

## Project Overview

This college-level NLP micro project finds the five closest medical terms for a typed phrase. It uses Sentence-BERT embeddings and FAISS for semantic matching, then compares the result with TF-IDF and cosine similarity. The project includes a small Flask dashboard while preserving the original CLI and Tkinter versions.

## Problem Statement

Exact keyword matching can miss useful relationships such as `kidney failure` and `renal failure`. This project demonstrates how Transformer embeddings can match terminology by meaning rather than only by shared spelling.

## Objectives

- Build a small, understandable semantic-search system.
- Compare Transformer similarity with traditional TF-IDF similarity.
- Display scores, categories, descriptions, and relation labels.
- Provide a polished browser interface suitable for a project demonstration.

## Features

- 136 medical terminology records across 10 categories.
- Sentence-BERT model: `all-MiniLM-L6-v2`.
- Normalized embeddings and FAISS inner-product search.
- Top five results with percentage scores and progress bars.
- Metadata cards with category, description, and relation type.
- TF-IDF comparison table and score visualization.
- Empty, very short, missing-data, and model-error handling.
- Model and dataset embeddings loaded once when the Flask app starts.
- Clear search, example queries, responsive layout, and light/dark mode.
- Educational disclaimer; this is not a diagnostic tool.

## Technologies

Python, Flask, HTML, CSS, JavaScript, Pandas, NumPy, Sentence-Transformers, Transformers, Scikit-learn, and FAISS.

## Project Structure

```text
medical_terminology_similarity/
├── app.py                         # Flask backend and JSON API
├── main.py                        # Original terminal application
├── ui.py                          # Optional Tkinter desktop UI
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── medical_terms.csv          # 136 terms and metadata
├── src/
│   ├── data_loader.py             # CSV validation and normalization
│   ├── embedding_model.py         # Sentence-BERT loading and encoding
│   ├── similarity_engine.py       # FAISS index and search
│   └── tfidf_comparison.py        # Traditional baseline
├── templates/
│   └── index.html                 # Flask dashboard markup
├── static/
    ├── css/style.css              # Responsive visual design
    └── js/script.js               # Search, cards, stats, and chart
├── docs/                          # GitHub Pages static frontend
│   ├── index.html
│   ├── style.css
│   └── script.js                  # Configurable Render API client
```

## GitHub Pages Frontend and Render Backend

The deployable architecture is:

- **Frontend:** GitHub Pages (`docs/`)
- **Backend:** Render (Flask API)
- **ML Model:** Sentence-BERT
- **Vector Search:** FAISS
- **Traditional NLP:** TF-IDF

The GitHub Pages frontend sends requests to the Render backend. Set `API_BASE_URL` near the top of `docs/script.js` to the public URL of your Render service if it differs from the default value.

### Enable GitHub Pages

1. Push the repository to GitHub.
2. Open **GitHub Repository -> Settings -> Pages**.
3. Under **Build and deployment**, choose **Deploy from branch**.
4. Select branch **main** and folder **/docs**.
5. Click **Save**.

GitHub will provide a URL such as `https://USERNAME.github.io/REPOSITORY-NAME/`. The frontend uses relative asset paths, so it works from a repository subpath.

### Test the deployed website

1. Open the GitHub Pages URL after the deployment finishes.
2. Confirm the status message says the backend is ready.
3. Search for `heart attack` or use one of the example buttons.
4. Confirm semantic results, similarity scores, badges, categories, descriptions, related terms, relation types, statistics, and the TF-IDF comparison appear.
5. If the request fails, open the browser developer console and verify that `API_BASE_URL` matches the Render service URL and that `/api/health` responds.

The Flask API enables CORS for `/api/*` so the GitHub Pages origin can call `/api/health` and `POST /api/search`. After pulling this change into Render, redeploy the backend so `Flask-Cors` is installed from `requirements.txt`.

## System Architecture

```text
User enters a term
        |
        v
Flask web dashboard (HTML/CSS/JavaScript)
        |
        v
POST /api/search
        |
        v
Normalize input -> Sentence-BERT embedding
        |                         |
        v                         v
FAISS top-5 search          TF-IDF baseline
        |                         |
        +------------+------------+
                     v
       JSON results, scores, metadata, and statistics
```

## Installation on Windows

In the VS Code terminal:

```powershell
cd "c:\Users\Appaji Maranur\OneDrive\Desktop\medical_terminology_similarity"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the environment is already present, run:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Select `.venv` through **Python: Select Interpreter** in VS Code. The first run downloads the Sentence-BERT model and may take longer.

## How to Run

### Flask dashboard

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Use the search box, example buttons, result cards, comparison table, and chart.

### Terminal version

```powershell
python main.py
```

### Optional Tkinter version

```powershell
python ui.py
```

## Deploy on Render

The repository includes `render.yaml`, so Render can configure the service automatically.

1. Push the project to GitHub.
2. In Render, choose **New +** and then **Blueprint**.
3. Connect `appumaranur/medical-terminology-similarity`.
4. Confirm the service from `render.yaml` and click **Apply**.

Render will install `requirements.txt`, start Gunicorn, bind the service to its assigned `PORT`, and use `/api/health` for health checks. The first deployment downloads and loads the Sentence-BERT model, so the initial startup can take several minutes. The free plan may sleep when idle and reload the model on the next request cycle.

If you create the service manually instead, use:

```text
Runtime: Python 3
Build command: pip install -r requirements.txt
Start command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 app:app
Health check path: /api/health
```

This project is intentionally configured with one worker because each worker would load a separate copy of the Transformer model into memory.

## Dataset

`data/medical_terms.csv` contains 136 unique terms in categories such as Cardiology, Neurology, Respiratory, Endocrinology, Nephrology, Gastroenterology, Infectious Diseases, Orthopedics, Urology, and Common Conditions. Each row contains:

- `term`: the medical phrase.
- `category`: the broad medical area.
- `description`: a short educational explanation.
- `related_terms`: pipe-separated related or equivalent phrases.
- `relation_type`: for example, Primary term, Equivalent term, Related term, or Symptom.

## Workflow

1. Flask loads and validates the CSV.
2. The model converts every dataset term into a normalized embedding once.
3. FAISS stores those vectors in an inner-product index.
4. A user query is whitespace-normalized and encoded once.
5. FAISS returns the five highest-scoring terms.
6. Scores are converted to percentages and enriched with CSV metadata.
7. TF-IDF independently ranks the terms for comparison.
8. The frontend displays cards, statistics, a chart, and a comparison table.

## Example

Input: `lung inflammation`

The semantic results may include `lung infection`, `inflammation of the bronchi`, `bronchitis`, and `pneumonia`, even though the exact input is not in the dataset. Exact scores can vary slightly by package version.

## Key Concepts

- **Sentence-BERT:** A pretrained Transformer model that creates comparable fixed-length vectors for phrases and sentences.
- **Embeddings:** Numeric vectors representing the language meaning and context of text.
- **FAISS:** A library for fast vector search. This project uses exact inner-product search because normalized vectors make inner product equivalent to cosine similarity.
- **Cosine similarity:** Measures the angle between two vectors. A score near 1 means the vectors point in a similar direction.
- **TF-IDF:** A traditional representation that gives higher weight to informative words. It mainly benefits from exact word overlap.
- **Why Transformers help:** Sentence-BERT can connect paraphrases and synonyms that do not share the same words, while TF-IDF may score them lower.

## Transformer vs Traditional NLP

Sentence-BERT is better for contextual meaning and synonym matching. TF-IDF is simpler, faster, and easier to interpret, but it mostly sees words rather than meaning. The dashboard shows both methods for an easy project demonstration.

## Limitations and Future Enhancements

- The dataset is small and educational, not a clinical terminology authority.
- `all-MiniLM-L6-v2` is a general-purpose model rather than a biomedical-specific model.
- Scores show text similarity, not clinical relevance or diagnosis.
- Future work could add evaluation labels, definitions from a trusted source, or a biomedical sentence-embedding model.

## Disclaimer

**Educational project only. This system is designed for terminology matching and does not provide medical diagnosis or medical advice.**

## Viva Questions and Short Answers

**Why use Sentence-BERT?** It creates sentence-level embeddings that capture semantic similarity.

**Why normalize embeddings?** Normalization makes FAISS inner product behave like cosine similarity.

**Why use FAISS?** It provides an efficient and simple vector-search index.

**Why compare TF-IDF?** TF-IDF is a familiar traditional baseline and makes the Transformer improvement easier to demonstrate.

**Does the model diagnose patients?** No. It only matches text from a small educational dataset.

**Why load the model once?** Model loading is expensive, so reusing it makes later searches faster.

**What happens for an unknown phrase?** The phrase is embedded and matched by meaning even if it is not an exact CSV row.
