"""Simple Tkinter interface for medical terminology similarity search."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.data_loader import load_medical_terms
from src.embedding_model import create_embeddings, load_embedding_model
from src.similarity_engine import build_faiss_index, search_similar_terms
from src.tfidf_comparison import find_tfidf_matches


DATASET_PATH = Path(__file__).parent / "data" / "medical_terms.csv"


class SimilarityApp:
    """Small desktop interface for the existing similarity engine."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Medical Terminology Similarity")
        self.root.geometry("760x620")
        self.root.minsize(620, 480)

        self.model = None
        self.faiss_index = None
        self.terms: list[str] = []

        self._build_widgets()
        self._load_resources_in_background()

    def _build_widgets(self) -> None:
        """Create the simple search form and results area."""
        main_frame = ttk.Frame(self.root, padding=24)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        ttk.Label(
            main_frame,
            text="Medical Terminology Similarity",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            main_frame,
            text="Find related medical terms using Sentence-BERT and FAISS.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 20))

        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=2, column=0, sticky="ew")
        search_frame.columnconfigure(0, weight=1)

        self.query_entry = ttk.Entry(search_frame, font=("Segoe UI", 12))
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.query_entry.bind("<Return>", lambda _event: self.search())
        self.query_entry.focus()

        self.search_button = ttk.Button(
            search_frame, text="Search", command=self.search, state="disabled"
        )
        self.search_button.grid(row=0, column=1)

        self.status_label = ttk.Label(
            main_frame, text="Loading model and medical terms..."
        )
        self.status_label.grid(row=3, column=0, sticky="w", pady=(12, 10))

        self.results_text = tk.Text(
            main_frame,
            wrap="word",
            font=("Consolas", 11),
            padx=12,
            pady=12,
            state="disabled",
        )
        self.results_text.grid(row=4, column=0, sticky="nsew")

    def _load_resources_in_background(self) -> None:
        """Load the model without freezing the window."""
        threading.Thread(target=self._load_resources, daemon=True).start()

    def _load_resources(self) -> None:
        try:
            medical_data = load_medical_terms(DATASET_PATH)
            terms = medical_data["term"].tolist()
            model = load_embedding_model()
            embeddings = create_embeddings(model, terms)
            faiss_index = build_faiss_index(embeddings)
            self.root.after(0, self._resources_loaded, model, terms, faiss_index)
        except Exception as error:  # Show setup errors inside the beginner-friendly UI.
            self.root.after(0, self._show_load_error, error)

    def _resources_loaded(self, model, terms, faiss_index) -> None:
        self.model = model
        self.terms = terms
        self.faiss_index = faiss_index
        self.search_button.configure(state="normal")
        self.status_label.configure(text=f"Ready. Loaded {len(terms)} medical terms.")

    def _show_load_error(self, error: Exception) -> None:
        self.status_label.configure(text="Could not load the project resources.")
        messagebox.showerror("Startup error", str(error))

    def search(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty input", "Please enter a medical term.")
            return
        if self.model is None:
            messagebox.showinfo("Please wait", "The model is still loading.")
            return

        self.search_button.configure(state="disabled")
        self.status_label.configure(text="Searching...")
        threading.Thread(target=self._run_search, args=(query,), daemon=True).start()

    def _run_search(self, query: str) -> None:
        try:
            query_embedding = create_embeddings(self.model, [query])
            semantic_results = search_similar_terms(
                self.faiss_index, query_embedding, self.terms
            )
            tfidf_results = find_tfidf_matches(self.terms, query)
            self.root.after(
                0, self._display_results, query, semantic_results, tfidf_results
            )
        except Exception as error:
            self.root.after(0, self._show_search_error, error)

    def _display_results(self, query, semantic_results, tfidf_results) -> None:
        output = [f"Query: {query}", "", "Sentence-BERT + FAISS results:"]
        output.extend(
            f"{rank}. {term} - Similarity: {score:.4f}"
            for rank, (term, score) in enumerate(semantic_results, start=1)
        )
        output.extend(["", "TF-IDF + cosine similarity baseline:"])
        output.extend(
            f"{rank}. {term} - Similarity: {score:.4f}"
            for rank, (term, score) in enumerate(tfidf_results, start=1)
        )

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", "\n".join(output))
        self.results_text.configure(state="disabled")
        self.status_label.configure(text="Search complete.")
        self.search_button.configure(state="normal")

    def _show_search_error(self, error: Exception) -> None:
        self.status_label.configure(text="Search failed.")
        self.search_button.configure(state="normal")
        messagebox.showerror("Search error", str(error))


def main() -> None:
    root = tk.Tk()
    SimilarityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
