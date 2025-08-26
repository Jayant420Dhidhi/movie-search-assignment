# 🎬 Movie Semantic Search

This repository showcases a **semantic search system for movies** using **plot descriptions**. By leveraging **sentence embeddings**, this project allows users to search for movies based on natural language queries and retrieve the most relevant titles with their plots.

---

## 🚀 Features

-   **Semantic search**: Find movies not just by keywords, but by understanding the meaning of your query.
-   **Sentence embeddings**: Utilizes state-of-the-art `Sentence Transformers` to encode plot descriptions.
-   **Interactive exploration**: Includes a Jupyter notebook for testing and exploring the search functionality.
-   **Extensible & modular**: Easily add more movies or enhance the search logic.
-   **Unit tested**: Comes with automated tests to ensure reliability and correctness.

---

## 📂 Repository Structure

| File/Folder                 | Description                                                              |
| --------------------------- | ------------------------------------------------------------------------ |
| `movie_search.py`           | Core Python module for movie search using embeddings and similarity scoring. |
| `movies.csv`                | Sample dataset containing movie titles and plot descriptions.            |
| `requirements.txt`          | Python dependencies required to run the project.                         |
| `tests/test_movie_search.py`| Unit tests validating search functionality.                              |
| `solution.ipynb`            | Jupyter notebook for interactive exploration and demonstration.          |

---

## ⚙️ Setup Instructions

1.  **Clone the repository**

    ```bash
    git clone <your-repo-url>
    cd <repo-directory>
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebook**

    Open `solution.ipynb` in Jupyter or VS Code to explore the search system interactively.

4.  **Run tests**

    Verify everything works correctly:

    ```bash
    python -m unittest discover -s tests -p "test_*.py" -v
    ```

---

## 🛠️ Usage

Use the `search_movies` function in `movie_search.py` to find the top N movies matching your query:

```python
from movie_search import search_movies

# Search for movies related to a spy thriller set in Paris
results = search_movies("spy thriller in Paris", top_n=3)
print(results)
```

**Output:** A `pandas.DataFrame` containing the following columns:

-   `title`: Movie title
-   `plot`: Movie plot description
-   `similarity`: Semantic similarity score

---

## 📦 Requirements

-   Python 3.8+
-   `pandas`
-   `sentence-transformers`
-   `scikit-learn`
-   `torch`

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 💡 How It Works

1.  **Embedding Plots**: Each movie plot is converted into a high-dimensional vector using a pre-trained `Sentence Transformer` model.
2.  **Query Embedding**: User queries are also transformed into vectors using the same model.
3.  **Similarity Search**: The system computes the cosine similarity between the query vector and all movie plot vectors.
4.  **Top Results**: It returns the top N movies with the highest similarity scores.

This allows you to find movies even if your query does not contain the exact words from the plot, making the search smarter and more human-like.

---

## 📚 Learning & Exploration

-   Explore the Jupyter notebook to see embeddings and similarity scores in action.
-   Modify or extend the dataset in `movies.csv` for larger-scale experiments.
-   Try different pre-trained models from the `Sentence Transformers` library.

---

## 📝 License

This project is for educational purposes only.
