import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# Load movies.csv from the same directory as this script
try:
    csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")
    df = pd.read_csv(csv_path)  # Read CSV into a pandas DataFrame
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find 'movies.csv' at path: {csv_path}")
except pd.errors.EmptyDataError:
    raise ValueError(f"The file 'movies.csv' is empty or invalid.")
except Exception as e:
    raise RuntimeError(f"Error loading CSV: {e}")

# Load a pre-trained sentence embedding model from Sentence Transformers
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")

# Encode all movie plots into embeddings
try:
    embeddings = model.encode(
        df["plot"].tolist(),  # Convert all movie plots into a list
        convert_to_numpy=True,  # Return a numpy array
        normalize_embeddings=True  # Normalize embeddings for cosine similarity
    )
except KeyError:
    raise KeyError("The CSV file must contain a 'plot' column.")
except Exception as e:
    raise RuntimeError(f"Error encoding movie plots: {e}")


def search_movies(query, top_n=5):
    """
    Search for movies similar to a query using sentence embeddings.
    
    Parameters:
    - query: str, the search query (e.g., "space adventure")
    - top_n: int, number of top results to return (default 5)
    
    Returns:
    - DataFrame with the top_n most similar movies, including a 'similarity' score
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    try:
        # Encode the query into an embedding
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Compute cosine similarity between query and all movie plots
        sims = util.cos_sim(q_emb, embeddings).cpu().numpy().flatten()
        
        # Get indices of top_n most similar movies
        idx = sims.argsort()[::-1][:top_n]
        
        # Return top_n movies with similarity score
        return df.iloc[idx].assign(similarity=sims[idx])
    except Exception as e:
        raise RuntimeError(f"Error searching movies: {e}")
