import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------

load_dotenv()

MYSQL_CONFIG = {
    "user": os.getenv("USER"),
    "password": os.getenv("PASSWORD"),
    "host": os.getenv("HOST"),
    "database": os.getenv("DATABASE")
}

engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
)

# -----------------------------
# BUILD MOVIE DOCUMENT
# -----------------------------
def build_movie_document(row):
    """
    Combine relevant metadata fields into a natural-language
    document suitable for transformer embedding.
    """
    parts = []

    # Title
    if pd.notna(row.title) and row.title.strip() != "":
        parts.append(f"{row.title}")

    # Director
    if pd.notna(row.director) and row.director.strip() != "":
        parts.append(f"Directed by {row.director}.")

    # Studios
    if pd.notna(row.studio) and row.studio.strip() != "":
        parts.append(f"Studios: {row.studio}.")

    # Publishers
    if pd.notna(row.publisher) and row.publisher.strip() != "":
        parts.append(f"Published by {row.publisher}.")

    # Genres
    if pd.notna(row.genre) and row.genre.strip() != "":
        parts.append(f"Genres: {row.genre}.")

    # Plot
    if pd.notna(row["plot"]) and row["plot"].strip() != "":
        parts.append(f"Plot: {row["plot"]}")

    return " ".join(parts)

# -----------------------------
# MAIN
# -----------------------------
def main():
    # ---- Load movies from MySQL ----
    query = "SELECT movie_id, title, director, studio, publisher, genre, plot FROM movies"
    df = pd.read_sql(query, engine)

    # Use movie_id as index
    df.set_index("movie_id", inplace=True)

    # ---- Build documents ----
    docs = df.apply(build_movie_document, axis=1).tolist()

    # ---- Load embedding model ----
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    # ---- Encode documents ----
    print("Generating embeddings...")
    embeddings = model.encode(
        docs,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # ---- Save outputs ----
    print("Saving embeddings...")
    np.save("movie_embeddings.npy", embeddings)

    # Save movie_id order so you can reference later
    movie_ids = df.index.values
    np.save("movie_ids.npy", movie_ids)

    print("Done! Saved:")
    print(" - movie_embeddings.npy")
    print(" - movie_ids.npy")

# -----------------------------
if __name__ == "__main__":
    main()
