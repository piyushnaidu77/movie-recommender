import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
import faiss
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
# LOAD MODELS & ARRAYS
# -----------------------------
print("Loading embeddings...")
movie_embeddings = np.load("movie_embeddings.npy")
movie_ids = np.load("movie_ids.npy")   # MySQL movie_id order

print("Loading FAISS index...")
faiss_index = faiss.read_index("movie_faiss.index")
faiss_movie_ids = np.load("movie_ids_faiss.npy", allow_pickle=True)

movieId_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

print("Loading SVD model...")
svd_model = pickle.load(open("svd_model.pkl", "rb"))

# -----------------------------
# LOAD DB TABLES
# -----------------------------
movies_df = pd.read_sql(
    "SELECT movie_id, title FROM movies",
    engine
)

users_df = pd.read_sql(
    "SELECT user_id, username FROM users",
    engine
)

ratings_df = pd.read_sql(
    "SELECT user_id, movie_id, rating FROM ratings",
    engine
)

username_to_id = dict(zip(users_df.username, users_df.user_id))
movie_id_to_title = dict(zip(movies_df.movie_id, movies_df.title))

# -----------------------------
# HYBRID RECOMMENDER
# -----------------------------
def hybrid_recommend(
    user_name,
    top_k=10,
    alpha=0.6,
    faiss_k=200
):
    user_id = username_to_id[user_name]

    # Movies user already rated
    rated_movies = ratings_df.loc[
        ratings_df["user_id"] == user_id, "movie_id"
    ].tolist()

    rated_movies = [m for m in rated_movies if m in movieId_to_idx]

    if len(rated_movies) == 0:
        print("Cold start user")
        return pd.DataFrame()

    # -------------------------
    # USER EMBEDDING
    # -------------------------
    user_vectors = np.array([
        movie_embeddings[movieId_to_idx[mid]]
        for mid in rated_movies
    ])

    user_embedding = user_vectors.mean(axis=0).astype("float32")
    faiss.normalize_L2(user_embedding.reshape(1, -1))

    # -------------------------
    # FAISS SEARCH
    # -------------------------
    scores, indices = faiss_index.search(
        user_embedding.reshape(1, -1),
        faiss_k
    )

    candidate_movies = [
        faiss_movie_ids[idx]
        for idx in indices[0]
        if faiss_movie_ids[idx] not in rated_movies
    ]

    # -------------------------
    # CF SCORES (correct raw IDs)
    # -------------------------
    cf_scores = np.array([
        svd_model.predict(user_id, mid).est
        for mid in candidate_movies
    ])

    # -------------------------
    # CONTENT SCORES (already cosine)
    # -------------------------
    content_scores = scores[0][:len(candidate_movies)]

    # -------------------------
    # HYBRID
    # -------------------------
    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.ptp() + 1e-8)
    content_norm = (content_scores - content_scores.min()) / (content_scores.ptp() + 1e-8)

    hybrid_scores = alpha * cf_norm + (1 - alpha) * content_norm

    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]

    return pd.DataFrame({
        "movie_id": [candidate_movies[i] for i in top_idx],
        "title": [movie_id_to_title[candidate_movies[i]] for i in top_idx],
        "hybrid_score": hybrid_scores[top_idx]
    })


# -----------------------------
# EXAMPLE
# -----------------------------
if __name__ == "__main__":
    recs = hybrid_recommend("kurstboy", top_k=10, alpha=0.6)
    print(recs)
