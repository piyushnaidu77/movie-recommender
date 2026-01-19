import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
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
    "port": int(os.getenv("PORT", 3306)),
    "database": os.getenv("DB")
}

ENGINE_URL = (
    f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# -----------------------------
# LOAD MODELS & ARRAYS
# -----------------------------
print("Loading embeddings...")
movie_embeddings = np.load("movie_embeddings.npy").astype("float32")
movie_ids = np.load("movie_ids.npy")

movieId_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

print("Loading SVD model...")
svd_model = pickle.load(open("svd_model.pkl", "rb"))

# -----------------------------
# LOAD DB TABLES
# -----------------------------
movies_df = pd.read_sql("SELECT movie_id, title FROM movies", engine)
users_df = pd.read_sql("SELECT user_id, username FROM users", engine)
ratings_df = pd.read_sql("SELECT user_id, movie_id FROM ratings", engine)

username_to_id = dict(zip(users_df.username, users_df.user_id))
movie_id_to_title = dict(zip(movies_df.movie_id, movies_df.title))

# -----------------------------
# HYBRID RECOMMENDER (NO FAISS)
# -----------------------------
def hybrid_recommend(
    user_name,
    top_k=10,
    alpha=0.6
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

    user_embedding = user_vectors.mean(axis=0).reshape(1, -1)

    # -------------------------
    # COSINE SIMILARITY
    # -------------------------
    content_scores = cosine_similarity(
        user_embedding,
        movie_embeddings
    ).flatten()

    # Remove already-rated movies
    for mid in rated_movies:
        content_scores[movieId_to_idx[mid]] = -1

    # -------------------------
    # CF SCORES
    # -------------------------
    cf_scores = np.array([
        svd_model.predict(user_id, mid).est
        for mid in movie_ids
    ])

    for mid in rated_movies:
        cf_scores[movieId_to_idx[mid]] = -1

    # -------------------------
    # NORMALIZATION (CRITICAL)
    # -------------------------
    print("CF std:", np.std(cf_scores))
    print("Content std:", np.std(content_scores))

    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.ptp() + 1e-8)
    content_norm = (content_scores - content_scores.min()) / (content_scores.ptp() + 1e-8)

    hybrid_scores = alpha * cf_norm + (1 - alpha) * content_norm

    # -------------------------
    # TOP-K
    # -------------------------
    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]

    return pd.DataFrame({
        "movie_id": movie_ids[top_idx],
        "title": [movie_id_to_title[mid] for mid in movie_ids[top_idx]],
        "hybrid_score": hybrid_scores[top_idx]
    })


# -----------------------------
# EXAMPLE
# -----------------------------
if __name__ == "__main__":
    recs = hybrid_recommend("silentdawn", top_k=10, alpha=0.6)
    print(recs)
