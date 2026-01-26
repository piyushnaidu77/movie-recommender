import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading data...")

movies_df = pd.read_parquet(f"{DATA_DIR}/movies.parquet")
users_df = pd.read_parquet(f"{DATA_DIR}/users.parquet")
ratings_df = pd.read_parquet(f"{DATA_DIR}/ratings.parquet")

svd_model = pickle.load(open("svd_model.pkl", "rb"))

movie_embeddings = np.load(f"{DATA_DIR}/movie_embeddings.npy").astype("float32")
movie_ids = np.load(f"{DATA_DIR}/movie_ids.npy")

# --------------------------------------------------
# PRECOMPUTE ALIGNMENT (CRITICAL SIMPLIFICATION)
# --------------------------------------------------
movies_df = movies_df.set_index("movie_id").loc[movie_ids].reset_index()

username_to_id = dict(zip(users_df.username, users_df.user_id))
movieId_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

# --------------------------------------------------
# HELPER: NORMALIZE
# --------------------------------------------------
def normalize(x):
    return (x - x.min()) / (x.ptp() + 1e-8)

# --------------------------------------------------
# HYBRID RECOMMENDER
# --------------------------------------------------
def hybrid_recommend(user_name, top_k=10, alpha=0.6):
    user_id = username_to_id[user_name]

    # Movies user already rated (indices, not ids)
    rated = ratings_df.loc[
        ratings_df.user_id == user_id, "movie_id"
    ].map(movieId_to_idx).dropna().astype(int).values

    if len(rated) == 0:
        print("Cold start user")
        return pd.DataFrame()

    # --------------------------------------------------
    # USER EMBEDDING (mean of watched)
    # --------------------------------------------------
    user_embedding = movie_embeddings[rated].mean(axis=0, keepdims=True)

    # --------------------------------------------------
    # CONTENT SCORE (vectorized)
    # --------------------------------------------------
    content_scores = cosine_similarity(user_embedding, movie_embeddings).ravel()

    # --------------------------------------------------
    # CF SCORE (vectorized)
    # --------------------------------------------------
    cf_scores = np.array([
        svd_model.predict(user_id, mid).est
        for mid in movie_ids
    ])

    # --------------------------------------------------
    # REMOVE ALREADY RATED (no loops)
    # --------------------------------------------------
    content_scores[rated] = -1
    cf_scores[rated] = -1

    # --------------------------------------------------
    # NORMALIZE + HYBRID
    # --------------------------------------------------
    norm_cf = normalize(cf_scores)
    norm_content = normalize(content_scores)

    hybrid_scores = alpha * norm_cf + (1 - alpha) * norm_content

    # --------------------------------------------------
    # TOP-K
    # --------------------------------------------------
    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]

    result = movies_df.iloc[top_idx][
        ["movie_id", "title", "year", "poster"]
    ].copy()

    result["hybrid_score"] = hybrid_scores[top_idx]

    return result.reset_index(drop=True)


# --------------------------------------------------
# EXAMPLE
# --------------------------------------------------
if __name__ == "__main__":
    print(hybrid_recommend("silentdawn", top_k=10))
    print(hybrid_recommend("kurstboy", top_k=10))
