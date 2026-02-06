import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

DATA_DIR = "data"
TOP_K = 10
TEST_RATIO = 0.2

print("Loading data...")

movies_df = pd.read_parquet(f"{DATA_DIR}/movies.parquet")
ratings_df = pd.read_parquet(f"{DATA_DIR}/ratings.parquet")
users = ratings_df.user_id.unique()

movie_embeddings = np.load(f"{DATA_DIR}/movie_embeddings.npy").astype("float32")
movie_ids = np.load(f"{DATA_DIR}/movie_ids.npy")
movieId_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

svd_model = pickle.load(open("data/svd_model.pkl", "rb"))

sorted_users = np.argsort(users)


# --------------------------------------------------
# Helper
# --------------------------------------------------
def normalize(x):
    return (x - x.min()) / (x.ptp() + 1e-8)

def precision_recall(hit_set, recs, k):
    hits = len(set(recs[:k]) & hit_set)
    precision = hits / k
    recall = hits / len(hit_set)
    return precision, recall, hits > 0

def report(name, prec, rec, hit):
    print(f"\n{name}")
    print("Precision@10:", np.mean(prec))
    print("Recall@10:", np.mean(rec))
    print("HitRate@10:", np.mean(hit))

def evaluation(alpha, n):
    emb_prec, emb_rec, emb_hit = [], [], []
    cf_prec, cf_rec, cf_hit = [], [], []
    hy_prec, hy_rec, hy_hit = [], [], []

    for user_id in tqdm(sorted_users[:n]):
        user_ratings = ratings_df[ratings_df.user_id == user_id]

        if len(user_ratings) < 10:
            continue

        # Split train/test
        test = user_ratings.sample(frac=TEST_RATIO, random_state=42)
        train = user_ratings.drop(test.index)

        test_movies = set(test.movie_id.values)

        # Only consider movies present in embeddings
        train_idx = train.movie_id.map(movieId_to_idx).dropna().astype(int).values
        if len(train_idx) == 0:
            continue

        # ---------------- EMBEDDING MODEL ----------------
        user_embedding = movie_embeddings[train_idx].mean(axis=0, keepdims=True)
        emb_scores = cosine_similarity(user_embedding, movie_embeddings).ravel()
        emb_scores[train_idx] = -1
        emb_top = np.argsort(emb_scores)[::-1][:TOP_K]
        emb_recs = movie_ids[emb_top]

        p, r, h = precision_recall(test_movies, emb_recs, TOP_K)
        emb_prec.append(p); emb_rec.append(r); emb_hit.append(h)

        # ---------------- CF MODEL ----------------
        cf_scores = np.array([
            svd_model.predict(user_id, mid).est
            for mid in movie_ids
        ])
        cf_scores[train_idx] = -1
        cf_top = np.argsort(cf_scores)[::-1][:TOP_K]
        cf_recs = movie_ids[cf_top]

        p, r, h = precision_recall(test_movies, cf_recs, TOP_K)
        cf_prec.append(p); cf_rec.append(r); cf_hit.append(h)

        # ---------------- HYBRID ----------------
        hy_scores = alpha * normalize(cf_scores) + \
                    (1 - alpha) * normalize(emb_scores)
        hy_top = np.argsort(hy_scores)[::-1][:TOP_K]
        hy_recs = movie_ids[hy_top]

        p, r, h = precision_recall(test_movies, hy_recs, TOP_K)
        hy_prec.append(p); hy_rec.append(r); hy_hit.append(h)

    print("Alpha:",alpha)
    report("Embeddings Only", emb_prec, emb_rec, emb_hit)
    report("Collaborative Filtering Only", cf_prec, cf_rec, cf_hit)
    report("Hybrid Model", hy_prec, hy_rec, hy_hit)
    print("----------------------------------------------")

for a in [0.2, 0.4, 0.6, 0.8]:
    evaluation(a, 100)