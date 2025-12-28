import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from surprise import Dataset, Reader, SVD
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
# LOAD RATINGS FROM MYSQL
# -----------------------------
query = """
SELECT
    r.user_id,
    r.movie_id,
    r.rating
FROM ratings r
"""

df = pd.read_sql(query, engine)

print(f"Loaded {len(df)} ratings")

# -----------------------------
# PREPARE SURPRISE DATASET
# -----------------------------
reader = Reader(rating_scale=(0, 5))

data = Dataset.load_from_df(
    df[["user_id", "movie_id", "rating"]],
    reader
)

trainset = data.build_full_trainset()

# -----------------------------
# TRAIN SVD MODEL
# -----------------------------
model = SVD(
    n_factors=100,
    n_epochs=30,
    biased=True,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("Training SVD model...")
model.fit(trainset)

pickle.dump(model, open("svd_model.pkl", "wb"))

# -----------------------------
# EXTRACT LATENT FACTORS
# -----------------------------
user_factors = model.pu        # shape: [n_users, n_factors]
movie_factors = model.qi       # shape: [n_movies, n_factors]

np.save("user_factors.npy", user_factors)
np.save("movie_factors.npy", movie_factors)

# -----------------------------
# SAVE ID INDEX MAPS
# -----------------------------
# Surprise uses internal indexing → we must save mappings
user_index_map = {trainset.to_raw_uid(i): i for i in trainset.all_users()}
movie_index_map = {trainset.to_raw_iid(i): i for i in trainset.all_items()}

np.save("user_index_map.npy", user_index_map)
np.save("movie_index_map.npy", movie_index_map)

# -----------------------------
# SUMMARY
# -----------------------------
print("✅ CF training complete")
print(f"Users: {user_factors.shape[0]}")
print(f"Movies: {movie_factors.shape[0]}")
print("Saved:")
print(" - svd_model.pkl")
print(" - user_factors.npy")
print(" - movie_factors.npy")
print(" - user_index_map.npy")
print(" - movie_index_map.npy")
