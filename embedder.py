import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import boto3
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
    "database": os.getenv("DB"),
}

ENGINE_URL = (
    f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

engine = create_engine(ENGINE_URL, pool_pre_ping=True, pool_recycle=3600)

AWS_REGION = os.getenv("AWS_REGION")
VECTOR_BUCKET = os.getenv("S3_VECTOR_BUCKET")
VECTOR_INDEX = os.getenv("S3_VECTOR_INDEX")

s3vectors = boto3.client("s3vectors", region_name=AWS_REGION)

# -----------------------------
# BUILD MOVIE DOCUMENT
# -----------------------------
def build_movie_document(row):
    parts = []

    if pd.notna(row.title) and row.title.strip():
        parts.append(row.title)

    if pd.notna(row.director) and row.director.strip():
        parts.append(f"Directed by {row.director}.")

    if pd.notna(row.studio) and row.studio.strip():
        parts.append(f"Studios: {row.studio}.")

    if pd.notna(row.publisher) and row.publisher.strip():
        parts.append(f"Published by {row.publisher}.")

    if pd.notna(row.genre) and row.genre.strip():
        parts.append(f"Genres: {row.genre}.")

    if pd.notna(row.plot) and row["plot"].strip():
        parts.append(f"Plot: {row["plot"]}")

    return " ".join(parts)

# -----------------------------
# UPLOAD EMBEDDINGS
# -----------------------------
def upload_vectors(df, embeddings, batch_size=400):
    records = []

    for movie_id, embedding in zip(df.index, embeddings):
        records.append({
            "id": str(movie_id),
            "vector": embedding.tolist(),
            "metadata": {
                "title": df.loc[movie_id, "title"],
                "director": df.loc[movie_id, "director"],
            }
        })

        if len(records) == batch_size:
            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET,
                indexName=VECTOR_INDEX,
                vectors=records
            )
            records = []

    if records:
        s3vectors.put_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            vectors=records
        )

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Retrieving data from RDS...")
    query = """
        SELECT movie_id, title, director, studio, publisher, genre, plot
        FROM movies
    """
    df = pd.read_sql(query, engine)
    df.set_index("movie_id", inplace=True)
    print(f"Retrieved {len(df)} movies")

    print("Building documents...")
    docs = df.apply(build_movie_document, axis=1).tolist()

    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    print("Generating embeddings...")
    embeddings = model.encode(
        docs,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("Uploading embeddings to S3 Vectors...")
    upload_vectors(df, embeddings)

    print("Done! Vectors stored in S3 Vectors index.")

# -----------------------------
if __name__ == "__main__":
    main()
