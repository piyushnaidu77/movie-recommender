import faiss
import numpy as np

print("Loading embeddings...")
embeddings = np.load("movie_embeddings.npy").astype("float32")
movie_ids = np.load("movie_ids.npy", allow_pickle=True)

# -------------------------
# Normalize embeddings
# -------------------------
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]

print("Building FAISS index...")
index = faiss.IndexFlatIP(dim)   # cosine similarity
index.add(embeddings)

print(f"FAISS index size: {index.ntotal}")

faiss.write_index(index, "movie_faiss.index")
np.save("movie_ids_faiss.npy", movie_ids)

print("Saved:")
print(" - movie_faiss.index")
print(" - movie_ids_faiss.npy")
