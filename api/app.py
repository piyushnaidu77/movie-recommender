import os
import numpy as np
import pandas as pd
import pickle
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from mangum import Mangum

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded data
movies_df = None
users_df = None
ratings_df = None
svd_model = None
movie_embeddings = None
movie_ids = None
username_to_id = None
movieId_to_idx = None

S3_BUCKET = os.environ.get("S3_BUCKET", "DEFAULT_BUCKET")
DATA_DIR = "/tmp/data"

def download_from_s3(bucket, key, local_path):
    """Download file from S3 to local path"""
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def load_data():
    """Load all data from S3 on cold start"""
    global movies_df, users_df, ratings_df, svd_model
    global movie_embeddings, movie_ids, username_to_id, movieId_to_idx
    
    print("Loading data from S3...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download files from S3
    download_from_s3(S3_BUCKET, "data/movies.parquet", f"{DATA_DIR}/movies.parquet")
    download_from_s3(S3_BUCKET, "data/users.parquet", f"{DATA_DIR}/users.parquet")
    download_from_s3(S3_BUCKET, "data/ratings.parquet", f"{DATA_DIR}/ratings.parquet")
    download_from_s3(S3_BUCKET, "svd_model.pkl", f"{DATA_DIR}/svd_model.pkl")
    download_from_s3(S3_BUCKET, "data/movie_embeddings.npy", f"{DATA_DIR}/movie_embeddings.npy")
    download_from_s3(S3_BUCKET, "data/movie_ids.npy", f"{DATA_DIR}/movie_ids.npy")
    
    # Load data
    movies_df = pd.read_parquet(f"{DATA_DIR}/movies.parquet")
    users_df = pd.read_parquet(f"{DATA_DIR}/users.parquet")
    ratings_df = pd.read_parquet(f"{DATA_DIR}/ratings.parquet")
    
    svd_model = pickle.load(open(f"{DATA_DIR}/svd_model.pkl", "rb"))
    
    movie_embeddings = np.load(f"{DATA_DIR}/movie_embeddings.npy").astype("float32")
    movie_ids = np.load(f"{DATA_DIR}/movie_ids.npy")
    
    # Precompute alignment
    movies_df = movies_df.set_index("movie_id").loc[movie_ids].reset_index()
    
    username_to_id = dict(zip(users_df.username, users_df.user_id))
    movieId_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    
    print("Data loaded successfully!")

def normalize(x):
    """Normalize array to 0-1 range"""
    return (x - x.min()) / (x.ptp() + 1e-8)

def hybrid_recommend(user_name, top_k=10, alpha=0.6):
    """Generate hybrid recommendations for a user"""
    if user_name not in username_to_id:
        raise ValueError(f"User '{user_name}' not found")
    
    user_id = username_to_id[user_name]
    
    # Movies user already rated
    rated = ratings_df.loc[
        ratings_df.user_id == user_id, "movie_id"
    ].map(movieId_to_idx).dropna().astype(int).values
    
    if len(rated) == 0:
        return pd.DataFrame()
    
    # User embedding (mean of watched)
    user_embedding = movie_embeddings[rated].mean(axis=0, keepdims=True)
    
    # Content score
    content_scores = cosine_similarity(user_embedding, movie_embeddings).ravel()
    
    # CF score
    cf_scores = np.array([
        svd_model.predict(user_id, mid).est
        for mid in movie_ids
    ])
    
    # Remove already rated
    content_scores[rated] = -1
    cf_scores[rated] = -1
    
    # Normalize + hybrid
    norm_cf = normalize(cf_scores)
    norm_content = normalize(content_scores)
    
    hybrid_scores = alpha * norm_cf + (1 - alpha) * norm_content
    
    # Top-K
    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]
    
    result = movies_df.iloc[top_idx][
        ["movie_id", "title", "year", "poster"]
    ].copy()
    
    result["hybrid_score"] = hybrid_scores[top_idx]
    
    return result.reset_index(drop=True)


# Request/Response models
class RecommendRequest(BaseModel):
    username: str
    top_k: int = 10
    alpha: float = 0.6

class MovieResponse(BaseModel):
    movie_id: int
    title: str
    year: int
    poster: str
    hybrid_score: float


@app.on_event("startup")
async def startup_event():
    """Load data when the API starts"""
    load_data()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Movie Recommender API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get movie recommendations for a user"""
    try:
        recommendations = hybrid_recommend(
            request.username, 
            top_k=request.top_k, 
            alpha=request.alpha
        )
        
        if recommendations.empty:
            return {
                "username": request.username,
                "recommendations": [],
                "message": "No recommendations available (cold start user)"
            }
        
        return {
            "username": request.username,
            "recommendations": recommendations.to_dict(orient="records")
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/users")
async def list_users():
    """List all available usernames"""
    return {
        "users": list(username_to_id.keys()),
        "total": len(username_to_id)
    }


# Lambda handler
handler = Mangum(app)