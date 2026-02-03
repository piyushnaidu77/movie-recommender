# Hybrid Movie Recommender — Serverless REST API on AWS Lambda
A production-grade hybrid movie recommendation system combining:

- Collaborative Filtering (SVD)
- Content-Based Filtering (Embeddings + Cosine Similarity)
- Fully serverless inference on AWS Lambda
- REST API via API Gateway

This project demonstrates how to deploy a real ML recommender as a high-performance inference API using only precomputed artifacts.

## Architecture
Client → API Gateway → AWS Lambda (Docker) → In-memory model → JSON response

## Hybrid Recommendation Logic
Final score is a weighted combination:
Hybrid Score = α * CF_score + (1 - α) * Content_score

Where:

- CF_score: SVD predicted rating for the user/movie
- Content_score: Cosine similarity between user embedding and movie embedding
- α: Weight parameter (default 0.6)

## Project Structure
```
.
├── app.py
├── Dockerfile
├── requirements.txt
├── data/
│   ├── movies.parquet
│   ├── users.parquet
│   ├── ratings.parquet
│   ├── movie_embeddings.npy
│   ├── movie_ids.npy
│   └── svd_model.pkl
└── README.md
```

## Local Testing
```bash
pip install -r requirements.txt
python app.py
```

## 🌐 API Usage

### Endpoint
```
GET /recommend
```

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `user` | string | required | Username |
| `top_k` | int | 10 | Number of recommendations |
| `alpha` | float | 0.6 | CF vs Content weight |

### Example
```
https://<api-id>.execute-api.us-east-1.amazonaws.com/recommend?user=silentdawn&top_k=10&alpha=0.6
Response
json[
  {
    "movie_id": 123,
    "title": "Inception",
    "year": 2010,
    "poster": "url",
    "hybrid_score": 0.9821
  }
]
```

## Future Improvements

- Cache recommendations per user (Redis / DynamoDB)
- Store data in S3 and download on cold start
- Add authentication
- Add pagination
- Batch recommendation endpoint
