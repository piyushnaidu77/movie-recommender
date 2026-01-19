import pandas as pd
import mysql.connector
import numpy as np
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
    "port": int(os.getenv("PORT", 3306)),
    "database": os.getenv("DB"),
    "ssl_disabled": False
}

ENGINE_URL = (
    f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

engine = create_engine(
    ENGINE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)

# -----------------------------
# LOAD CSVs
# -----------------------------
tmdb = pd.read_csv("data/tmdb_clean.csv")
watch = pd.read_csv("data/watch_clean.csv")

# -----------------------------
# NORMALIZE MOVIE SLUGS
# -----------------------------
tmdb["movie_slug"] = tmdb["moviename"].astype(str)
watch["movie_slug"] = watch["moviename"].astype(str)

# -----------------------------
# AGGREGATE MOVIES
# -----------------------------
all_movies = pd.DataFrame({
    "movie_slug": pd.concat([
        tmdb["movie_slug"],
        watch["movie_slug"]
    ]).dropna().unique()
})

# Left join TMDB metadata (may be missing)
movies = all_movies.merge(
    tmdb.drop(columns=["moviename"]),
    on="movie_slug",
    how="left"
)

# -----------------------------
# FILL SAFE DEFAULTS
# -----------------------------
movies = movies.fillna({
    "title": movies["movie_slug"],
    "year": 0.0,
    "media_type": "movie",
    "tmdb_id": 0.0,
    "director": "Unknown",
    "runtime": 0.0,
    "studio": "Unknown",
    "publisher": "Unknown",
    "genre": "Unknown",
    "plot": "No description available.",
    "poster": ""
})

# -----------------------------
# CONNECT TO MYSQL
# -----------------------------
conn = mysql.connector.connect(**MYSQL_CONFIG)
cursor = conn.cursor()

print("Connection established")

# -----------------------------
# CREATE TABLES
# -----------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    movie_id INT AUTO_INCREMENT PRIMARY KEY,
    movie_slug VARCHAR(255) UNIQUE,
    title VARCHAR(255),
    media_type VARCHAR(25),
    year FLOAT,
    tmdb_id FLOAT,
    director TEXT,
    runtime FLOAT,
    studio TEXT,
    publisher TEXT,
    genre TEXT,
    plot TEXT,
    poster TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ratings (
    user_id INT REFERENCES users(user_id),
    movie_id INT REFERENCES movies(movie_id),
    rating FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id)
)
""")

conn.commit()

print("Created tables")

def chunked(iterable, size=1000):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# -----------------------------
# UPSERT MOVIES
# -----------------------------
movie_sql = """
INSERT INTO movies (
    movie_slug, media_type, tmdb_id, title, year, director,
    runtime, studio, publisher, genre, plot, poster
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
    title=VALUES(title),
    media_type=VALUES(media_type),
    year=VALUES(year),
    tmdb_id=VALUES(tmdb_id),
    director=VALUES(director),
    runtime=VALUES(runtime),
    studio=VALUES(studio),
    publisher=VALUES(publisher),
    genre=VALUES(genre),
    plot=VALUES(plot),
    poster=VALUES(poster)
"""

movie_rows = movies[[
    "movie_slug", "media_type", "tmdb_id", "title", "year",
    "director", "runtime", "studio", "publisher", "genre",
    "plot", "poster"
]].values.tolist()

for i, batch in enumerate(chunked(movie_rows, 1000), start=1):
    cursor.executemany(movie_sql, batch)
    conn.commit()
    print(f"{i * 1000}")


conn.commit()

print("Movie data added")

# -----------------------------
# UPSERT USERS
# -----------------------------
user_sql = """
INSERT INTO users (username)
VALUES (%s)
ON DUPLICATE KEY UPDATE username=username
"""

user_rows = [(u,) for u in watch["username"].dropna().unique()]

for i, batch in enumerate(chunked(user_rows, 1000), start=1):
    cursor.executemany(user_sql, batch)
    conn.commit()
    print(f"{i * 1000}")


conn.commit()

print("User data added")

# -----------------------------
# BUILD LOOKUPS
# -----------------------------
movies_lookup = pd.read_sql("SELECT movie_id, movie_slug FROM movies", engine).set_index("movie_slug")["movie_id"].to_dict()
users_lookup = pd.read_sql("SELECT user_id, username FROM users", engine).set_index("username")["user_id"].to_dict()

# -----------------------------
# UPSERT RATINGS
# -----------------------------
rating_sql = """
INSERT INTO ratings (user_id, movie_id, rating)
VALUES (%s,%s,%s)
ON DUPLICATE KEY UPDATE
    rating=VALUES(rating),
    timestamp=CURRENT_TIMESTAMP
"""

rating_rows = [
    (
        users_lookup[row["username"]],
        movies_lookup[row["movie_slug"]],
        float(row["rating"])
    )
    for _, row in watch.iterrows()
]

print("Uploading ratings")

for i, batch in enumerate(chunked(rating_rows, 10000), start=1):
    cursor.executemany(rating_sql, batch)
    conn.commit()
    print(f"{i * 10000}")


conn.commit()

print("Ratings data added")

conn.close()

print("RDS ingestion complete")
print(f"Movies: {len(movies)}")
print(f"Users: {len(users_lookup)}")
print(f"Ratings: {len(watch)}")