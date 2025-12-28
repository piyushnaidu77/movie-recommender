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
    "database": os.getenv("DATABASE")
}

engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
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

# -----------------------------
# CREATE TABLES
# -----------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    movie_id INT AUTO_INCREMENT PRIMARY KEY,
    movie_slug VARCHAR(255) UNIQUE,
    title VARCHAR(255),
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
    user_id INT,
    movie_id INT,
    rating FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_rating (user_id, movie_id)
)
""")

conn.commit()

# -----------------------------
# UPSERT MOVIES
# -----------------------------
movie_sql = """
INSERT INTO movies (
    movie_slug, title, year, tmdb_id, director,
    runtime, studio, publisher, genre, plot, poster
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
    title=VALUES(title),
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
count = 1
for _, row in movies.iterrows():
    cursor.execute(movie_sql, tuple(row))
    if count%1000 == 0:
        print(count)
        count += 1
    else:
        count += 1

conn.commit()

# -----------------------------
# UPSERT USERS
# -----------------------------
user_sql = """
INSERT INTO users (username)
VALUES (%s)
ON DUPLICATE KEY UPDATE username=username
"""

for username in watch["username"].dropna().unique():
    cursor.execute(user_sql, (username,))

conn.commit()

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
count = 1
for _, row in watch.iterrows():
    user_id = users_lookup[row["username"]]
    movie_id = movies_lookup[row["movie_slug"]]

    cursor.execute(
        rating_sql,
        (user_id, movie_id, float(row["rating"]))
    )
    if count%1000 == 0:
        print(count)
        count += 1
    else:
        count += 1

conn.commit()
conn.close()

print("✅ MySQL ingestion complete")
print(f"Movies: {len(movies)}")
print(f"Users: {len(users_lookup)}")
print(f"Ratings: {len(watch)}")
