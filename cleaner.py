import pandas as pd
import os

df = pd.read_csv("data/movies_tmdb.csv")
print(df.isna().sum())
df_cleaned = df.dropna(subset=['plot', 'poster'])
print(df_cleaned.isna().sum())
df_cleaned.to_csv("data/tmdb_clean.csv", mode='a', index=False, header=not os.path.exists("data/tmdb_clean.csv"))