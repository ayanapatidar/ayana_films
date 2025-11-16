import pandas as pd
import requests
import time
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

TMDB_TOKEN = os.getenv("TMDB_TOKEN")
HEADERS = {"Authorization": f"Bearer {TMDB_TOKEN}"}
BASE_URL = "https://api.themoviedb.org/3"

def load_movielens(min_ratings=1000):
    print("Loading MovieLens files :}")

    movies = pd.read_csv("data/ml-25m/movies.csv")
    links = pd.read_csv("data/ml-25m/links.csv")
    tags_df = pd.read_csv("data/ml-25m/tags.csv")

    print("Computing rating stats (this takes a moment)...")
    ratings = pd.read_csv("data/ml-25m/ratings.csv")
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    stats.columns = ["ml_avg_rating", "ml_rating_count"]
    stats = stats[stats["ml_rating_count"] >= min_ratings].reset_index()
    print(f"  {len(stats)} films with {min_ratings}+ ratings")

    # aggregate tags per film
    print("Aggregating tags :}")
    tags_agg = tags_df.groupby("movieId")["tag"].apply(
        lambda x: list(x.str.lower().unique())
    ).reset_index()
    tags_agg.columns = ["movieId", "ml_tags"]

    # merge everything
    df = stats.merge(movies, on="movieId")
    df = df.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    df = df.merge(tags_agg, on="movieId", how="left")
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").dropna()
    df = df.dropna(subset=["tmdbId"])
    df["tmdbId"] = df["tmdbId"].astype(int)

    print(f"  {len(df)} films with valid TMDB IDs")
    return df

def get_tmdb_details(tmdb_id):
    r = requests.get(f"{BASE_URL}/movie/{tmdb_id}", headers=HEADERS)
    if r.status_code != 200:
        return None
    return r.json()

def clean(val, fallback=None):
    if val is None:
        return fallback
    try:
        if pd.isna(val):
            return fallback
    except (TypeError, ValueError):
        pass
    return val
# movielens hates me

def enrich_and_store(df):
    # fetch already-stored tmdb IDs so we don't re-enrich
    existing = supabase.table("candidate_films").select("tmdb_id").execute()
    existing_ids = {r["tmdb_id"] for r in existing.data}

    todo = df[~df["tmdbId"].isin(existing_ids)]
    print(f"  {len(todo)} films to enrich.")

    for i, (_, row) in enumerate(todo.iterrows()):
        details = get_tmdb_details(int(row["tmdbId"]))
        if not details or details.get("status_code") == 34:
            continue

        genres = [g["name"] for g in details.get("genres", [])]
        record = {
            "tmdb_id": int(row["tmdbId"]),
            "name": clean(details.get("title"), row["title"]),
            "year": int(details["release_date"][:4]) if details.get("release_date") else None,
            "genres": genres,
            "runtime_mins": clean(details.get("runtime")),
            "original_lang": clean(details.get("original_language")),
            "overview": clean(details.get("overview")),
            "ml_avg_rating": round(float(row["ml_avg_rating"]), 3) if pd.notna(row["ml_avg_rating"]) else None,
            "ml_rating_count": int(row["ml_rating_count"]) if pd.notna(row["ml_rating_count"]) else None,
            "ml_tags": [t for t in row["ml_tags"] if isinstance(t, str)] if isinstance(row["ml_tags"], list) else [] # chungking express WHY
        }

        try:
            supabase.table("candidate_films").upsert(record, on_conflict="tmdb_id").execute()
        except Exception as e:
            print(f"Failed on {record['name']}: {e}")
            print(f"Failed on {record['name']}:")
            for k, v in record.items():
                print(f"  {k}: {repr(v)}")

            continue

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] enriched so far...") # i will go crazy if my terminal doesn't talk to me

        time.sleep(0.25) # be niceys always! 

    print("Enrichment done!")

if __name__ == "__main__":
    df = load_movielens(min_ratings=1000)
    enrich_and_store(df)