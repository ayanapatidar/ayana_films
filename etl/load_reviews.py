import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

MODEL_NAME = "all-MiniLM-L6-v2"

def load_reviews(path="data/reviews.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["review"])
    df["watched_date"] = pd.to_datetime(df["watched_date"]).dt.date.astype(str)
    return df

def fetch_film_lookup():
    result = supabase.table("films").select("id, name, year").execute()
    return {(r["name"], r["year"]): r["id"] for r in result.data}

def embed_and_store(df, film_lookup):
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # fetch film overviews
    films_result = supabase.table("films").select("id, overview").execute()
    overview_lookup = {r["id"]: r["overview"] or "" for r in films_result.data}

    records = []
    skipped_no_match = 0

    for _, row in df.iterrows():
        film_id = film_lookup.get((row["name"], int(row["year"])))
        if not film_id:
            skipped_no_match += 1
            continue

        overview = overview_lookup.get(film_id, "")
        combined = f"{row['review']} {overview}".strip()

        records.append({
            "film_id": film_id,
            "review_text": row["review"],
            "combined_text": combined,
            "review_date": row["watched_date"],
            "film_name": row["name"],
        })

    print(f"  {len(records)} reviews to embed ({skipped_no_match} skipped — no film match)")

    print("Embedding reviews...")
    texts = [r["combined_text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print("Storing to Supabase...")
    for i, (record, embedding) in enumerate(zip(records, embeddings)):
        film_name = record.pop("film_name")
        embedding_clean = [0.0 if np.isnan(v) else float(v) for v in embedding]
        record["embedding_v2"] = embedding_clean

        try:
            supabase.table("reviews").upsert(record, on_conflict="film_id").execute()
        except Exception as e:
            print(f"  Failed on {film_name}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}] stored...")

    print("Done!")

if __name__ == "__main__":
    print("Loading reviews...")
    df = load_reviews()
    print(f"  {len(df)} reviews loaded")

    print("Fetching film lookup...")
    film_lookup = fetch_film_lookup()

    print("Embedding and storing...")
    embed_and_store(df, film_lookup)