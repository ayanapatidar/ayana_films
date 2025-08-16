import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def load_diary(path="data/diary.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["watched_date"] = pd.to_datetime(df["watched_date"]).dt.date.astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["rewatch"] = df["rewatch"].fillna("No") == "Yes"
    df["rating"] = df["rating"].astype(float)

    def categorize_tags(tag_str):
        if pd.isna(tag_str):
            return []
        tags = [t.strip().lower() for t in tag_str.split(",")]
        categories = []
        for tag in tags:
            if tag.startswith("w/"):
                categories.append("social")
            elif tag in ["theater", "theatre", "screening", "imax"]:
                categories.append("theater")
            elif tag in ["criterion"]:
                categories.append("criterion")
            elif tag in ["class"]:
                categories.append("class")
            elif "break" in tag or "plane" in tag:
                categories.append("travel_or_break")
            elif "brother" in tag:
                categories.append("background")
        return categories

    df["tag_categories"] = df["tags"].apply(categorize_tags)

    return df

def upsert_films(df, supabase):
    films = df[["name", "year", "letterboxd_uri"]].drop_duplicates(subset=["name", "year"])
    
    records = [
        {
            "name": row["name"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "letterboxd_uri": row["letterboxd_uri"]
        }
        for _, row in films.iterrows()
    ]

    print(f"Upserting {len(records)} films...")
    result = supabase.table("films").upsert(records, on_conflict="name,year").execute()
    print("Films done.")
    return result

def fetch_film_ids(supabase):
    result = supabase.table("films").select("id, name, year").execute()
    lookup = {(r["name"], r["year"]): r["id"] for r in result.data}
    return lookup

def upsert_watches(df, film_lookup, supabase):
    records = []
    skipped = 0

    for _, row in df.iterrows():
        film_id = film_lookup.get((row["name"], int(row["year"])))
        if not film_id:
            skipped += 1
            continue

        records.append({
            "film_id": film_id,
            "watched_date": row["watched_date"],
            "logged_date": row["date"],
            "rating": row["rating"],
            "rewatch": bool(row["rewatch"]),
            "tags": row["tag_categories"] if row["tag_categories"] else None
        })

    print(f"Upserting {len(records)} watches ({skipped} skipped)...")
    result = supabase.table("watches").upsert(records).execute()
    print("Watches done.")
    return result

if __name__ == "__main__":
    print("Loading diary...")
    df = load_diary()

    print("Upserting films...")
    upsert_films(df, supabase)

    print("Fetching film IDs...")
    film_lookup = fetch_film_ids(supabase)

    print("Upserting watches...")
    upsert_watches(df, film_lookup, supabase)

    print("All done!")