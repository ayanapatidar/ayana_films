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

def search_film(name, year):
    params = {"query": name, "year": year, "include_adult": False}
    r = requests.get(f"{BASE_URL}/search/movie", headers=HEADERS, params=params)
    results = r.json().get("results", [])
    if results:
        return results[0]
    r = requests.get(f"{BASE_URL}/search/movie", headers=HEADERS, params={"query": name})
    results = r.json().get("results", [])
    return results[0] if results else None

def get_film_details(tmdb_id):
    r = requests.get(f"{BASE_URL}/movie/{tmdb_id}", headers=HEADERS)
    return r.json()

def enrich_films():
    result = supabase.table("films").select("id, name, year, tmdb_id").execute()
    films = result.data

    unenriched = [f for f in films if not f["tmdb_id"]]
    print(f"{len(unenriched)} who up and enriching...")

    for i, film in enumerate(unenriched):
        print(f"[{i+1}/{len(unenriched)}] {film['name']} ({film['year']})", end=" ... ")

        match = search_film(film["name"], film["year"])
        if not match:
            print("no match :(")
            continue

        details = get_film_details(match["id"])
        genres = [g["name"] for g in details.get("genres", [])]

        update = {
            "tmdb_id": details["id"],
            "genres": genres,
            "runtime_mins": details.get("runtime"),
            "original_lang": details.get("original_language"),
            "overview": details.get("overview")
        }

        supabase.table("films").update(update).eq("id", film["id"]).execute()
        print(f"✓ {genres}")

        time.sleep(0.25)  #let's all be niceys

    print("Enrichment done!")

if __name__ == "__main__":
    enrich_films()