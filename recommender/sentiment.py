import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
import json
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

MODEL_NAME = "all-MiniLM-L6-v2"

MOOD_TAGS = {
    "atmospheric", "funny", "surreal", "cinematography", "suspense",
    "stylized", "visually appealing", "dark", "cult film", "mindfuck",
    "slow", "disturbing", "dreamlike", "soundtrack", "classic",
    "violence", "horror", "romance", "comedy", "drama", "thriller",
    "melancholy", "feel-good", "heartwarming", "funny", "witty",
    "dark humor", "satirical", "whimsical", "haunting", "tense",
    "uplifting", "emotional", "thought-provoking", "visually stunning",
    "slow burn", "quirky", "charming", "bleak", "hopeful", "nostalgic",
    "poetic", "intimate", "epic", "gripping", "mysterious", "eerie",
    "playful", "sincere", "bittersweet", "absurd", "lyrical",
    "meditative", "unsettling", "violent", "romantic", "exciting",
    "boring", "predictable", "overrated", "action", "crime", "family",
    "music", "love", "new york city", "based on a book", "murder",
    "sci-fi", "fantasy", "animation", "historical", "political",
    "social commentary", "feminist", "lgbtq", "coming of age",
    "friendship", "loss", "grief", "identity", "alienation"
}

GENRE_MOOD_MAP = {
    "funny and warm": ["Comedy", "Romance", "Animation", "Family"],
    "comfort": ["Comedy", "Romance", "Animation", "Family"],
    "melancholy": ["Drama"],
    "slow": ["Drama"],
    "cinematography": ["Drama", "History"],
    "tense": ["Thriller", "Horror", "Crime", "Mystery"],
    "atmospheric": ["Horror", "Thriller", "Mystery"],
    "uneasy": ["Horror", "Thriller", "Mystery"],
    "eerie": ["Horror", "Mystery"],
}

def get_film_ml_tags(tmdb_ids):
    if not tmdb_ids:
        return {}
    result = supabase.table("candidate_films").select(
        "tmdb_id, ml_tags"
    ).in_("tmdb_id", tmdb_ids).execute()
    return {
        r["tmdb_id"]: [t for t in (r["ml_tags"] or []) if t in MOOD_TAGS]
        for r in result.data
    }

def tag_similarity(query_embedding, tags, model):
    if not tags:
        return 0.0
    tag_text = " ".join(tags)
    tag_embedding = model.encode(tag_text)
    return cosine_similarity(query_embedding, tag_embedding)

def genre_bonus(query, genres):
    if not genres:
        return 0.0
    bonus = 0.0
    penalty = 0.0
    query_lower = query.lower()
    for mood_keyword, mood_genres in GENRE_MOOD_MAP.items():
        if mood_keyword in query_lower:
            matching = set(genres) & set(mood_genres)
            bonus += len(matching) * 0.05
            if mood_genres and not matching:
                penalty += 0.03

    return min(bonus, 0.2) - min(penalty, 0.1)

def fetch_reviews_with_embeddings():
    result = supabase.table("reviews").select(
        "film_id, review_text, embedding_v2, films(name, year, genres, original_lang)"
    ).execute()
    return result.data

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def mood_search(query, n=10, min_rating=3.5):
    print(f"Searching for: '{query}'")

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query).tolist()

    reviews = fetch_reviews_with_embeddings()

    watches = supabase.table("watches").select(
        "film_id, rating, tags, films(tmdb_id, genres)"
    ).execute().data
    rating_lookup = {w["film_id"]: w["rating"] for w in watches}
    tag_lookup = {w["film_id"]: w["tags"] or [] for w in watches}
    tmdb_lookup = {w["film_id"]: w["films"]["tmdb_id"] for w in watches if w["films"]}
    genre_lookup = {w["film_id"]: w["films"]["genres"] or [] for w in watches if w["films"]}

    tmdb_ids = [tid for tid in tmdb_lookup.values() if tid]
    ml_tags_lookup = get_film_ml_tags(tmdb_ids)
    tmdb_to_film = {v: k for k, v in tmdb_lookup.items()}

    results = []
    for review in reviews:
        if not review["embedding_v2"]:
            continue

        film_id = review["film_id"]
        rating = rating_lookup.get(film_id, 0)

        if rating < min_rating:
            continue
        if "background" in tag_lookup.get(film_id, []):
            continue

        embedding = review["embedding_v2"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        review_sim = cosine_similarity(query_embedding, embedding)
        tmdb_id = tmdb_lookup.get(film_id)
        ml_tags = ml_tags_lookup.get(tmdb_id, [])
        tag_sim = tag_similarity(query_embedding, ml_tags, model)

        genres = genre_lookup.get(film_id, [])
        g_bonus = genre_bonus(query, genres)

        final_score = (review_sim * 0.65) + (tag_sim * 0.2) + g_bonus

        film = review["films"]

        results.append({
            "name": film["name"],
            "year": film["year"],
            "genres": film["genres"],
            "original_lang": film["original_lang"],
            "rating": rating,
            "review_sim": round(review_sim, 4),
            "tag_sim": round(tag_sim, 4),
            "final_score": round(final_score, 4),
            "review": review["review_text"][:120] + "..." if len(review["review_text"]) > 120 else review["review_text"]
        })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:n]

if __name__ == "__main__":
    queries = [
        "something melancholy and slow with beautiful cinematography",
        "funny and warm, a comfort watch",
        "tense and atmospheric, makes me feel uneasy",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print('=' * 60)
        results = mood_search(query)
        for r in results:
            print(f"{r['name']} ({r['year']}) — {r['original_lang']} — ★{r['rating']} — review:{r['review_sim']} tag:{r['tag_sim']} final:{r['final_score']}")
            print(f"  {r['review']}")
            print()