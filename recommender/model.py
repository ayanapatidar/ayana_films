import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def fetch_candidates():
    result = supabase.table("candidate_films").select(
        "tmdb_id, name, year, genres, runtime_mins, original_lang, ml_avg_rating, ml_rating_count, ml_tags"
    ).execute()
    df = pd.DataFrame(result.data)
    return df


def fetch_watches():
    result = supabase.table("watches").select(
        "rating, rewatch, tags, films(id, name, year, genres, runtime_mins, original_lang, tmdb_id)"
    ).execute()
    df = pd.json_normalize(result.data)
    df.columns = [c.replace("films.", "") for c in df.columns]

    candidates = supabase.table("candidate_films").select("tmdb_id, ml_tags").execute()
    tag_lookup = {r["tmdb_id"]: r["ml_tags"] for r in candidates.data}
    df["ml_tags"] = df["tmdb_id"].map(tag_lookup).apply(lambda x: x if isinstance(x, list) else [])

    return df

def apply_weights(df):
    df = df.copy()

    # drop brother nefarious (background) entirely i am not going to recommend the wrong paris to people
    df = df[~df["tags"].apply(
        lambda x: "background" in x if isinstance(x, list) else False
    )]

    # start with raw rating
    df["weighted_rating"] = df["rating"]

    # rewatch bonus
    df.loc[df["rewatch"] == True, "weighted_rating"] += 0.5

    # criterion bonus
    df.loc[df["tags"].apply(
        lambda x: "criterion" in x if isinstance(x, list) else False
    ), "weighted_rating"] += 0.25

    # theater bonus
    df.loc[df["tags"].apply(
        lambda x: "theater" in x if isinstance(x, list) else False
    ), "weighted_rating"] += 0.25

    # i fear a lot of bollywood involves my fav genres but i lowkey am not
    # a fan of a lot of these media choices like, critically if that makes sense
    # i am a victim of nostalgia ! this will not affect my recommendations i prommyyyy
    df.loc[df["original_lang"] == "hi", "weighted_rating"] -= 0.5

    # cap at 5.0
    df["weighted_rating"] = df["weighted_rating"].clip(upper=5.0)

    return df

def deduplicate(df):
    # keep highest weighted_rating per film
    df = df.sort_values("weighted_rating", ascending=False)
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df.reset_index(drop=True)

def build_feature_matrix(df, top_tags = None):
    df = df.reset_index(drop=True)

    # one-hot encode genres
    genres_dummies = df["genres"].explode().str.get_dummies().groupby(level=0).max()

    # era feature
    # what can i say i love a 70s film 
    df["era_score"] = df["year"].apply(lambda y: max(0, (2000 - y) / 100) * 3 if pd.notna(y) else 0)

    # language feature (non-english bonus)
    df["lang_score"] = df["original_lang"].apply(lambda l: 0.3 if (l != "en" or l != "hi") else 0)

    if "ml_tags" in df.columns:
        if top_tags is None:
            all_tags = df["ml_tags"].explode().dropna()
            top_tags = all_tags.value_counts().head(50).index.tolist()
       
        def tag_vector(tag_list):
            return {t: 1 for t in tag_list if t in top_tags}
        
        tag_dummies = pd.DataFrame(
            df["ml_tags"].apply(tag_vector).tolist(),
            columns=top_tags
        ).fillna(0)
    else:
        tag_dummies = pd.DataFrame(index=df.index)

    features = pd.concat([
        genres_dummies.reset_index(drop=True),
        df[["era_score", "lang_score"]].reset_index(drop=True),
        tag_dummies.reset_index(drop=True)
    ], axis=1).fillna(0)

    return features

def train(df, n_components=20):
    all_tags = df["ml_tags"].explode().dropna() if "ml_tags" in df.columns else pd.Series([])
    top_tags = all_tags.value_counts().head(50).index.tolist()

    features = build_feature_matrix(df, top_tags)
    weighted = df["weighted_rating"].values.reshape(-1, 1)
    # weight features by rating
    weighted_features = features.values * weighted
    normalized = normalize(weighted_features)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(normalized)

    # build ayana taste vector
    taste_vector = svd.transform(normalized).mean(axis=0)

    return svd, taste_vector, features.columns.tolist(), top_tags

def recommend(svd, taste_vector, df, feature_cols, n=10):
    features = build_feature_matrix(df)
    film_vectors = svd.transform(normalize(features.values))

    # cosine similarity to taste vector
    similarities = film_vectors @ taste_vector / (
        np.linalg.norm(film_vectors, axis=1) * np.linalg.norm(taste_vector) + 1e-9
    )

    df = df.copy()
    df["similarity"] = similarities

    # exclude low-rated watches 
    seen_liked = df[df["weighted_rating"] >= 3.5]["id"].tolist()
    seen_disliked = df[df["weighted_rating"] < 3.5]["id"].tolist()

    # for now just from my seen pool
    candidates = df[
        df["id"].isin(seen_liked) &
        ~df["id"].isin(seen_disliked)
    ].sort_values("similarity", ascending=False)

    return candidates[["name", "year", "genres", "similarity", "weighted_rating"]].head(n)

def recommend_unseen(svd, taste_vector, watched_df, feature_cols_list, top_tags, n=10):
    candidates = fetch_candidates()

    watched_tmdb_ids = [r["tmdb_id"] for r in supabase.table("films").select("tmdb_id").execute().data]
    candidates = candidates[~candidates["tmdb_id"].isin(watched_tmdb_ids)]
    print(f"  {len(candidates)} unseen candidate films")

    candidates = candidates.reset_index(drop=True)
    features = build_feature_matrix(candidates, top_tags)

    # align to training columns exactly
    features = features.reindex(columns=feature_cols_list, fill_value=0)

    film_vectors = svd.transform(normalize(features.values))
    similarities = film_vectors @ taste_vector / (
        np.linalg.norm(film_vectors, axis=1) * np.linalg.norm(taste_vector) + 1e-9
    )

    candidates["similarity"] = similarities
    candidates["final_score"] = (
        candidates["similarity"] * 0.7 +
        (candidates["ml_avg_rating"] / 5.0) * 0.3
    )

    return candidates.sort_values("final_score", ascending=False)[
        ["name", "year", "genres", "original_lang", "ml_avg_rating", "final_score"]
    ].head(n)

if __name__ == "__main__":
    print("Fetching watches.")
    df = fetch_watches()

    print("Applying weights.")
    df = apply_weights(df)
    print(f"  {len(df)} watches after removing background")
    df = deduplicate(df)
    print(f"  {len(df)} unique films after deduplication")

    print("Training model.")
    svd, taste_vector, feature_cols, top_tags = train(df)
    print(f"  taste vector shape: {taste_vector.shape}")
    print(f"  top tags: {top_tags[:10]}")

    print("Top recommendations from my watched films:")
    recs = recommend_unseen(svd, taste_vector, df, feature_cols, top_tags)
    print(recs.to_string(index=False))