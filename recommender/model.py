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

def fetch_watches():
    result = supabase.table("watches").select(
        "rating, rewatch, tags, films(id, name, year, genres, runtime_mins, original_lang)"
    ).execute()
    df = pd.json_normalize(result.data)
    df.columns = [c.replace("films.", "") for c in df.columns]
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

def build_feature_matrix(df):
    df = df.reset_index(drop=True)

    # one-hot encode genres
    genres_dummies = df["genres"].explode().str.get_dummies().groupby(level=0).max()

    # era feature
    # what can i say i love a 70s film 
    df["era_score"] = df["year"].apply(lambda y: max(0, (2000 - y) / 100) * 3 if pd.notna(y) else 0)

    # language feature (non-english bonus)
    df["lang_score"] = df["original_lang"].apply(lambda l: 0.3 if (l != "en" or l != "hi") else 0)

    features = pd.concat([
        genres_dummies,
        df[["era_score", "lang_score"]].reset_index(drop=True)
    ], axis=1).fillna(0)

    return features

def train(df, n_components=20):
    features = build_feature_matrix(df)
    weighted = df["weighted_rating"].values.reshape(-1, 1)

    # weight features by rating
    weighted_features = features.values * weighted
    normalized = normalize(weighted_features)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(normalized)

    # build ayana taste vector
    taste_vector = svd.transform(normalized).mean(axis=0)

    return svd, taste_vector, features.columns.tolist()

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

if __name__ == "__main__":
    print("Fetching watches.")
    df = fetch_watches()

    print("Applying weights.")
    df = apply_weights(df)
    print(f"  {len(df)} watches after removing background")
    df = deduplicate(df)
    print(f"  {len(df)} unique films after deduplication")

    print("Training model.")
    svd, taste_vector, feature_cols = train(df)
    print(f"  taste vector shape: {taste_vector.shape}")

    print("Top recommendations from my watched films:")
    recs = recommend(svd, taste_vector, df, feature_cols)
    print(recs.to_string(index=False))