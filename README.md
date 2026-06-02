# ayana-films

a personal film recommender and taste analysis system built on my letterboxd watch history. i love film, and i love letterboxd, so i wanted to see what i could do with my own data!

note that all numbers are variable. since letterboxd doesn't allow api calls, i manually export my data and load it in. this repo can be used to generate insights on your own viewing history as well--- it just might skew results because of certain personal considerations i made while weighting my dataset and carrying out the embeddings. 

## what it does

- loads and cleans my personal letterboxd diary (586 watches, 2021–20 May, 2026) into postgresql via supabase
- enriches every film with metadata from the tmdb api: genres, runtime, language, overview
- augments the dataset with the movielens 25m dataset (3,784 well-rated films) for population-level taste signal
- trains a weighted svd recommender that accounts for context — rewatches, criterion watches, theater viewings, and background watches are all treated differently
- recommends films i haven't seen yet, scored against my personal taste vector
- embeds all 491 of my letterboxd reviews using sentence transformers and supports mood-based search: type a vibe, get recommendations grounded in my actual writing

## what i found out about myself

some things the data made clear that i sort of already knew:

- my ratings skew heavily toward older films; 1960s–1990s average well above 4.0, while 2015–2019 sits at 3.29. what can i say: al pacino's beautiful baby brown eyes bewitched me mind body and soul freshman year of college. 
- non-english cinema rates significantly higher across the board: japanese (4.39, n=27), french (4.33, n=15), cantonese (5.0, n=5)
- background watches (films i had on while doing something else, tagged accordingly)(don't ask why the actual tag is brother nefarious. it's a long story) average 2.43 stars vs 3.91 for everything else, which is a 1.5 star gap that meaningfully affects model quality if not handled
- october is horror and art house month, every year, without fail. make some noise for hooptober!
- criterion collection watches average 4.30; my most reliable signal for intentional, high-quality viewing. this makes sense. that subscription is like 40% of the reason i put so many hours on my timesheet during college. 

## how it's built

```
ayana-films/
├── data/                        # local only, not committed
│   ├── diary.csv                # letterboxd export
│   └── ml-25m/                  # movielens dataset
├── notebooks/
│   ├── looksee1.ipynb           # data exploration and cleaning
│   └── analysis.ipynb           # taste analysis (era, language, genre, mood)
├── etl/
│   ├── load.py                  # diary.csv -> film and watches table
│   ├── enrich.py                # tmdb enrichment for watched films
│   ├── load_movielens.py        # movielens -> candidate_films table
│   └── load_reviews.py          # review embeddings via sentence-transformers
├── recommender/
│   ├── model.py                 # svd recommender; weighted training + inference
│   └── sentiment.py             # mood-based search over review embeddings
├── api/                         # TO-DO 
│   ├── main.py                  # fastapi — /recommend and /mood endpoints 
│   └── schemas.py               # pydantic response types
├── sql/
│   ├── schema.sql               # table definitions
│   └── queries.sql              # analytical queries
└── requirements.txt
```

## database schema

four tables in postgresql (hosted on supabase), as of May 20, 2026:

- **films**: 491 films from my watch history, enriched with tmdb metadata
- **watches**: 586 diary entries with ratings, dates, rewatch flags, and categorized tags
- **reviews**: 491 letterboxd reviews with sentence-transformer embeddings for semantic search
- **candidate_films**: 3,784 well-rated movielens films with tmdb metadata and user-generated mood tags, used as the recommendation pool

## the recommender

the model is a weighted svd trained on a feature matrix of genres, era, and language, along with a small rewatch bonus.

weights applied before training:
- rewatch: +0.5
- criterion collection: +0.25
- theater viewing: +0.25
- hindi films: -0.5 (i LOVE bollywood i just tend to over rate it highly due to nostalgia... we all have our flaws mine is a lot of love for shah rukh khan!!!)
- background watches: excluded entirely. i will NOT be recommending the divergent trilogy to anyone!

era and language features are scaled up to compete with genre one-hot encoding, so the model's taste vector can reflect my bias toward older and non-english cinema.

## mood search

the sentiment search embeds a free-text mood query using `all-MiniLM-L6-v2` and scores it against three signals:

1. **review embeddings** (primary): cosine similarity between the query and my combined review + tmdb overview text
2. **movielens tag similarity** (secondary, 0.2 weight): similarity against curated mood/style tags from the movielens dataset
3. **genre bonus** (tertiary): a small boost when the film's genres align with the mood query, with a penalty for clear mismatches

example queries that work well:
- "something melancholy and slow with beautiful cinematography"
- "funny and warm, a comfort watch"
- "tense and atmospheric, makes me feel uneasy"

## stack

- **python**: pandas, scikit-learn, sentence-transformers, requests
- **postgresql**: supabase (with pgvector for embeddings)
- **apis**: tmdb api, movielens 25m dataset
- **TO-DO**: fastapi, next.js website integration

## future plans

### sentiment-adjusted ratings
the current recommender uses star ratings alone. planned: run sentiment analysis on each review using a model fine-tuned on film criticism, produce a sentiment score between -1 and 1, and blend it with the star rating to produce a more honest `adjusted_rating`. a lukewarm 4-star review should be treated differently than an enthusiastic one.

### fastapi integration
expose two endpoints — `/recommend` and `/mood`, so this system can be queried from anywhere. the `/recommend` endpoint returns my top unseen film picks, and `/mood` accepts a free-text string and returns films from my watch history that match the vibe. both will be integrated into my personal website as a public-facing "what i'd recommend" widget.

### taste profile dimensions
extract what i write positively and negatively about across all 491 reviews; cinematography, dialogue, atmosphere, pacing, plot; to build a critical profile that goes beyond genre and era. these dimensions will feed both the recommender and the taste profile dashboard on the website.

### sentiment search improvements
the current mood search works well for descriptive queries but struggles when my reviews are oblique or purely reactive rather than descriptive of the film. planned improvements:
- experiment with larger embedding models (`all-mpnet-base-v2`) for better semantic nuance
- add a director diversity filter to prevent one filmmaker dominating a single mood query (david lynch is like 90% of "tense and atmospheric" right now)
- explore fine-tuning the embedding model on film review corpora

### movielens collaborative filtering expansion
the current model uses svd on my personal taste vector compared against tmdb/movielens features. a natural extension is full collaborative filtering (finding movielens users whose rating patterns most resemble mine and surfacing what they loved that i haven't seen). this would surface some unexpected recommendations rather than films that just match my known genre preferences.

---

built by ayana. letterboxd: [ayana](https://letterboxd.com/taohun/)
