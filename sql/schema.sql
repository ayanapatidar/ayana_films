CREATE TABLE IF NOT EXISTS films (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    year            INTEGER,
    letterboxd_uri  TEXT,
    tmdb_id         INTEGER UNIQUE,
    genres          TEXT[],
    runtime_mins    INTEGER,
    original_lang   TEXT,
    overview        TEXT,
    UNIQUE (name, year)
);

CREATE TABLE IF NOT EXISTS watches (
    id              SERIAL PRIMARY KEY,
    film_id         INTEGER REFERENCES films(id) ON DELETE CASCADE,
    watched_date    DATE,
    logged_date     DATE,
    rating          NUMERIC(2,1) CHECK (rating >= 0.5 AND rating <= 5.0),
    rewatch         BOOLEAN DEFAULT false,
    tags            TEXT[]
);

