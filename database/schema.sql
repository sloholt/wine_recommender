-- Unified wine database schema (fresh start).
-- Design: winery -> wine (the product) -> vintage (a specific year of that wine).
-- Prices, critic scores and source rankings can attach to a vintage or, when
-- vintage_id is NULL, to the wine as a whole (e.g. X-Wines aggregate ratings).

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- geography
CREATE TABLE countries (
    country_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    iso_code   TEXT
);

CREATE TABLE regions (
    region_id        INTEGER PRIMARY KEY,
    country_id       INTEGER NOT NULL REFERENCES countries(country_id),
    parent_region_id INTEGER REFERENCES regions(region_id),
    name             TEXT NOT NULL COLLATE NOCASE,
    region_type      TEXT
);
-- COALESCE so top-level regions (NULL parent) are still unique per country.
CREATE UNIQUE INDEX ux_regions ON regions(country_id, COALESCE(parent_region_id, 0), name);

CREATE TABLE region_aliases (
    alias_id  INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(region_id),
    alias     TEXT NOT NULL COLLATE NOCASE,
    UNIQUE (region_id, alias)
);

-- ------------------------------------------------------------------ wineries
CREATE TABLE wineries (
    winery_id INTEGER PRIMARY KEY,
    name      TEXT NOT NULL COLLATE NOCASE,
    region_id INTEGER REFERENCES regions(region_id),
    website   TEXT
);
CREATE UNIQUE INDEX ux_wineries ON wineries(name, COALESCE(region_id, 0));

-- --------------------------------------------------------------------- wines
-- One row per wine (product), independent of year.
CREATE TABLE wines (
    wine_id   INTEGER PRIMARY KEY,
    winery_id INTEGER REFERENCES wineries(winery_id),
    region_id INTEGER REFERENCES regions(region_id),
    title     TEXT NOT NULL COLLATE NOCASE,
    color     TEXT,      -- Red / White / Rosé / Orange / Sparkling / Dessert ...
    varietal  TEXT,      -- only when a single grape; otherwise see wine_grapes
    abv       REAL,
    closure   TEXT
);
-- Identity = winery + title. (Rows with no known winery are not deduplicated by
-- this index, because NULLs are distinct; the ingest code must handle that case.)
CREATE UNIQUE INDEX ux_wines ON wines(COALESCE(winery_id, 0), title);
CREATE INDEX ix_wines_region ON wines(region_id);
CREATE INDEX ix_wines_color  ON wines(color);

-- One row per wine + year. vintage NULL = non-vintage (N.V.).
CREATE TABLE wine_vintages (
    vintage_id INTEGER PRIMARY KEY,
    wine_id    INTEGER NOT NULL REFERENCES wines(wine_id),
    vintage    INTEGER
);
CREATE UNIQUE INDEX ux_wine_vintages ON wine_vintages(wine_id, COALESCE(vintage, 0));
CREATE INDEX ix_wine_vintages_year ON wine_vintages(vintage);

-- ------------------------------------------------------ wine characteristics
-- Each grape name is stored once; wine_grapes links wines to grapes.
CREATE TABLE grapes (
    grape_id INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE wine_grapes (
    wine_id          INTEGER NOT NULL REFERENCES wines(wine_id),
    grape_id         INTEGER NOT NULL REFERENCES grapes(grape_id),
    blend_percentage REAL,           -- NULL when the source gives no percentage
    PRIMARY KEY (wine_id, grape_id)
);
CREATE INDEX ix_wine_grapes_grape ON wine_grapes(grape_id);

CREATE TABLE wine_sensory (
    sensory_id INTEGER PRIMARY KEY,
    wine_id    INTEGER NOT NULL UNIQUE REFERENCES wines(wine_id),
    body       REAL,   -- ordinal 1 (very light) .. 5 (very full)
    acidity    REAL,   -- ordinal 1 (low) .. 3 (high)
    structure  REAL,
    finish     REAL,
    balance    REAL,
    tannin     REAL,
    sweetness  REAL
);

CREATE TABLE wine_aromas (
    aroma_id       INTEGER PRIMARY KEY,
    wine_id        INTEGER NOT NULL REFERENCES wines(wine_id),
    aroma_category TEXT,
    aroma_name     TEXT NOT NULL,
    intensity      REAL,
    UNIQUE (wine_id, aroma_name)
);

-- --------------------------------------------------- ratings, prices, sources
CREATE TABLE wine_ratings (
    rating_id    INTEGER PRIMARY KEY,
    wine_id      INTEGER NOT NULL REFERENCES wines(wine_id),
    vintage_id   INTEGER REFERENCES wine_vintages(vintage_id),  -- NULL = all vintages
    score        REAL NOT NULL,
    scale        REAL NOT NULL,       -- 100 for critic points, 5 for X-Wines users
    rating_count INTEGER,             -- number of ratings behind an average
    reviewer     TEXT,
    source       TEXT,
    source_url   TEXT,
    score_pct    REAL GENERATED ALWAYS AS (score * 100.0 / scale) VIRTUAL
);
CREATE INDEX ix_wine_ratings_wine ON wine_ratings(wine_id);

CREATE TABLE wine_prices (
    price_id    INTEGER PRIMARY KEY,
    wine_id     INTEGER NOT NULL REFERENCES wines(wine_id),
    vintage_id  INTEGER REFERENCES wine_vintages(vintage_id),
    price       REAL NOT NULL,
    currency    TEXT NOT NULL,
    retailer    TEXT,
    source_url  TEXT,
    observed_at TEXT
);
CREATE INDEX ix_wine_prices_wine ON wine_prices(wine_id);

CREATE TABLE wine_sources (
    source_id            INTEGER PRIMARY KEY,
    wine_id              INTEGER NOT NULL REFERENCES wines(wine_id),
    vintage_id           INTEGER REFERENCES wine_vintages(vintage_id),
    source_name          TEXT NOT NULL,
    external_id          TEXT,         -- X-Wines WineID, or "<report>#<rank>" for lists
    source_url           TEXT,
    rank                 INTEGER,
    tasting_note         TEXT,
    tasting_note_truncated INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX ux_wine_sources ON wine_sources(source_name, external_id)
    WHERE external_id IS NOT NULL;
CREATE INDEX ix_wine_sources_wine ON wine_sources(wine_id);