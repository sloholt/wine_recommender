
-- ============================================================
-- WINE RECOMMENDER DATABASE SCHEMA
-- ============================================================

PRAGMA foreign_keys = ON;


-- ============================================================
-- 1. COUNTRIES
-- ============================================================

CREATE TABLE IF NOT EXISTS countries (
    country_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    iso_code TEXT UNIQUE
);


-- ============================================================
-- 2. REGIONS
-- Geographic hierarchy:
-- Country -> State/Province -> Wine Region -> Subregion
-- ============================================================

CREATE TABLE IF NOT EXISTS regions (
    region_id INTEGER PRIMARY KEY,
    country_id INTEGER NOT NULL,
    parent_region_id INTEGER,
    name TEXT NOT NULL,
    region_type TEXT,

    FOREIGN KEY (country_id)
        REFERENCES countries(country_id),

    FOREIGN KEY (parent_region_id)
        REFERENCES regions(region_id),

    CHECK (parent_region_id IS NULL
           OR parent_region_id != region_id)
);


-- ============================================================
-- 3. REGION ALIASES
-- Alternative names used by different data sources
-- ============================================================

CREATE TABLE IF NOT EXISTS region_aliases (
    alias_id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL,
    alias TEXT NOT NULL,

    FOREIGN KEY (region_id)
        REFERENCES regions(region_id)
        ON DELETE CASCADE,

    UNIQUE (region_id, alias)
);

-- ============================================================
-- 4. WINES
-- Central wine table
-- Each wine/vintage combination is a separate record
-- ============================================================

CREATE TABLE IF NOT EXISTS wines (
    wine_id INTEGER PRIMARY KEY,
    region_id INTEGER,

    title TEXT NOT NULL,
    winery TEXT,
    varietal TEXT,
    vintage INTEGER,
    closure TEXT,

    FOREIGN KEY (region_id)
        REFERENCES regions(region_id)
        ON DELETE SET NULL,

    CHECK (
        vintage IS NULL
        OR vintage BETWEEN 1800 AND 2100
    ),

    CHECK (
        closure IS NULL
        OR closure IN (
            'Cork',
            'Screw Cap',
            'Synthetic Cork',
            'Other'
        )
    )
);


-- ============================================================
-- 5. WINE GRAPES
-- Stores grape varieties and blend percentages
-- ============================================================

CREATE TABLE IF NOT EXISTS wine_grapes (
    wine_grape_id INTEGER PRIMARY KEY,
    wine_id INTEGER NOT NULL,

    grape_variety TEXT NOT NULL,
    blend_percentage REAL,

    FOREIGN KEY (wine_id)
        REFERENCES wines(wine_id)
        ON DELETE CASCADE,

    CHECK (
        blend_percentage IS NULL
        OR blend_percentage BETWEEN 0 AND 100
    ),

    UNIQUE (wine_id, grape_variety)
);


-- ============================================================
-- 6. WINE AROMAS
-- One row per aroma descriptor per wine
-- ============================================================
CREATE TABLE IF NOT EXISTS wine_aromas (
    aroma_id INTEGER PRIMARY KEY,
    wine_id INTEGER NOT NULL,

    aroma_category TEXT NOT NULL,
    aroma_name TEXT NOT NULL,
    intensity TEXT,

    FOREIGN KEY (wine_id)
        REFERENCES wines(wine_id)
        ON DELETE CASCADE,

    CHECK (
        aroma_category IN (
            'Fruit',
            'Floral',
            'Other'
        )
    ),

    UNIQUE (wine_id, aroma_category, aroma_name)
);

-- ============================================================
-- 7. WINE SENSORY
-- Structural and sensory characteristics
-- Scale: 0-5
-- One sensory profile per wine
-- ============================================================

CREATE TABLE IF NOT EXISTS wine_sensory (
    sensory_id INTEGER PRIMARY KEY,
    wine_id INTEGER NOT NULL UNIQUE,

    structure TEXT,
    finish TEXT,
    balance TEXT,
    body TEXT,
    acidity TEXT,
    tannin TEXT,
    sweetness TEXT,

    FOREIGN KEY (wine_id)
        REFERENCES wines(wine_id)
        ON DELETE CASCADE
);

-- ============================================================
-- 8. WINE RATINGS
-- Multiple ratings per wine from different reviewers
-- ============================================================

CREATE TABLE IF NOT EXISTS wine_ratings (
    rating_id INTEGER PRIMARY KEY,
    wine_id INTEGER NOT NULL,

    score REAL NOT NULL,
    reviewer TEXT,
    source TEXT,

    FOREIGN KEY (wine_id)
        REFERENCES wines(wine_id)
        ON DELETE CASCADE,

    CHECK (score BETWEEN 0 AND 100)
);


-- ============================================================
-- 9. WINE PRICES
-- Multiple prices per wine across retailers
-- ============================================================

CREATE TABLE IF NOT EXISTS wine_prices (
    price_id INTEGER PRIMARY KEY,
    wine_id INTEGER NOT NULL,

    price REAL NOT NULL,
    currency TEXT NOT NULL,
    retailer TEXT,

    FOREIGN KEY (wine_id)
        REFERENCES wines(wine_id)
        ON DELETE CASCADE,

    CHECK (price >= 0),
    CHECK (length(currency) = 3)
);


-- ============================================================
-- 10. WINE SOURCES
-- Tracks origin of collected wine data
-- ============================================================

CREATE TABLE IF NOT EXISTS wine_sources (
    source_id INTEGER PRIMARY KEY,
    wine_id INTEGER NOT NULL,

    source_name TEXT NOT NULL,
    external_id TEXT,
    source_url TEXT,

    FOREIGN KEY (wine_id)
        REFERENCES wines(wine_id)
        ON DELETE CASCADE,

    UNIQUE (source_name, external_id)
);


-- ============================================================
-- INDEXES
-- Improve search, filtering, and recommendation queries
-- ============================================================

-- Geographic lookups
CREATE INDEX IF NOT EXISTS idx_regions_country
    ON regions(country_id);

CREATE INDEX IF NOT EXISTS idx_regions_parent
    ON regions(parent_region_id);

CREATE INDEX IF NOT EXISTS idx_region_aliases_alias
    ON region_aliases(alias);


-- Wine filtering
CREATE INDEX IF NOT EXISTS idx_wines_region
    ON wines(region_id);

CREATE INDEX IF NOT EXISTS idx_wines_varietal
    ON wines(varietal);

CREATE INDEX IF NOT EXISTS idx_wines_vintage
    ON wines(vintage);

CREATE INDEX IF NOT EXISTS idx_wines_winery
    ON wines(winery);


-- Recommendation features
CREATE INDEX IF NOT EXISTS idx_wine_grapes_variety
    ON wine_grapes(grape_variety);

CREATE INDEX IF NOT EXISTS idx_wine_aromas_name
    ON wine_aromas(aroma_name);

CREATE INDEX IF NOT EXISTS idx_wine_aromas_category
    ON wine_aromas(aroma_category);


-- Ratings and pricing
CREATE INDEX IF NOT EXISTS idx_wine_ratings_wine
    ON wine_ratings(wine_id);

CREATE INDEX IF NOT EXISTS idx_wine_prices_wine
    ON wine_prices(wine_id);


-- Source lookups
CREATE INDEX IF NOT EXISTS idx_wine_sources_wine
    ON wine_sources(wine_id);
