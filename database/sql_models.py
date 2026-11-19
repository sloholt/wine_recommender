"""
SQLAlchemy models for the wine recommendation database.

Mirrors schema.sql exactly (same tables, columns, keys and indexes):

    countries -> regions -> wineries -> wines -> wine_vintages
    wines -> wine_grapes -> grapes ; wines -> wine_sensory / wine_aromas
    wines (+ optional wine_vintages) -> wine_ratings / wine_prices / wine_sources

Save this as ``models.py`` in the package that holds ``connection.py``.
The database is created from schema.sql; do not use ``Base.metadata.create_all``
to build it (it is safe on an existing database, but schema.sql is the source of truth).

Conventions:
  * Sensory numbers are on the scales documented in ingestion/core/tasting.py
    (body 1-5, acidity 1-3, ...). NULL means "unknown", never "zero".
  * ``vintage_id`` NULL on a rating / price / source means "applies to the whole wine".
  * ``wine_vintages.vintage`` NULL means non-vintage (N.V.).
"""

from sqlalchemy import (
    REAL,
    Column,
    Computed,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from .connection import Base


class Country(Base):
    __tablename__ = "countries"

    country_id = Column(Integer, primary_key=True)
    name = Column(Text(collation="NOCASE"), nullable=False, unique=True)
    iso_code = Column(Text)

    regions = relationship("Region", back_populates="country")

    def __repr__(self):
        return f"<Country({self.country_id}, {self.name!r})>"


class Region(Base):
    __tablename__ = "regions"

    region_id = Column(Integer, primary_key=True)
    country_id = Column(Integer, ForeignKey("countries.country_id"), nullable=False)
    parent_region_id = Column(Integer, ForeignKey("regions.region_id"))
    name = Column(Text(collation="NOCASE"), nullable=False)
    region_type = Column(Text)

    country = relationship("Country", back_populates="regions")
    parent = relationship("Region", remote_side=[region_id], back_populates="children")
    children = relationship("Region", back_populates="parent")
    aliases = relationship(
        "RegionAlias", back_populates="region", cascade="all, delete-orphan"
    )
    wineries = relationship("Winery", back_populates="region")
    wines = relationship("Wine", back_populates="region")

    def __repr__(self):
        return f"<Region({self.region_id}, {self.name!r})>"


Index(
    "ux_regions",
    Region.__table__.c.country_id,
    func.coalesce(Region.__table__.c.parent_region_id, 0),
    Region.__table__.c.name,
    unique=True,
)


class RegionAlias(Base):
    __tablename__ = "region_aliases"
    __table_args__ = (UniqueConstraint("region_id", "alias"),)

    alias_id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey("regions.region_id"), nullable=False)
    alias = Column(Text(collation="NOCASE"), nullable=False)

    region = relationship("Region", back_populates="aliases")


class Winery(Base):
    __tablename__ = "wineries"

    winery_id = Column(Integer, primary_key=True)
    name = Column(Text(collation="NOCASE"), nullable=False)
    region_id = Column(Integer, ForeignKey("regions.region_id"))
    website = Column(Text)

    region = relationship("Region", back_populates="wineries")
    wines = relationship("Wine", back_populates="winery")

    def __repr__(self):
        return f"<Winery({self.winery_id}, {self.name!r})>"


Index(
    "ux_wineries",
    Winery.__table__.c.name,
    func.coalesce(Winery.__table__.c.region_id, 0),
    unique=True,
)


class Wine(Base):
    """One wine (product), independent of year. Years live in WineVintage."""

    __tablename__ = "wines"

    wine_id = Column(Integer, primary_key=True)
    winery_id = Column(Integer, ForeignKey("wineries.winery_id"))
    region_id = Column(Integer, ForeignKey("regions.region_id"))
    name = Column(Text(collation="NOCASE"), nullable=False)
    full_title = Column(Text)
    style = Column(Text)  # Sparkling / Dessert / Fortified; NULL = still
    color = Column(Text)  # Red / White / Rosé / Orange
    varietal = Column(Text)  # only when a single grape; see WineGrape otherwise
    abv = Column(REAL)
    closure = Column(Text)

    winery = relationship("Winery", back_populates="wines")
    region = relationship("Region", back_populates="wines")
    vintages = relationship(
        "WineVintage", back_populates="wine", cascade="all, delete-orphan"
    )
    grapes = relationship(
        "WineGrape", back_populates="wine", cascade="all, delete-orphan"
    )
    sensory = relationship(
        "WineSensory",
        back_populates="wine",
        uselist=False,
        cascade="all, delete-orphan",
    )
    aromas = relationship(
        "WineAroma", back_populates="wine", cascade="all, delete-orphan"
    )
    ratings = relationship(
        "WineRating", back_populates="wine", cascade="all, delete-orphan"
    )
    prices = relationship(
        "WinePrice", back_populates="wine", cascade="all, delete-orphan"
    )
    sources = relationship(
        "WineSource", back_populates="wine", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Wine({self.wine_id}, {self.title!r})>"


Index(
    "ux_wines",
    func.coalesce(Wine.__table__.c.winery_id, 0),
    Wine.__table__.c.name,
    unique=True,
)
Index("ix_wines_region", Wine.__table__.c.region_id)
Index("ix_wines_color", Wine.__table__.c.color)


class WineVintage(Base):
    __tablename__ = "wine_vintages"

    vintage_id = Column(Integer, primary_key=True)
    wine_id = Column(Integer, ForeignKey("wines.wine_id"), nullable=False)
    vintage = Column(Integer)  # NULL = non-vintage

    wine = relationship("Wine", back_populates="vintages")

    def __repr__(self):
        return f"<WineVintage(wine_id={self.wine_id}, vintage={self.vintage})>"


Index(
    "ux_wine_vintages",
    WineVintage.__table__.c.wine_id,
    func.coalesce(WineVintage.__table__.c.vintage, 0),
    unique=True,
)
Index("ix_wine_vintages_year", WineVintage.__table__.c.vintage)


class Grape(Base):
    """Each grape name is stored once; WineGrape links wines to grapes."""

    __tablename__ = "grapes"

    grape_id = Column(Integer, primary_key=True)
    name = Column(Text(collation="NOCASE"), nullable=False, unique=True)

    wine_links = relationship("WineGrape", back_populates="grape")

    def __repr__(self):
        return f"<Grape({self.grape_id}, {self.name!r})>"


class WineGrape(Base):
    __tablename__ = "wine_grapes"

    wine_id = Column(Integer, ForeignKey("wines.wine_id"), primary_key=True)
    grape_id = Column(Integer, ForeignKey("grapes.grape_id"), primary_key=True)
    blend_percentage = Column(REAL)  # NULL when the source gives no percentage

    wine = relationship("Wine", back_populates="grapes")
    grape = relationship("Grape", back_populates="wine_links")


Index("ix_wine_grapes_grape", WineGrape.__table__.c.grape_id)


class WineSensory(Base):
    """Numeric tasting structure, one row per wine. NULL = unknown."""

    __tablename__ = "wine_sensory"

    sensory_id = Column(Integer, primary_key=True)
    wine_id = Column(Integer, ForeignKey("wines.wine_id"), nullable=False, unique=True)
    body = Column(REAL)
    acidity = Column(REAL)
    structure = Column(REAL)
    finish = Column(REAL)
    balance = Column(REAL)
    tannin = Column(REAL)
    sweetness = Column(REAL)

    wine = relationship("Wine", back_populates="sensory")


class WineAroma(Base):
    __tablename__ = "wine_aromas"
    __table_args__ = (UniqueConstraint("wine_id", "aroma_name"),)

    aroma_id = Column(Integer, primary_key=True)
    wine_id = Column(Integer, ForeignKey("wines.wine_id"), nullable=False)
    aroma_category = Column(Text)
    aroma_name = Column(Text, nullable=False)
    intensity = Column(REAL)

    wine = relationship("Wine", back_populates="aromas")


class WineRating(Base):
    __tablename__ = "wine_ratings"

    rating_id = Column(Integer, primary_key=True)
    wine_id = Column(Integer, ForeignKey("wines.wine_id"), nullable=False)
    vintage_id = Column(
        Integer, ForeignKey("wine_vintages.vintage_id")
    )  # NULL = all vintages
    score = Column(REAL, nullable=False)
    scale = Column(REAL, nullable=False)  # 100 for critics, 5 for X-Wines users
    rating_count = Column(Integer)
    reviewer = Column(Text)
    source = Column(Text)
    source_url = Column(Text)
    # score as a percentage of its scale, so 4/5 and 80/100 compare equal
    score_pct = Column(REAL, Computed("score * 100.0 / scale", persisted=False))

    wine = relationship("Wine", back_populates="ratings")
    vintage = relationship("WineVintage")


Index("ix_wine_ratings_wine", WineRating.__table__.c.wine_id)


class WinePrice(Base):
    __tablename__ = "wine_prices"

    price_id = Column(Integer, primary_key=True)
    wine_id = Column(Integer, ForeignKey("wines.wine_id"), nullable=False)
    vintage_id = Column(Integer, ForeignKey("wine_vintages.vintage_id"))
    price = Column(REAL, nullable=False)
    currency = Column(Text, nullable=False)
    retailer = Column(Text)
    source_url = Column(Text)
    observed_at = Column(Text)

    wine = relationship("Wine", back_populates="prices")
    vintage = relationship("WineVintage")


Index("ix_wine_prices_wine", WinePrice.__table__.c.wine_id)


class WineSource(Base):
    """Where a wine's data came from. (source_name, external_id) is unique."""

    __tablename__ = "wine_sources"

    source_id = Column(Integer, primary_key=True)
    wine_id = Column(Integer, ForeignKey("wines.wine_id"), nullable=False)
    vintage_id = Column(Integer, ForeignKey("wine_vintages.vintage_id"))
    source_name = Column(Text, nullable=False)
    external_id = Column(Text)
    source_url = Column(Text)
    rank = Column(Integer)
    tasting_note = Column(Text)
    tasting_note_truncated = Column(Integer, nullable=False, server_default="0")

    wine = relationship("Wine", back_populates="sources")
    vintage = relationship("WineVintage")


Index(
    "ux_wine_sources",
    WineSource.__table__.c.source_name,
    WineSource.__table__.c.external_id,
    unique=True,
    sqlite_where=WineSource.__table__.c.external_id.isnot(None),
)
Index("ix_wine_sources_wine", WineSource.__table__.c.wine_id)
