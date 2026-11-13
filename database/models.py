"""
SQLAlchemy models for the wine recommendation database.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)

from sqlalchemy.orm import relationship

from .connection import Base


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Wine(Base):
    """
    Core wine identification and metadata.
    """

    __tablename__ = "wines"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Wine identification
    producer = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    vintage = Column(Integer, nullable=True)

    # Classification
    grape = Column(String(255), nullable=True)
    wine_type = Column(String(50), nullable=True)

    # Geographic information
    country = Column(String(100), nullable=True)
    region = Column(String(255), nullable=True)
    appellation = Column(String(255), nullable=True)

    # Additional metadata
    description = Column(Text, nullable=True)
    alcohol_percent = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )

    # Prevent duplicate named vintage records
    __table_args__ = (
        UniqueConstraint(
            "producer",
            "name",
            "vintage",
            name="uq_wine_identity",
        ),
    )

    # Relationships
    attributes = relationship(
        "WineAttributes",
        back_populates="wine",
        uselist=False,
        cascade="all, delete-orphan",
    )

    sources = relationship(
        "WineSource",
        back_populates="wine",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return (
            f"<Wine(id={self.id}, "
            f"name='{self.name}', "
            f"producer='{self.producer}', "
            f"vintage={self.vintage})>"
        )


class WineAttributes(Base):
    """
    Numerical wine characteristics used for recommendations.

    Sensory values should be normalized to [0, 1].
    Missing values should remain NULL.
    """

    __tablename__ = "wine_attributes"

    id = Column(Integer, primary_key=True)

    wine_id = Column(
        Integer,
        ForeignKey("wines.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Sensory attributes
    body = Column(Float, nullable=True)
    acidity = Column(Float, nullable=True)
    tannin = Column(Float, nullable=True)
    sweetness = Column(Float, nullable=True)
    fruitiness = Column(Float, nullable=True)
    oakiness = Column(Float, nullable=True)
    intensity = Column(Float, nullable=True)

    # Text-based characteristics
    tasting_notes = Column(Text, nullable=True)
    food_pairings = Column(Text, nullable=True)

    # Attribute provenance
    attribute_source = Column(String(100), nullable=True)

    updated_at = Column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )

    wine = relationship(
        "Wine",
        back_populates="attributes",
    )


class WineSource(Base):
    """
    Track external sources associated with each wine.

    Examples:
        CSV
        Wine Folly API
        Manual Entry
    """

    __tablename__ = "wine_sources"

    id = Column(Integer, primary_key=True)

    wine_id = Column(
        Integer,
        ForeignKey("wines.id", ondelete="CASCADE"),
        nullable=False,
    )

    source_name = Column(String(100), nullable=False)
    external_id = Column(String(255), nullable=True)
    source_url = Column(Text, nullable=True)

    last_fetched = Column(DateTime, default=utc_now)

    wine = relationship(
        "Wine",
        back_populates="sources",
    )
