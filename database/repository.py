"""
Database operations for the wine recommendation system.
"""

import pandas as pd

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError

from .connection import SessionLocal, engine
from .models import Wine, WineAttributes, WineSource

# --------------------------------------------------
# CREATE
# --------------------------------------------------


def add_wine(
    producer,
    name,
    vintage=None,
    grape=None,
    wine_type=None,
    country=None,
    region=None,
    appellation=None,
    description=None,
    alcohol_percent=None,
    attributes=None,
    source_name=None,
    external_id=None,
):
    """
    Insert a new wine into the database.

    Optionally include sensory attributes and source information.

    Returns:
        int: ID of the inserted wine.
    """

    with SessionLocal() as session:

        wine = Wine(
            producer=producer,
            name=name,
            vintage=vintage,
            grape=grape,
            wine_type=wine_type,
            country=country,
            region=region,
            appellation=appellation,
            description=description,
            alcohol_percent=alcohol_percent,
        )

        session.add(wine)
        session.flush()

        if attributes:
            wine_attributes = WineAttributes(
                wine_id=wine.id,
                **attributes,
            )
            session.add(wine_attributes)

        if source_name:
            source = WineSource(
                wine_id=wine.id,
                source_name=source_name,
                external_id=external_id,
            )
            session.add(source)

        session.commit()

        return wine.id


# --------------------------------------------------
# READ
# --------------------------------------------------


def get_wine_by_id(wine_id):
    """
    Retrieve a wine and its attributes by ID.
    """

    with SessionLocal() as session:

        wine = session.get(Wine, wine_id)

        if wine is None:
            return None

        result = {
            "id": wine.id,
            "producer": wine.producer,
            "name": wine.name,
            "vintage": wine.vintage,
            "grape": wine.grape,
            "wine_type": wine.wine_type,
            "country": wine.country,
            "region": wine.region,
            "appellation": wine.appellation,
            "description": wine.description,
            "alcohol_percent": wine.alcohol_percent,
        }

        if wine.attributes:
            result["attributes"] = {
                "body": wine.attributes.body,
                "acidity": wine.attributes.acidity,
                "tannin": wine.attributes.tannin,
                "sweetness": wine.attributes.sweetness,
                "fruitiness": wine.attributes.fruitiness,
                "oakiness": wine.attributes.oakiness,
                "intensity": wine.attributes.intensity,
                "tasting_notes": wine.attributes.tasting_notes,
                "food_pairings": wine.attributes.food_pairings,
            }

        return result


def get_all_wines():
    """
    Return all wines as a Pandas DataFrame.
    """

    query = select(Wine)

    return pd.read_sql(query, engine)


def search_wines(
    name=None,
    producer=None,
    grape=None,
    region=None,
    wine_type=None,
    limit=50,
):
    """
    Search the database using optional filters.
    """

    query = select(Wine)

    if name:
        query = query.where(Wine.name.ilike(f"%{name}%"))

    if producer:
        query = query.where(Wine.producer.ilike(f"%{producer}%"))

    if grape:
        query = query.where(Wine.grape.ilike(f"%{grape}%"))

    if region:
        query = query.where(Wine.region.ilike(f"%{region}%"))

    if wine_type:
        query = query.where(Wine.wine_type.ilike(f"%{wine_type}%"))

    query = query.limit(limit)

    return pd.read_sql(query, engine)


def get_wine_features():
    """
    Retrieve wine metadata and numerical attributes.

    Intended for similarity calculations and recommendation models.
    """

    query = select(
        Wine.id.label("wine_id"),
        Wine.producer,
        Wine.name,
        Wine.vintage,
        Wine.grape,
        Wine.wine_type,
        Wine.country,
        Wine.region,
        Wine.alcohol_percent,
        WineAttributes.body,
        WineAttributes.acidity,
        WineAttributes.tannin,
        WineAttributes.sweetness,
        WineAttributes.fruitiness,
        WineAttributes.oakiness,
        WineAttributes.intensity,
        WineAttributes.tasting_notes,
        WineAttributes.food_pairings,
    ).outerjoin(
        WineAttributes,
        Wine.id == WineAttributes.wine_id,
    )

    return pd.read_sql(query, engine)


def count_wines():
    """
    Return the total number of wines.
    """

    with SessionLocal() as session:
        return session.scalar(select(func.count(Wine.id)))


# --------------------------------------------------
# UPDATE
# --------------------------------------------------


def update_wine(wine_id, **kwargs):
    """
    Update existing wine metadata.

    Returns True if the wine exists, False otherwise.
    """

    allowed_fields = {
        "producer",
        "name",
        "vintage",
        "grape",
        "wine_type",
        "country",
        "region",
        "appellation",
        "description",
        "alcohol_percent",
    }

    unknown = set(kwargs) - allowed_fields

    if unknown:
        raise ValueError(f"Invalid wine fields: {sorted(unknown)}")

    with SessionLocal() as session:

        wine = session.get(Wine, wine_id)

        if wine is None:
            return False

        for key, value in kwargs.items():
            setattr(wine, key, value)

        session.commit()

        return True


def update_wine_attributes(wine_id, **kwargs):
    """
    Add or update numerical wine attributes.
    """

    allowed_fields = {
        "body",
        "acidity",
        "tannin",
        "sweetness",
        "fruitiness",
        "oakiness",
        "intensity",
        "tasting_notes",
        "food_pairings",
        "attribute_source",
    }

    unknown = set(kwargs) - allowed_fields

    if unknown:
        raise ValueError(f"Invalid attribute fields: {sorted(unknown)}")

    with SessionLocal() as session:

        wine = session.get(Wine, wine_id)

        if wine is None:
            return False

        if wine.attributes is None:
            wine.attributes = WineAttributes()

        for key, value in kwargs.items():
            setattr(wine.attributes, key, value)

        session.commit()

        return True


# --------------------------------------------------
# DELETE
# --------------------------------------------------


def delete_wine(wine_id):
    """
    Delete a wine and its associated records.
    """

    with SessionLocal() as session:

        wine = session.get(Wine, wine_id)

        if wine is None:
            return False

        session.delete(wine)
        session.commit()

        return True
