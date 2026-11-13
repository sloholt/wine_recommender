"""
Initialize the SQLite wine database.

Run from the project root:

    python -m database.initialize
"""

from sqlalchemy import inspect

from .connection import Base, engine, DATABASE_PATH
from . import models  # Register models with SQLAlchemy


def initialize_database():
    """
    Create all database tables if they do not already exist.

    Existing tables and data are preserved.
    """

    print("\nInitializing Wine Database...")

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\nDatabase location: {DATABASE_PATH}")
    print("\nAvailable tables:")

    for table in tables:
        print(f"  - {table}")

    print("\nWine database initialized successfully!")


if __name__ == "__main__":
    initialize_database()
