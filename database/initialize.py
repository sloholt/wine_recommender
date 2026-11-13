"""
Initialize SQLite database using schema.sql.

Run:
    python -m database.initialize
"""

import sqlite3
from pathlib import Path

from .connection import DATABASE_PATH, get_connection

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def initialize_database():
    """
    Create database tables and indexes from schema.sql.

    Existing tables and records are preserved.
    """

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    # Read SQL schema
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    print("\nInitializing Wine Database...")
    print(f"Database location: {DATABASE_PATH}")

    # Connect and execute schema
    with get_connection() as conn:
        conn.executescript(schema_sql)

        # Retrieve created tables
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """).fetchall()

    print("\nDatabase tables:")

    for table in tables:
        print(f"  - {table['name']}")

    print(f"\nTotal tables: {len(tables)}")
    print("\nDatabase initialized successfully!")


if __name__ == "__main__":
    initialize_database()
