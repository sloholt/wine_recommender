"""
SQLite database connection management.
"""

import sqlite3
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Database location
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "wines.db"


def get_connection():
    """
    Create and return a SQLite database connection.

    Foreign key enforcement is enabled for each connection.
    """

    conn = sqlite3.connect(DATABASE_PATH)

    # Return rows as dictionary-like objects
    conn.row_factory = sqlite3.Row

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    return conn
