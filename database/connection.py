"""
SQLite database connection and session management.
"""

from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Project root (one directory above database/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Database storage location
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "wines.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"


# Initialize SQLite engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"timeout": 30},
)


# Enable foreign key constraints in SQLite
@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


# Base class for all database models
class Base(DeclarativeBase):
    pass


def get_db():
    """
    Yield a database session and close it afterward.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
