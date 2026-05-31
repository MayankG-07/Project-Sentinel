from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Calculate the base directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Set the database path
DB_PATH = BASE_DIR / "backend" / "data" / "db_data" / "company.db"

# Ensure the parent directory for the database exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Instantiate the engine
engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
)

# Instantiate the session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a dependency generator function named get_db()
def get_db():
    """
    FastAPI dependency to get a database session.
    Ensures the session is closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
