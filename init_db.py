#!/usr/bin/env python3
"""
Initialize database - create all tables from SQLAlchemy models
"""

import sys
from pathlib import Path

# Add services/shared to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "shared"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from services.shared.db.base import Base
from services.shared.db import models  # Import all models

# Database URL - use localhost:5433 for host access to Docker container
DATABASE_URL = "postgresql+psycopg://spectra:spectra@localhost:5433/spectra"

def init_database():
    """Create all tables from models"""
    print(f"Connecting to database...")
    engine = create_engine(DATABASE_URL, echo=True)

    print(f"\nCreating all tables...")
    Base.metadata.create_all(bind=engine)

    print(f"\n✓ Database initialized successfully!")
    print(f"  Created {len(Base.metadata.tables)} tables:")
    for table_name in sorted(Base.metadata.tables.keys()):
        print(f"    - {table_name}")

if __name__ == "__main__":
    init_database()
