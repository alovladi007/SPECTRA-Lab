#!/usr/bin/env python3
"""
Generate DDL SQL from SQLAlchemy models
"""

import sys
from pathlib import Path
from io import StringIO

# Add services/shared to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "shared"))

from sqlalchemy import create_engine
from sqlalchemy.schema import CreateTable
from services.shared.db.base import Base
from services.shared.db import models  # Import all models

# Create a mock engine just for DDL generation
engine = create_engine("postgresql://", strategy='mock', executor=lambda sql, *_: None)

# Generate CREATE statements for all tables
print("-- SPECTRA-Lab Database Schema")
print("-- Generated from SQLAlchemy models\\n")

for table in Base.metadata.sorted_tables:
    create_stmt = str(CreateTable(table).compile(engine))
    print(f"{create_stmt};\\n")

print("-- Schema generation complete")
