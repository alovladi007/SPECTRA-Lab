#!/bin/bash

# Entrypoint script for SPECTRA-Lab backend

set -e

echo "Starting SPECTRA-Lab backend..."

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Wait for Redis
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is ready!"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Create default admin user if not exists
echo "Initializing default data..."
python scripts/init_data.py

# Start the application
echo "Starting FastAPI application..."
exec "$@"