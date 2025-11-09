# Session 17 Troubleshooting Guide

## Current Issue: Docker Port Mapping Not Working

### Problem Description

The PostgreSQL database container (spectra-db) is configured to expose port 5432 internally to port 5433 on the host, but the port is not being exposed despite correct docker-compose.yml configuration.

**Evidence:**
```bash
$ docker ps | grep spectra-db
spectra-db   5432/tcp  # Missing 0.0.0.0:5433->5432/tcp
```

**Expected:**
```bash
$ docker ps | grep spectra-db
spectra-db   0.0.0.0:5433->5432/tcp  # Port mapped correctly
```

### Root Cause

This is a Docker Desktop-specific networking issue on macOS. The database works perfectly within the Docker network but is not accessible from the host machine at localhost:5433.

### Verification Steps

1. **Database IS working inside Docker:**
   ```bash
   $ docker exec spectra-db psql -U spectra -d spectra -c "SELECT version();"
   #  Returns PostgreSQL 15.14 - database is healthy
   ```

2. **Port mapping configuration IS correct:**
   ```yaml
   # docker-compose.yml
   db:
     ports:
       - "5433:5432"  #  Correct configuration
   ```

3. **Host connection FAILS:**
   ```bash
   $ psql -h localhost -p 5433 -U spectra -d spectra
   #  Connection refused or times out
   ```

## Solutions & Workarounds

### Solution 1: Restart Docker Desktop (Recommended First Step)

```bash
# 1. Stop all containers
make db-down-session17

# 2. Quit Docker Desktop completely
# - Right-click Docker Desktop icon ’ Quit Docker Desktop

# 3. Restart Docker Desktop
# - Open Docker Desktop application

# 4. Start database with forced recreation
make db-reset-session17

# 5. Verify port mapping
docker ps | grep spectra-db
```

### Solution 2: Check Docker Desktop Settings

1. Open Docker Desktop ’ Settings ’ Resources
2. Verify "Use kernel networking for UDP" is disabled
3. Check File Sharing includes `/Users`
4. Under Advanced, verify sufficient resources allocated

### Solution 3: Use Docker Network Hostname

Since the database works within Docker, run migrations from a container that's in the same network:

```bash
# Option A: Use Analysis service container
docker exec -it spectra-analysis bash
cd /app
export DATABASE_URL="postgresql+psycopg://spectra:spectra@db:5432/spectra"
alembic upgrade head

# Option B: Run temporary container
docker run --rm -it \
  --network spectra-lab_spectra-network \
  -v "$PWD:/app" \
  -w /app \
  python:3.11 \
  bash -c "pip install -r requirements.txt && cd /app && export DATABASE_URL='postgresql+psycopg://spectra:spectra@db:5432/spectra' && alembic upgrade head"
```

### Solution 4: Use Port Forwarding

```bash
# In one terminal, create a port forward
docker exec -it spectra-db sh -c "socat TCP-LISTEN:5433,fork TCP:localhost:5432" &

# In another terminal, connect via forwarded port
export DATABASE_URL="postgresql+psycopg://spectra:spectra@localhost:5433/spectra"
make migrate-session17
```

### Solution 5: Direct exec into Database Container

```bash
# Copy migration files into container
docker cp alembic spectra-db:/tmp/
docker cp services/shared/db spectra-db:/tmp/

# Run migrations directly in container
docker exec -it spectra-db sh -c "
  cd /tmp &&
  pip install alembic sqlalchemy psycopg[binary] &&
  export DATABASE_URL='postgresql+psycopg://spectra:spectra@localhost:5432/spectra' &&
  alembic upgrade head
"
```

### Solution 6: Alternative Port (Last Resort)

If port 5433 is somehow blocked, try a different port:

```yaml
# docker-compose.yml
db:
  ports:
    - "15432:5432"  # Try completely different port
```

Then update DATABASE_URL in all configs.

## Quick Start After Port Fix

Once the port mapping is resolved, initialize the database:

```bash
# Option 1: Single command initialization
make init-session17

# Option 2: Step-by-step
make db-up-session17      # Start database
make migrate-session17    # Apply migrations
make seed-session17       # Load demo data
make test-session17       # Run tests

# Verify everything works
make db-status-session17
```

## Testing Database Connectivity

### From Host (After Fix)

```bash
# Test with psql
PGPASSWORD=spectra psql -h localhost -p 5433 -U spectra -d spectra -c "SELECT 1;"

# Test with Python
python3 -c "
import psycopg
conn = psycopg.connect('postgresql://spectra:spectra@localhost:5433/spectra')
print(' Connected successfully')
conn.close()
"

# Test Alembic
export DATABASE_URL="postgresql+psycopg://spectra:spectra@localhost:5433/spectra"
cd services/shared && alembic current
```

### From Docker Network (Works Now)

```bash
# Test from inside database container
docker exec spectra-db psql -U spectra -d spectra -c "SELECT 1;"

# Test from inside analysis service (if running)
docker exec spectra-analysis python3 -c "
import psycopg
conn = psycopg.connect('postgresql://spectra:spectra@db:5432/spectra')
print(' Connected successfully')
conn.close()
"
```

## Common Error Messages

### "nodename nor servname provided, or not known"
- **Cause**: Docker hostname not resolvable from host
- **Solution**: Use `localhost` instead of `spectra-db` when connecting from host

### "password authentication for user spectra failed"
- **Cause**: Wrong credentials or user doesn't exist
- **Solution**: Verify credentials in docker-compose.yml match connection string

### "Connection refused"
- **Cause**: Port not exposed or PostgreSQL not listening
- **Solution**: Check `docker ps` output, verify port mapping

### "FATAL: database \"spectra\" does not exist"
- **Cause**: Database not created during container initialization
- **Solution**: Recreate container with `make db-reset-session17`

## Status of Session 17 Implementation

###  Completed (80%)

- Database schema (20+ tables)
- SQLAlchemy models
- Alembic migrations (ready to apply)
- JWT authentication
- RBAC guards
- LIMS endpoints (samples, recipes, SOPs, ELN)
- Analysis endpoints (runs, calibrations)
- Unit tests (35 tests)
- Integration tests (15+ test cases)
- Documentation
- Makefile commands
- Seed script

### ó Blocked by Port Issue (20%)

- **Apply migrations** - Ready to run once database accessible
- **Load seed data** - Script ready, needs migration first
- **Run tests** - Tests complete, need database connection

## Next Steps

1. **Fix Port Mapping** - Try solutions above in order
2. **Apply Migrations** - `make migrate-session17`
3. **Seed Data** - `make seed-session17`
4. **Test Everything** - `make test-session17`
5. **Start Services** - Analysis (8001) + LIMS (8002)

## Support Resources

- Docker Desktop Troubleshooting: https://docs.docker.com/desktop/troubleshoot/overview/
- PostgreSQL Connection Docs: https://www.postgresql.org/docs/current/libpq-connect.html
- SQLAlchemy Connection Issues: https://docs.sqlalchemy.org/en/20/core/engines.html

## Contact

For Session 17-specific issues, refer to:
- [SESSION_17.md](SESSION_17.md) - Full implementation documentation
- [Makefile](Makefile) - All Session 17 commands
- `make help` - See all available commands

---

**Last Updated**: 2025-11-09
**Status**: Database healthy, awaiting host connectivity resolution
