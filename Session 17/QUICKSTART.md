# SPECTRA-Lab Session 17 - Quick Start Guide

Get up and running in **5 minutes** ⚡

---

## Prerequisites ✅

```bash
# Check you have:
docker --version          # Docker 24+
docker compose version    # Docker Compose v2+
python3 --version         # Python 3.11+
```

---

## Installation (2 minutes)

```bash
# Step 1: Extract package
cd session17

# Step 2: Run automated deployment
./deploy_session17.sh dev
```

**That's it!** The script will:
- ✅ Start PostgreSQL
- ✅ Install Python dependencies
- ✅ Run database migrations
- ✅ Seed demo data
- ✅ Start all services

---

## Access the Platform

### Web UI
```
http://localhost:3012
```

### API Documentation
```
Analysis API: http://localhost:8001/docs
LIMS API:     http://localhost:8002/docs
```

### Demo Login
```
Email:    engineer@demo.lab
Password: eng123
```

---

## Test Your Installation (1 minute)

### 1. Login via API
```bash
curl -X POST http://localhost:8002/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"engineer@demo.lab","password":"eng123"}'
```

**Expected:** JSON response with `access_token`

### 2. List Samples
```bash
# Save token from previous step
export TOKEN="your_access_token_here"

curl http://localhost:8002/api/lims/samples \
  -H "Authorization: Bearer $TOKEN"
```

**Expected:** JSON array with 5 demo samples

### 3. Check Calibration Status
```bash
curl http://localhost:8001/api/analysis/calibrations \
  -H "Authorization: Bearer $TOKEN"
```

**Expected:** List of instrument calibrations (some valid, some expired)

---

## Common Commands

```bash
# View logs
docker compose logs -f

# Restart services
docker compose restart

# Stop everything
docker compose down

# Database shell
docker compose exec db psql -U spectra

# Run tests
pytest tests/integration/test_session17.py -v
```

---

## Demo Users

| Email | Password | Role | Use Case |
|-------|----------|------|----------|
| admin@demo.lab | admin123 | Admin | System configuration |
| pi@demo.lab | pi123 | PI | Approve recipes |
| engineer@demo.lab | eng123 | Engineer | Create experiments |
| tech@demo.lab | tech123 | Technician | Run tests |
| viewer@demo.lab | view123 | Viewer | View results |

---

## Key Features Demonstrated

### 1. Multi-Org Tenancy
- Demo Lab organization created
- All data scoped to organization
- Users cannot access other orgs' data

### 2. Role-Based Access Control
- 5 role levels with different permissions
- Admin > PI > Engineer > Technician > Viewer
- Enforced at API level

### 3. Calibration Lockout
- Instruments require valid calibration
- Runs blocked if calibration expired
- Demo includes expired XRD calibration

### 4. Complete Audit Trail
- All actions timestamped
- User actions tracked
- Immutable ELN entries with e-signatures

### 5. SPC Foundation
- Control chart series defined
- Ready for data points
- Alert system configured

---

## Troubleshooting

### Problem: "Port already in use"
**Solution:**
```bash
# Stop conflicting services
docker compose down
sudo lsof -ti:5432 | xargs kill -9  # PostgreSQL
sudo lsof -ti:8001 | xargs kill -9  # Analysis API
sudo lsof -ti:8002 | xargs kill -9  # LIMS API
```

### Problem: "Database connection failed"
**Solution:**
```bash
# Restart PostgreSQL
docker compose restart db

# Wait for ready
docker compose exec db pg_isready -U spectra
```

### Problem: "Token expired"
**Solution:**
```bash
# Tokens expire after 15 minutes
# Login again to get fresh token
curl -X POST http://localhost:8002/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"engineer@demo.lab","password":"eng123"}'
```

---

## Next Steps

1. **Explore API Docs**
   - Visit http://localhost:8001/docs
   - Try interactive API calls
   - Review request/response schemas

2. **Review Architecture**
   - Read SESSION_17.md for details
   - Understand database schema
   - Review auth flow

3. **Run Integration Tests**
   ```bash
   pytest tests/integration/test_session17.py -v
   ```

4. **Customize for Your Lab**
   - Add your instruments
   - Create your materials library
   - Define your SOPs

---

## Production Deployment

This is a **development setup**. For production:

1. **Use OIDC** (Auth0/Keycloak) instead of dev JWT
2. **Enable TLS** on all endpoints
3. **Use managed PostgreSQL** (AWS RDS, GCP Cloud SQL)
4. **Set strong secrets** (no defaults)
5. **Enable monitoring** (Prometheus + Grafana)
6. **Configure backups** (automated daily)
7. **Review security** checklist in SESSION_17.md

---

## Support

- **Documentation:** See README.md and SESSION_17.md
- **Issues:** Report in project repository
- **Questions:** Contact development team

---

## Success Indicators ✅

Your deployment is successful if:

- [ ] All services show "healthy" status
- [ ] Can login with demo credentials
- [ ] API returns sample list
- [ ] Can create new sample
- [ ] Calibration lockout blocks expired instrument
- [ ] Role guards prevent unauthorized actions

```bash
# Quick health check
curl http://localhost:8001/health
curl http://localhost:8002/health
```

---

**Generated:** October 26, 2025  
**Session:** 17 - Production Database & Auth  
**Status:** Ready for deployment 🚀
