# File Organization Plan

## Current State
All project files are in `semiconductorlab_all_project_files/` as text files without proper extensions.

## Action Plan

### Step 1: Identify File Types
Files need to be categorized and given proper extensions:
- Python backend files → `.py`
- TypeScript/React components → `.tsx` or `.ts`
- Documentation → `.md`
- Configuration → `.yaml`, `.json`, etc.

### Step 2: Move to Correct Directories

**Python Files** → Move to appropriate locations:
- Analysis modules → `services/analysis/app/methods/`
- Drivers → `services/instruments/app/drivers/`
- Models → `src/backend/models/`
- Tests → `tests/`

**TypeScript/React Files** → Move to:
- UI Components → `apps/web/src/components/`
- Pages → `apps/web/src/app/(dashboard)/`

**Documentation** → Move to:
- Session docs → `docs/sessions/`
- Method playbooks → `docs/methods/`
- Architecture → `docs/architecture/`

### Step 3: Extract and Format Code
Each file contains:
1. Description/comments
2. Actual code
3. Sometimes multiple modules

Need to extract and format properly.

## Manual Steps Required

Due to the volume (88 files) and complexity, please confirm which files you want me to process first.

Recommended order:
1. Core infrastructure (database, API, models)
2. Analysis modules (electrical, optical)
3. UI components
4. Tests
5. Documentation

Would you like me to proceed with this organization?
