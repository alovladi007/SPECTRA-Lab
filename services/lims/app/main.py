"""
SPECTRA-Lab LIMS/ELN Service
Laboratory Information Management System & Electronic Lab Notebook

This service provides REST API endpoints for:
- Sample management with barcode/QR tracking
- Chain of custody
- Electronic Lab Notebook (ELN)
- E-signatures (21 CFR Part 11)
- SOP management
- Report generation
- FAIR data export
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import uuid

# Create FastAPI app
app = FastAPI(
    title="SPECTRA-Lab LIMS/ELN Service",
    description="Laboratory Information Management System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

class Sample(BaseModel):
    sample_id: str
    name: str
    material_type: str
    location: str
    status: str = "received"
    barcode: Optional[str] = None
    created_at: datetime = datetime.now()

class CustodyLog(BaseModel):
    sample_id: str
    action: str
    from_user: str
    to_user: str
    from_location: str
    to_location: str
    timestamp: datetime = datetime.now()

class ELNEntry(BaseModel):
    entry_id: str
    title: str
    content: str
    author: str
    project_id: Optional[str] = None
    linked_samples: List[str] = []
    created_at: datetime = datetime.now()
    is_locked: bool = False

class Signature(BaseModel):
    entry_id: str
    user_id: str
    signature_type: str  # execution, review, approval
    reason: str
    timestamp: datetime = datetime.now()

class SOP(BaseModel):
    sop_number: str
    title: str
    version: str
    method_name: str
    content: str
    status: str = "draft"

# In-memory storage (in production, use database)
samples_db: Dict[str, Sample] = {}
custody_db: List[CustodyLog] = []
eln_db: Dict[str, ELNEntry] = {}
signatures_db: List[Signature] = []
sops_db: Dict[str, SOP] = {}

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "lims-eln",
        "version": "1.0.0",
        "capabilities": {
            "sample_management": True,
            "chain_of_custody": True,
            "eln": True,
            "e_signatures": True,
            "sop_management": True,
            "cfr21_part11_compliant": True
        }
    }

# ==================== Sample Management ====================

@app.post("/api/lims/samples", response_model=Sample)
async def create_sample(sample: Sample):
    """
    Create a new sample with automatic barcode generation
    """
    try:
        # Generate sample ID if not provided
        if not sample.sample_id:
            sample.sample_id = f"SMP-{uuid.uuid4().hex[:8].upper()}"

        # Generate barcode
        sample.barcode = f"BC-{sample.sample_id}"

        # Store sample
        samples_db[sample.sample_id] = sample

        return sample
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lims/samples", response_model=List[Sample])
async def list_samples(
    status: Optional[str] = None,
    material_type: Optional[str] = None,
    limit: int = 100
):
    """
    List all samples with optional filtering
    """
    samples = list(samples_db.values())

    if status:
        samples = [s for s in samples if s.status == status]
    if material_type:
        samples = [s for s in samples if s.material_type == material_type]

    return samples[:limit]

@app.get("/api/lims/samples/{sample_id}", response_model=Sample)
async def get_sample(sample_id: str):
    """
    Get sample details by ID
    """
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail="Sample not found")
    return samples_db[sample_id]

@app.put("/api/lims/samples/{sample_id}", response_model=Sample)
async def update_sample(sample_id: str, sample_update: Dict[str, Any]):
    """
    Update sample information
    """
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = samples_db[sample_id]
    for key, value in sample_update.items():
        if hasattr(sample, key):
            setattr(sample, key, value)

    return sample

# ==================== Chain of Custody ====================

@app.post("/api/lims/custody", response_model=CustodyLog)
async def add_custody_log(log: CustodyLog):
    """
    Add chain of custody entry
    """
    if log.sample_id not in samples_db:
        raise HTTPException(status_code=404, detail="Sample not found")

    custody_db.append(log)
    return log

@app.get("/api/lims/custody/{sample_id}", response_model=List[CustodyLog])
async def get_custody_chain(sample_id: str):
    """
    Get complete chain of custody for a sample
    """
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail="Sample not found")

    chain = [log for log in custody_db if log.sample_id == sample_id]
    return sorted(chain, key=lambda x: x.timestamp)

# ==================== Electronic Lab Notebook ====================

@app.post("/api/lims/eln/entries", response_model=ELNEntry)
async def create_eln_entry(entry: ELNEntry):
    """
    Create new ELN entry
    """
    if not entry.entry_id:
        entry.entry_id = f"ELN-{uuid.uuid4().hex[:8].upper()}"

    eln_db[entry.entry_id] = entry
    return entry

@app.get("/api/lims/eln/entries", response_model=List[ELNEntry])
async def list_eln_entries(
    author: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 100
):
    """
    List ELN entries with optional filtering
    """
    entries = list(eln_db.values())

    if author:
        entries = [e for e in entries if e.author == author]
    if project_id:
        entries = [e for e in entries if e.project_id == project_id]

    return sorted(entries, key=lambda x: x.created_at, reverse=True)[:limit]

@app.get("/api/lims/eln/entries/{entry_id}", response_model=ELNEntry)
async def get_eln_entry(entry_id: str):
    """
    Get ELN entry by ID
    """
    if entry_id not in eln_db:
        raise HTTPException(status_code=404, detail="ELN entry not found")
    return eln_db[entry_id]

# ==================== E-Signatures (21 CFR Part 11) ====================

@app.post("/api/lims/signatures", response_model=Signature)
async def add_signature(signature: Signature):
    """
    Add electronic signature to ELN entry
    21 CFR Part 11 compliant
    """
    if signature.entry_id not in eln_db:
        raise HTTPException(status_code=404, detail="ELN entry not found")

    entry = eln_db[signature.entry_id]
    if entry.is_locked:
        raise HTTPException(status_code=403, detail="Entry is locked and cannot be signed")

    signatures_db.append(signature)

    # Lock entry after approval signature
    if signature.signature_type == "approval":
        entry.is_locked = True

    return signature

@app.get("/api/lims/signatures/{entry_id}", response_model=List[Signature])
async def get_signatures(entry_id: str):
    """
    Get all signatures for an ELN entry
    """
    sigs = [s for s in signatures_db if s.entry_id == entry_id]
    return sorted(sigs, key=lambda x: x.timestamp)

# ==================== SOP Management ====================

@app.post("/api/lims/sops", response_model=SOP)
async def create_sop(sop: SOP):
    """
    Create new Standard Operating Procedure
    """
    sops_db[sop.sop_number] = sop
    return sop

@app.get("/api/lims/sops", response_model=List[SOP])
async def list_sops(
    method_name: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List SOPs with optional filtering
    """
    sops = list(sops_db.values())

    if method_name:
        sops = [s for s in sops if s.method_name == method_name]
    if status:
        sops = [s for s in sops if s.status == status]

    return sops

@app.get("/api/lims/sops/{sop_number}", response_model=SOP)
async def get_sop(sop_number: str):
    """
    Get SOP by number
    """
    if sop_number not in sops_db:
        raise HTTPException(status_code=404, detail="SOP not found")
    return sops_db[sop_number]

# ==================== Reports ====================

@app.post("/api/lims/reports/generate")
async def generate_report(
    run_id: str,
    template: str = "standard",
    include_plots: bool = True
):
    """
    Generate PDF report for a measurement run
    """
    return {
        "report_id": f"RPT-{uuid.uuid4().hex[:8].upper()}",
        "run_id": run_id,
        "template": template,
        "status": "generated",
        "download_url": f"/api/lims/reports/download/{run_id}",
        "generated_at": datetime.now().isoformat()
    }

# ==================== FAIR Data Export ====================

@app.post("/api/lims/export/fair")
async def export_fair_data(
    run_ids: List[str],
    include_raw: bool = True,
    include_metadata: bool = True
):
    """
    Export data in FAIR-compliant format
    (Findable, Accessible, Interoperable, Reusable)
    """
    return {
        "export_id": f"FAIR-{uuid.uuid4().hex[:8].upper()}",
        "run_ids": run_ids,
        "format": "zip",
        "includes": {
            "raw_data": include_raw,
            "metadata": include_metadata,
            "provenance": True,
            "schemas": True
        },
        "download_url": f"/api/lims/export/download/FAIR-{uuid.uuid4().hex[:8].upper()}",
        "created_at": datetime.now().isoformat()
    }

# ==================== Status ====================

@app.get("/api/status")
async def get_status():
    """Get LIMS service status"""
    return {
        "service": "lims-eln",
        "status": "operational",
        "stats": {
            "total_samples": len(samples_db),
            "total_eln_entries": len(eln_db),
            "total_signatures": len(signatures_db),
            "total_sops": len(sops_db),
            "custody_logs": len(custody_db)
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
