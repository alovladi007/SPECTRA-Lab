from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SPECTRA-Lab LIMS API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"service": "lims", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "lims"}
