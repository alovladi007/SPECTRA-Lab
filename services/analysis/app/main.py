"""
Analysis Service - Port 8001
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SPECTRA-Lab Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3012"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"service": "analysis", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "analysis"}
