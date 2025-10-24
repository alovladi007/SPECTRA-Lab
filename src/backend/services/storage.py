# packages/common/semiconductorlab_common/storage.py

“””
Object Storage Management and File Format Handlers

Provides:

- S3/MinIO storage abstraction
- File format readers/writers (HDF5, CSV, JCAMP-DX, OME-TIFF)
- Metadata sidecar generation
- Data integrity verification (SHA256 hashing)
- Retention and lifecycle management
  “””

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO
from datetime import datetime
import io

import numpy as np
import pandas as pd
import h5py
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================

# Storage Configuration

# ============================================================================

@dataclass
class StorageConfig:
“”“Configuration for object storage”””
endpoint_url: str = “http://localhost:9000”
access_key: str = “minioadmin”
secret_key: str = “minioadmin”
bucket_name: str = “semiconductorlab-data”
region: str = “us-east-1”
secure: bool = False  # Use HTTPS

class FileType(str, Enum):
“”“Supported file types”””
HDF5 = “hdf5”
CSV = “csv”
PARQUET = “parquet”
JCAMP = “jcamp”
OME_TIFF = “ome_tiff”
JSON = “json”
NPZ = “npz”

# ============================================================================

# Storage Path Management

# ============================================================================

class StoragePathManager:
“””
Manages storage paths with consistent naming conventions

Path structure: {bucket}/{org_id}/{project_id}/{year}/{month}/{run_id}/{filename}
Example: semiconductorlab-data/org123/proj456/2025/10/run789/raw_data.h5
"""

def __init__(self, bucket_name: str = "semiconductorlab-data"):
    self.bucket_name = bucket_name

def get_run_path(
    self,
    organization_id: str,
    project_id: str,
    run_id: str,
    filename: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Generate path for run data"""
    ts = timestamp or datetime.utcnow()
    path = f"{organization_id}/{project_id}/{ts.year}/{ts.month:02d}/{run_id}/{filename}"
    return path

def get_report_path(
    self,
    organization_id: str,
    report_id: str,
    filename: str
) -> str:
    """Generate path for reports"""
    ts = datetime.utcnow()
    path = f"{organization_id}/reports/{ts.year}/{ts.month:02d}/{report_id}/{filename}"
    return path

def get_model_path(
    self,
    organization_id: str,
    model_id: str,
    version: str,
    filename: str
) -> str:
    """Generate path for ML models"""
    path = f"{organization_id}/models/{model_id}/{version}/{filename}"
    return path

def parse_uri(self, uri: str) -> Dict[str, str]:
    """Parse storage URI into components"""
    # Format: s3://bucket/path/to/file or minio://bucket/path/to/file
    protocol, rest = uri.split("://", 1)
    bucket, *path_parts = rest.split("/", 1)
    path = path_parts[0] if path_parts else ""
    
    return {
        "protocol": protocol,
        "bucket": bucket,
        "path": path
    }

# ============================================================================

# Metadata Sidecar

# ============================================================================

@dataclass
class FileMetadata:
“”“Metadata sidecar for stored files”””
filename: str
file_size: int  # bytes
mime_type: str
file_hash: str  # SHA256
storage_uri: str
created_at: str  # ISO 8601

# Provenance
run_id: Optional[str] = None
instrument_id: Optional[str] = None
operator_id: Optional[str] = None

# Physical units and dimensions
units: Optional[Dict[str, str]] = None
dimensions: Optional[Dict[str, int]] = None

# Data description
variables: Optional[Dict[str, Any]] = None

# Custom metadata
custom: Optional[Dict[str, Any]] = None

def to_json(self) -> str:
    """Serialize to JSON"""
    return json.dumps(asdict(self), indent=2)

@classmethod
def from_json(cls, json_str: str) -> 'FileMetadata':
    """Deserialize from JSON"""
    return cls(**json.loads(json_str))

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary"""
    return asdict(self)

# ============================================================================

# File Format Handlers

# ============================================================================

class HDF5Handler:
“””
HDF5 file handler for numeric array data

HDF5 structure:
/run_metadata (attributes)
/measurements
    /voltage (dataset)
    /current (dataset)
    /time (dataset)
/results
    /vth (scalar)
    /ideality (scalar)
"""

@staticmethod
def write(
    filepath: Union[str, Path],
    data: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write data to HDF5 file
    
    Args:
        filepath: Output file path
        data: Dictionary of {name: array}
        metadata: Optional metadata to store as attributes
    """
    with h5py.File(filepath, 'w') as f:
        # Store metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = json.dumps(value)
        
        # Create groups
        measurements_group = f.create_group('measurements')
        results_group = f.create_group('results')
        
        # Store datasets
        for name, array in data.items():
            if array.size == 1:
                # Scalar result
                results_group.create_dataset(name, data=array)
            else:
                # Array measurement
                measurements_group.create_dataset(
                    name,
                    data=array,
                    compression='gzip',
                    compression_opts=9
                )

@staticmethod
def read(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read data from HDF5 file
    
    Returns:
        Dictionary with 'metadata', 'measurements', and 'results'
    """
    with h5py.File(filepath, 'r') as f:
        # Read metadata
        metadata = {key: f.attrs[key] for key in f.attrs.keys()}
        
        # Read measurements
        measurements = {}
        if 'measurements' in f:
            for key in f['measurements'].keys():
                measurements[key] = f['measurements'][key][:]
        
        # Read results
        results = {}
        if 'results' in f:
            for key in f['results'].keys():
                results[key] = f['results'][key][()]
        
        return {
            'metadata': metadata,
            'measurements': measurements,
            'results': results
        }

class CSVHandler:
“”“CSV file handler with schema validation”””

@staticmethod
def write(
    filepath: Union[str, Path],
    data: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write DataFrame to CSV with metadata header
    
    Args:
        filepath: Output file path
        data: Pandas DataFrame
        metadata: Optional metadata (stored as comment header)
    """
    with open(filepath, 'w') as f:
        # Write metadata as comments
        if metadata:
            f.write(f"# Metadata\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
        
        # Write DataFrame
        data.to_csv(f, index=False)

@staticmethod
def read(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read CSV with metadata
    
    Returns:
        Dictionary with 'metadata' and 'data'
    """
    metadata = {}
    data_start_line = 0
    
    # Parse metadata from comments
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
                data_start_line = i + 1
            else:
                break
    
    # Read data
    data = pd.read_csv(filepath, skiprows=data_start_line)
    
    return {
        'metadata': metadata,
        'data': data
    }

class JCAMPDXHandler:
“””
JCAMP-DX file handler for spectroscopy data

JCAMP-DX format is a standard for spectroscopy (FTIR, NMR, etc.)
"""

@staticmethod
def write(
    filepath: Union[str, Path],
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_units: str = "cm-1",
    y_units: str = "absorbance",
    title: str = "Spectrum",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Write spectrum to JCAMP-DX format"""
    with open(filepath, 'w') as f:
        # Header
        f.write("##TITLE=" + title + "\n")
        f.write("##JCAMP-DX=5.0\n")
        f.write("##DATA TYPE=INFRARED SPECTRUM\n")
        f.write(f"##XUNITS={x_units}\n")
        f.write(f"##YUNITS={y_units}\n")
        f.write(f"##FIRSTX={x_data[0]}\n")
        f.write(f"##LASTX={x_data[-1]}\n")
        f.write(f"##NPOINTS={len(x_data)}\n")
        
        # Custom metadata
        if metadata:
            for key, value in metadata.items():
                f.write(f"##${key}={value}\n")
        
        # Data block
        f.write("##XYDATA=(X++(Y..Y))\n")
        for i in range(0, len(x_data), 10):
            x_chunk = x_data[i:i+10]
            y_chunk = y_data[i:i+10]
            f.write(f"{x_chunk[0]:.6f} ")
            f.write(" ".join([f"{y:.6f}" for y in y_chunk]))
            f.write("\n")
        
        f.write("##END=\n")

@staticmethod
def read(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Read JCAMP-DX file"""
    metadata = {}
    x_data = []
    y_data = []
    
    with open(filepath, 'r') as f:
        in_data_block = False
        
        for line in f:
            line = line.strip()
            
            if line.startswith("##"):
                if "=" in line:
                    key, value = line[2:].split("=", 1)
                    metadata[key] = value
                
                if "XYDATA" in line:
                    in_data_block = True
                    continue
                elif "END" in line:
                    break
            
            elif in_data_block and line:
                parts = line.split()
                if parts:
                    x_val = float(parts[0])
                    y_vals = [float(y) for y in parts[1:]]
                    x_data.extend([x_val] * len(y_vals))
                    y_data.extend(y_vals)
    
    return {
        'metadata': metadata,
        'x_data': np.array(x_data),
        'y_data': np.array(y_data)
    }

class NPZHandler:
“”“NumPy .npz file handler (compressed arrays)”””

@staticmethod
def write(filepath: Union[str, Path], **arrays: np.ndarray) -> None:
    """Write multiple arrays to .npz file"""
    np.savez_compressed(filepath, **arrays)

@staticmethod
def read(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read arrays from .npz file"""
    with np.load(filepath) as data:
        return {key: data[key] for key in data.keys()}

# ============================================================================

# Data Integrity

# ============================================================================

def compute_file_hash(filepath: Union[str, Path, BinaryIO]) -> str:
“””
Compute SHA256 hash of file

Args:
    filepath: Path to file or file-like object
    
Returns:
    Hex string of SHA256 hash
"""
sha256_hash = hashlib.sha256()

if isinstance(filepath, (str, Path)):
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
else:
    # File-like object
    for byte_block in iter(lambda: filepath.read(4096), b""):
        sha256_hash.update(byte_block)

return sha256_hash.hexdigest()

def verify_file_integrity(filepath: Union[str, Path], expected_hash: str) -> bool:
“”“Verify file integrity against expected hash”””
actual_hash = compute_file_hash(filepath)
return actual_hash == expected_hash

# ============================================================================

# Storage Client (S3/MinIO abstraction)

# ============================================================================

class StorageClient:
“””
Abstraction layer for S3/MinIO storage

Note: This is a simplified implementation. In production, use boto3.
"""

def __init__(self, config: StorageConfig):
    self.config = config
    self.path_manager = StoragePathManager(config.bucket_name)
    
    # In production, initialize boto3 client here:
    # import boto3
    # self.client = boto3.client('s3', ...)

def upload_file(
    self,
    local_path: Union[str, Path],
    storage_key: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Upload file to object storage
    
    Args:
        local_path: Local file path
        storage_key: Object key in storage
        metadata: Optional metadata
        
    Returns:
        Storage URI
    """
    # Compute hash
    file_hash = compute_file_hash(local_path)
    file_size = Path(local_path).stat().st_size
    
    # Create metadata sidecar
    file_metadata = FileMetadata(
        filename=Path(local_path).name,
        file_size=file_size,
        mime_type=self._get_mime_type(local_path),
        file_hash=file_hash,
        storage_uri=f"s3://{self.config.bucket_name}/{storage_key}",
        created_at=datetime.utcnow().isoformat(),
        custom=metadata
    )
    
    # Upload file (pseudo-code, replace with boto3)
    # self.client.upload_file(local_path, self.config.bucket_name, storage_key)
    
    # Upload metadata sidecar
    metadata_key = f"{storage_key}.metadata.json"
    # self.client.put_object(
    #     Bucket=self.config.bucket_name,
    #     Key=metadata_key,
    #     Body=file_metadata.to_json()
    # )
    
    return file_metadata.storage_uri

def download_file(
    self,
    storage_key: str,
    local_path: Union[str, Path]
) -> None:
    """Download file from object storage"""
    # self.client.download_file(self.config.bucket_name, storage_key, local_path)
    pass

def get_metadata(self, storage_key: str) -> FileMetadata:
    """Retrieve metadata sidecar"""
    metadata_key = f"{storage_key}.metadata.json"
    # response = self.client.get_object(Bucket=self.config.bucket_name, Key=metadata_key)
    # metadata_json = response['Body'].read().decode('utf-8')
    # return FileMetadata.from_json(metadata_json)
    pass

def delete_file(self, storage_key: str) -> None:
    """Delete file and metadata"""
    # self.client.delete_object(Bucket=self.config.bucket_name, Key=storage_key)
    # self.client.delete_object(Bucket=self.config.bucket_name, Key=f"{storage_key}.metadata.json")
    pass

def _get_mime_type(self, filepath: Union[str, Path]) -> str:
    """Determine MIME type from file extension"""
    suffix = Path(filepath).suffix.lower()
    mime_types = {
        '.h5': 'application/x-hdf5',
        '.hdf5': 'application/x-hdf5',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.npz': 'application/x-numpy',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.pdf': 'application/pdf',
    }
    return mime_types.get(suffix, 'application/octet-stream')

# ============================================================================

# Example Usage

# ============================================================================

def example_usage():
“”“Demonstrate storage and file format handlers”””
print(”=” * 80)
print(“Storage & File Format Handlers - Example Usage”)
print(”=” * 80)

# 1. HDF5 Example
print("\n1. HDF5 File Handling:")

# Generate synthetic I-V data
voltage = np.linspace(0, 1.0, 100)
current = 1e-12 * (np.exp(voltage / 0.026) - 1) + np.random.normal(0, 1e-11, 100)

# Write to HDF5
data = {
    'voltage': voltage,
    'current': current,
    'time': np.arange(100) * 0.01
}
metadata = {
    'run_id': 'test-001',
    'instrument': 'SMU-001',
    'sample': 'Diode-Si-001',
    'temperature': 300.0
}

HDF5Handler.write('test_data.h5', data, metadata)
print(f"   Written test_data.h5 ({Path('test_data.h5').stat().st_size} bytes)")

# Read back
read_data = HDF5Handler.read('test_data.h5')
print(f"   Read back: {len(read_data['measurements'])} measurements, {len(read_data['results'])} results")

# 2. CSV Example
print("\n2. CSV File Handling:")
df = pd.DataFrame({
    'voltage': voltage,
    'current': current
})
CSVHandler.write('test_data.csv', df, metadata)
print(f"   Written test_data.csv")

# 3. File Hash
print("\n3. File Integrity:")
file_hash = compute_file_hash('test_data.h5')
print(f"   SHA256: {file_hash}")
print(f"   Verification: {verify_file_integrity('test_data.h5', file_hash)}")

# 4. Storage Path Management
print("\n4. Storage Path Management:")
path_mgr = StoragePathManager()
run_path = path_mgr.get_run_path(
    organization_id="org-123",
    project_id="proj-456",
    run_id="run-789",
    filename="raw_data.h5"
)
print(f"   Run path: {run_path}")

report_path = path_mgr.get_report_path(
    organization_id="org-123",
    report_id="report-001",
    filename="summary.pdf"
)
print(f"   Report path: {report_path}")

# 5. Metadata Sidecar
print("\n5. Metadata Sidecar:")
file_meta = FileMetadata(
    filename="raw_data.h5",
    file_size=12345,
    mime_type="application/x-hdf5",
    file_hash=file_hash,
    storage_uri="s3://bucket/path/to/file.h5",
    created_at=datetime.utcnow().isoformat(),
    run_id="run-789",
    units={'voltage': 'V', 'current': 'A'},
    dimensions={'points': 100}
)
print(f"   Metadata JSON:\n{file_meta.to_json()[:200]}...")

print("\n" + "=" * 80)
print("Storage handlers demonstration complete!")
print("=" * 80)

if **name** == “**main**”:
example_usage()