"""Application configuration."""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "SPECTRA-Lab"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://spectra:spectra_secret@localhost:5432/spectra_lab",
        env="DATABASE_URL"
    )
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-here-change-in-production",
        env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # MinIO
    MINIO_ENDPOINT: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(default="admin", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="admin123", env="MINIO_SECRET_KEY")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    
    # Hardware Configuration
    ENABLE_HARDWARE: bool = Field(default=False, env="ENABLE_HARDWARE")
    IMPLANT_CONNECTION: str = Field(
        default="TCPIP::192.168.1.100::5025::SOCKET",
        env="IMPLANT_CONNECTION"
    )
    RTP_CONNECTION: str = Field(
        default="opc.tcp://192.168.1.101:4840",
        env="RTP_CONNECTION"
    )
    
    # Simulation
    SIMULATION_MODE: str = Field(default="realistic", env="SIMULATION_MODE")
    ENABLE_FAULTS: bool = Field(default=False, env="ENABLE_FAULTS")
    
    # Telemetry
    TELEMETRY_BUFFER_SIZE: int = Field(default=10000, env="TELEMETRY_BUFFER_SIZE")
    TELEMETRY_RETENTION_DAYS: int = Field(default=30, env="TELEMETRY_RETENTION_DAYS")
    
    # SPC Configuration
    SPC_WINDOW_SIZE: int = Field(default=20, env="SPC_WINDOW_SIZE")
    SPC_ALERT_COOLDOWN_MINUTES: int = Field(default=15, env="SPC_ALERT_COOLDOWN_MINUTES")
    
    # VM Configuration
    VM_MODEL_PATH: str = Field(default="/app/models", env="VM_MODEL_PATH")
    VM_FEATURE_WINDOW: int = Field(default=100, env="VM_FEATURE_WINDOW")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()