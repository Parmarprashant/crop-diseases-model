"""
API module for Agritech Cloud Multi-Model Inference:
- server.py: FastAPI app & endpoints
- schemas.py: Pydantic request/response schemas
"""

from .schemas import DiagnosisResponse, HealthResponse
from .server import app

__all__ = ["app", "DiagnosisResponse", "HealthResponse"]
