"""
Core processing modules:
- AgricultureDiagnosticPipeline: Orchestrator
- GeminiVisionFallback: Smart cloud fallback (Gemini 2.0/1.5)
- AdvisoryEngine: Agricultural treatment & cultural advisory generator
"""

from .advisory_engine import AdvisoryEngine
from .gemini_fallback import GeminiVisionFallback
from .pipeline import AgricultureDiagnosticPipeline

__all__ = [
    "AdvisoryEngine",
    "GeminiVisionFallback",
    "AgricultureDiagnosticPipeline"
]
