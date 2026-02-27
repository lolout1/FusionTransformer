"""Service layer for testing framework.

Provides business logic that can be used by:
- Streamlit app (current)
- FastAPI endpoints (future)
- CLI interface
- Python scripts

This separation enables easy migration to different frontends.
"""

from .analysis_service import AnalysisService
from .inference_service import InferenceService

__all__ = ['AnalysisService', 'InferenceService']
