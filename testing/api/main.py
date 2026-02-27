"""FastAPI application (future implementation).

This is a stub showing how to migrate from Streamlit to FastAPI
using the same service layer.

Usage:
    uvicorn testing.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /analyze - Run analysis on predictions
    POST /inference - Run model inference
    GET /health - Health check
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Dict
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Fall Detection Analysis API",
        description="API for fall detection model testing and analysis",
        version="0.1.0"
    )

    # Pydantic models for API
    class HealthResponse(BaseModel):
        status: str
        version: str

    class DataLoadRequest(BaseModel):
        path: str

    class InferenceRequest(BaseModel):
        data_path: str
        config_path: str
        threshold: float = 0.5
        use_alpha_queue: bool = False

    class MetricsResponse(BaseModel):
        f1: float
        precision: float
        recall: float
        accuracy: float
        specificity: float
        tp: int
        fp: int
        tn: int
        fn: int

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="ok", version="0.1.0")

    @app.post("/inference")
    async def run_inference(request: InferenceRequest) -> Dict:
        """Run inference and return metrics.

        Example:
            curl -X POST http://localhost:8000/inference \
                -H "Content-Type: application/json" \
                -d '{"data_path": "data.json", "config_path": "config.yaml"}'
        """
        from ..services import InferenceService, AnalysisService
        from ..services.inference_service import DataLoadRequest as DLR, InferenceRequest as IR
        from ..services.analysis_service import AnalysisRequest

        try:
            # Load data
            inference_svc = InferenceService()
            data_resp = inference_svc.load_data(DLR(path=request.data_path))

            # Run inference
            inf_req = IR(
                windows=data_resp.windows,
                config_path=request.config_path,
                threshold=request.threshold,
                use_alpha_queue=request.use_alpha_queue
            )
            inf_resp = inference_svc.run_inference(inf_req)

            # Analyze
            analysis_svc = AnalysisService(threshold=request.threshold)
            analysis_req = AnalysisRequest(
                predictions=inf_resp.predictions,
                windows=data_resp.windows,
                threshold=request.threshold
            )
            result = analysis_svc.analyze(analysis_req)

            return {
                "config": request.config_path,
                "total_windows": result.total_windows,
                "metrics": {
                    k: v for k, v in result.metrics.items()
                    if isinstance(v, (int, float)) and v is not None
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/configs")
    async def list_configs() -> List[str]:
        """List available configurations."""
        from pathlib import Path
        from ..config import DEFAULT_CONFIGS_DIR

        config_dir = Path(DEFAULT_CONFIGS_DIR)
        if not config_dir.exists():
            return []

        return [c.name for c in sorted(config_dir.glob("*.yaml"))]

else:
    # Placeholder when FastAPI not installed
    app = None

    def main():
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        print("Then run: uvicorn testing.api.main:app --host 0.0.0.0 --port 8000")
