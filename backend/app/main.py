import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so modules (backend, inference, evaluation) can be imported
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NumPy 1.x / SciPy compatibility shim for scipy.sparse scalar lookups
try:
    import numpy as _np
    if not hasattr(_np, "long"):
        _np.long = int
    if not hasattr(_np, "ulong"):
        _np.ulong = int
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router as api_router
from backend.app.schemas.contracts import MANDATORY_DISCLAIMER

app = FastAPI(
    title="NeuroScope 3D API",
    description="Research and teaching platform for multi-sequence brain MRI segmentation and 3D visualization.",
    version="0.1.0",
)

# CORS Middleware Configuration - Allow local dev and Vercel domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(api_router)


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "product": "NeuroScope 3D",
        "version": "0.1.0",
        "docs_url": "/docs",
        "api_health": "/api/health",
        "disclaimer": MANDATORY_DISCLAIMER,
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True, reload_dirs=["backend"])
