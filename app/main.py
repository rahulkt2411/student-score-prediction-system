"""
LoanIQ Main Application Root Proxy
Exports FastAPI `app` from `backend.app.main` for root-level execution.
"""
import sys
from pathlib import Path

_root_dir = Path(__file__).resolve().parent.parent
_backend_dir = _root_dir / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Re-export FastAPI application and components from backend.app.main
from backend.app.main import app, lifespan, health_check, get_project_info

__all__ = ["app", "lifespan", "health_check", "get_project_info"]
