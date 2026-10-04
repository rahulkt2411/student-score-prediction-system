"""
LoanIQ Root Application Proxy Package
Allows running `python -m uvicorn app.main:app --reload` directly from the project root workspace.
Delegates all package imports and lookups to `backend/app`.
"""
import sys
from pathlib import Path

_root_dir = Path(__file__).resolve().parent.parent
_backend_dir = _root_dir / "backend"
_backend_app_dir = _backend_dir / "app"

for _p in [str(_backend_dir), str(_root_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if _backend_app_dir.exists() and str(_backend_app_dir) not in __path__:
    __path__.insert(0, str(_backend_app_dir))
