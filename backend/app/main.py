import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure backend and root directories are on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent
_ROOT_DIR = _BACKEND_DIR.parent

for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    PROJECT_INFO,
    DEFAULT_DATASET_PATH,
    TRAINING_METRICS_PATH,
    ACTIVE_MODEL_META_PATH
)
from app.ml.trainer import ModelTrainer
from app.routes import dataset, training, models_info, predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup check: If models are not trained yet, train them automatically on default dataset
    if not os.path.exists(TRAINING_METRICS_PATH) or not os.path.exists(ACTIVE_MODEL_META_PATH):
        print("[LoanIQ] Initializing default ML models on standard loan dataset...")
        try:
            trainer = ModelTrainer(
                dataset_path=str(DEFAULT_DATASET_PATH),
                target_column="Loan_Status",
                test_size=0.2,
                cv_folds=5,
                tune_hyperparameters=True
            )
            trainer.run_training_pipeline()
            print(f"[LoanIQ] Initial training complete! Best model: {trainer.best_model_name}")
        except Exception as e:
            print(f"[LoanIQ] Warning during initial model training: {e}")
    else:
        print("[LoanIQ] Pre-trained models and artifacts loaded successfully.")

    yield
    print("[LoanIQ] Shutting down server.")

app = FastAPI(
    title=PROJECT_INFO["title"],
    description=PROJECT_INFO["description"],
    version=PROJECT_INFO["version"],
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Register routers
app.include_router(dataset.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(models_info.router, prefix="/api")
app.include_router(predict.router, prefix="/api")

@app.get("/")
def read_root():
    return {
        "service": PROJECT_INFO["title"],
        "status": "online",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "health_check": "/api/health",
        "version": PROJECT_INFO["version"]
    }

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "LoanIQ Machine Learning API",
        "models_trained": os.path.exists(TRAINING_METRICS_PATH),
        "version": PROJECT_INFO["version"]
    }

@app.get("/api/info")
def get_project_info():
    return PROJECT_INFO

