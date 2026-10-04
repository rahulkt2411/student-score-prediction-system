import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = APP_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SAVED_MODELS_DIR = APP_DIR / "saved_models"

DEFAULT_DATASET_PATH = DATA_DIR / "loan_dataset.csv"
ACTIVE_MODEL_META_PATH = SAVED_MODELS_DIR / "active_model.json"
TRAINING_METRICS_PATH = SAVED_MODELS_DIR / "training_metrics.json"
PREPROCESSOR_PATH = SAVED_MODELS_DIR / "preprocessor.joblib"
MODELS_PATH = SAVED_MODELS_DIR / "trained_models.joblib"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Project Information
PROJECT_INFO = {
    "title": "LoanIQ - AI-Powered Loan Approval Prediction System",
    "course": "CSE - Artificial Intelligence & Machine Learning (Minor Project)",
    "version": "1.0.0",
    "description": "Production-grade machine learning system for loan eligibility prediction, cross-validation, hyperparameter tuning, and explainable AI analytics."
}
