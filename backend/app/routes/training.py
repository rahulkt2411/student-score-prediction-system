import os
import json
from fastapi import APIRouter, HTTPException

from app.schemas import TrainingConfigRequest
from app.ml.trainer import ModelTrainer
from app.config import (
    DEFAULT_DATASET_PATH,
    UPLOADS_DIR,
    TRAINING_METRICS_PATH,
    ACTIVE_MODEL_META_PATH
)
from app.ml.predictor import predictor

router = APIRouter(prefix="/train", tags=["Training"])

@router.get("/status")
def get_training_status():
    is_trained = os.path.exists(TRAINING_METRICS_PATH) and os.path.exists(ACTIVE_MODEL_META_PATH)
    active_model = "None"
    last_trained = "Never"
    if is_trained:
        try:
            with open(ACTIVE_MODEL_META_PATH, "r") as f:
                meta = json.load(f)
                active_model = meta.get("active_model_name", "None")
                last_trained = meta.get("last_trained", "Unknown")
        except Exception:
            pass

    return {
        "is_trained": is_trained,
        "active_model": active_model,
        "last_trained": last_trained
    }

@router.post("")
def start_model_training(config: TrainingConfigRequest):
    # Resolve dataset path
    if config.dataset_name == "loan_dataset.csv" or not config.dataset_name:
        dataset_path = DEFAULT_DATASET_PATH
    else:
        dataset_path = UPLOADS_DIR / config.dataset_name
        if not os.path.exists(dataset_path):
            dataset_path = DEFAULT_DATASET_PATH

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    try:
        trainer = ModelTrainer(
            dataset_path=str(dataset_path),
            target_column=config.target_column or "Loan_Status",
            test_size=config.test_size,
            cv_folds=config.cv_folds,
            models_to_train=config.models_to_train,
            tune_hyperparameters=config.tune_hyperparameters
        )
        results = trainer.run_training_pipeline()
        try:
            predictor.reload()
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"Successfully trained {len(trainer.trained_models)} classification models.",
            "data": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
