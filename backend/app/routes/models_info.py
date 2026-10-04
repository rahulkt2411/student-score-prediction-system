import os
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.schemas import SetActiveModelRequest
from app.config import (
    TRAINING_METRICS_PATH,
    ACTIVE_MODEL_META_PATH
)
from app.ml.predictor import predictor

router = APIRouter(prefix="", tags=["Model Information"])

def _load_metrics():
    if not os.path.exists(TRAINING_METRICS_PATH):
        raise HTTPException(
            status_code=400,
            detail="Models have not been trained yet. Please trigger model training first."
        )
    with open(TRAINING_METRICS_PATH, "r") as f:
        return json.load(f)

def _get_active_model_name():
    if os.path.exists(ACTIVE_MODEL_META_PATH):
        try:
            with open(ACTIVE_MODEL_META_PATH, "r") as f:
                data = json.load(f)
                return data.get("active_model_name", "Random Forest")
        except Exception:
            pass
    return "Random Forest"

@router.get("/models/comparison")
def get_model_comparison():
    data = _load_metrics()
    active_model = _get_active_model_name()

    return {
        "models": data.get("models_comparison", []),
        "best_model_name": data.get("best_model_name", "Random Forest"),
        "best_metric_used": data.get("best_metric_used", "Composite Score"),
        "best_model_reason": data.get("best_model_reason", ""),
        "active_model_name": active_model,
        "training_timestamp": data.get("timestamp", ""),
        "training_duration_seconds": data.get("training_duration_seconds", 0)
    }

@router.get("/evaluation")
def get_model_evaluation(model_name: Optional[str] = Query(None)):
    data = _load_metrics()
    active_model = _get_active_model_name()
    selected = model_name or active_model

    detailed_evals = data.get("detailed_evaluations", {})
    if selected not in detailed_evals:
        # Fallback to first available
        if detailed_evals:
            selected = list(detailed_evals.keys())[0]
        else:
            raise HTTPException(status_code=404, detail=f"No evaluation found for '{selected}'")

    eval_info = detailed_evals[selected]

    return {
        "model_name": selected,
        "active_model_name": active_model,
        "available_models": list(detailed_evals.keys()),
        "metrics": {
            "accuracy": eval_info["accuracy"],
            "precision": eval_info["precision"],
            "recall": eval_info["recall"],
            "f1_score": eval_info["f1_score"],
            "roc_auc": eval_info["roc_auc"],
            "baseline_accuracy": eval_info.get("baseline_accuracy"),
            "baseline_f1": eval_info.get("baseline_f1"),
            "best_params": eval_info.get("best_params")
        },
        "confusion_matrix": eval_info["confusion_matrix"],
        "classification_report": eval_info["classification_report"],
        "cross_validation": eval_info.get("cv_stats", {})
    }

@router.get("/feature-importance")
def get_feature_importance(model_name: Optional[str] = Query(None)):
    data = _load_metrics()
    active_model = _get_active_model_name()
    selected = model_name or active_model

    all_importances = data.get("feature_importances", {})
    if selected not in all_importances:
        if all_importances:
            selected = list(all_importances.keys())[0]
        else:
            raise HTTPException(status_code=404, detail=f"No feature importance data found for '{selected}'")

    return {
        "model_name": selected,
        "available_models": list(all_importances.keys()),
        "factors": all_importances[selected],
        "methodology": "Model-specific feature importance (Gini Impurity for Trees, Normalized Absolute Coefficients for Linear Models).",
        "causation_disclaimer": "Feature importance indicates predictive weight within this specific model and does not establish statistical causation."
    }

@router.post("/models/set-active")
def set_active_model(req: SetActiveModelRequest):
    data = _load_metrics()
    available = [m["name"] for m in data.get("models_comparison", [])]
    if req.model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_name}' is not in trained models: {available}"
        )

    active_meta = {
        "active_model_name": req.model_name,
        "dataset": data.get("dataset_path", "")
    }
    with open(ACTIVE_MODEL_META_PATH, "w") as f:
        json.dump(active_meta, f, indent=2)

    try:
        predictor.reload()
    except Exception:
        pass

    return {
        "message": f"Active model successfully changed to '{req.model_name}'.",
        "active_model_name": req.model_name
    }
