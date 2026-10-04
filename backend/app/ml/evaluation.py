import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_model_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Classification Report.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = 0.0
    if y_proba is not None:
        try:
            # Check if multi-column probability or single
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                proba_pos = y_proba[:, 1]
            else:
                proba_pos = y_proba
            roc_auc = float(roc_auc_score(y_true, proba_pos))
        except Exception:
            roc_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Ensure standard 2x2 shape
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(cm[0, 0]) if cm.shape[0] > 0 else 0
        fp, fn, tp = 0, 0, 0

    total_samples = len(y_true)
    cm_dict = {
        "matrix": [[int(val) for val in row] for row in cm],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "tn_pct": round((tn / total_samples) * 100, 1) if total_samples > 0 else 0,
        "fp_pct": round((fp / total_samples) * 100, 1) if total_samples > 0 else 0,
        "fn_pct": round((fn / total_samples) * 100, 1) if total_samples > 0 else 0,
        "tp_pct": round((tp / total_samples) * 100, 1) if total_samples > 0 else 0,
    }

    # Classification report
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm_dict,
        "classification_report": clf_report
    }

def format_cv_results(cv_scores: List[float], cv_f1_scores: List[float] = None) -> Dict[str, Any]:
    """Formats cross-validation fold scores into clean serializable output."""
    scores_arr = np.array(cv_scores)
    result = {
        "folds": [round(float(s), 4) for s in cv_scores],
        "mean_accuracy": round(float(scores_arr.mean()), 4),
        "std_accuracy": round(float(scores_arr.std()), 4),
    }
    if cv_f1_scores:
        f1_arr = np.array(cv_f1_scores)
        result["f1_folds"] = [round(float(s), 4) for s in cv_f1_scores]
        result["mean_f1"] = round(float(f1_arr.mean()), 4)
        result["std_f1"] = round(float(f1_arr.std()), 4)
    else:
        result["mean_f1"] = result["mean_accuracy"]
        result["std_f1"] = result["std_accuracy"]
        
    return result
