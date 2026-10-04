import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from app.ml.preprocessor import DataPreprocessor
from app.ml.explainability import explain_individual_prediction
from app.config import (
    PREPROCESSOR_PATH,
    MODELS_PATH,
    ACTIVE_MODEL_META_PATH
)
from app.schemas import PredictionResponse, FactorExplanation

class LoanPredictor:
    def __init__(self):
        self._preprocessor: Optional[DataPreprocessor] = None
        self._models: Optional[Dict[str, Any]] = None
        self._active_model_name: Optional[str] = None

    def reload(self):
        """Forces reloading models and preprocessor from disk."""
        self._preprocessor = None
        self._models = None
        self._active_model_name = None
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Loads serialized models and preprocessor pipeline on demand."""
        if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODELS_PATH):
            raise FileNotFoundError(
                "Trained models or preprocessor not found. Please train the models first using the Model Training page."
            )

        if self._preprocessor is None:
            self._preprocessor = DataPreprocessor.load(str(PREPROCESSOR_PATH))

        if self._models is None:
            self._models = joblib.load(str(MODELS_PATH))

        if os.path.exists(ACTIVE_MODEL_META_PATH):
            try:
                with open(ACTIVE_MODEL_META_PATH, "r") as f:
                    meta = json.load(f)
                    self._active_model_name = meta.get("active_model_name", "Random Forest")
            except Exception:
                self._active_model_name = "Random Forest"
        else:
            self._active_model_name = "Random Forest"

    def predict(self, input_data: Dict[str, Any], requested_model_name: Optional[str] = None) -> PredictionResponse:
        self._ensure_loaded()

        # Determine which model to use
        if requested_model_name:
            model_name = requested_model_name
        else:
            if os.path.exists(ACTIVE_MODEL_META_PATH):
                try:
                    with open(ACTIVE_MODEL_META_PATH, "r") as f:
                        meta = json.load(f)
                        self._active_model_name = meta.get("active_model_name", self._active_model_name)
                except Exception:
                    pass
            model_name = self._active_model_name

        if not self._models or model_name not in self._models:
            # Fallback to available model
            model_name = list(self._models.keys())[0] if self._models else "Random Forest"

        model = self._models[model_name]

        # Prepare single record dataframe
        record = {
            "Gender": input_data.get("Gender", "Male"),
            "Married": input_data.get("Married", "Yes"),
            "Dependents": str(input_data.get("Dependents", "0")),
            "Education": input_data.get("Education", "Graduate"),
            "Self_Employed": input_data.get("Self_Employed", "No"),
            "ApplicantIncome": float(input_data.get("ApplicantIncome", 0)),
            "CoapplicantIncome": float(input_data.get("CoapplicantIncome", 0)),
            "LoanAmount": float(input_data.get("LoanAmount", 100)),
            "Loan_Amount_Term": float(input_data.get("Loan_Amount_Term", 360)),
            "Credit_History": float(input_data.get("Credit_History", 1.0)),
            "Property_Area": input_data.get("Property_Area", "Urban"),
        }

        # Transform using fitted preprocessor pipeline
        df_single = pd.DataFrame([record])
        X_vec = self._preprocessor.transform(df_single)

        # Predict
        raw_pred = model.predict(X_vec)[0]
        
        # Determine probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_vec)[0]
            # Prob of class 1 (Approved)
            prob_approved = float(probs[1]) if len(probs) > 1 else float(probs[0])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(X_vec)[0])
            prob_approved = 1.0 / (1.0 + float(np.exp(-score)))
        else:
            prob_approved = 0.90 if raw_pred == 1 else 0.15

        is_approved = bool(raw_pred == 1)
        prediction_label = "APPROVED" if is_approved else "REJECTED"
        confidence_pct = round((prob_approved if is_approved else (1.0 - prob_approved)) * 100, 1)

        # Risk level assessment
        if is_approved:
            risk_level = "Low" if prob_approved >= 0.75 else "Moderate"
        else:
            risk_level = "High" if prob_approved <= 0.30 else "Moderate"

        # Financial derived metrics
        app_income = float(record["ApplicantIncome"])
        coapp_income = float(record["CoapplicantIncome"])
        total_income = app_income + coapp_income
        loan_amount_val = float(record["LoanAmount"]) * 1000 # convert k to currency
        annual_income = total_income * 12
        loan_to_income = round((loan_amount_val / annual_income), 2) if annual_income > 0 else 0.0

        # EMI calculation (assuming approx 8.5% annual interest for term in months)
        r = (8.5 / 12) / 100
        n = max(float(record["Loan_Amount_Term"]), 12)
        emi = round((loan_amount_val * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1), 2) if loan_amount_val > 0 else 0.0

        # Explainable factors
        factors_raw = explain_individual_prediction(record, is_approved, prob_approved)
        factors = [FactorExplanation(**f) for f in factors_raw]

        disclaimer = (
            "Prediction is based on the trained machine learning model and statistical probabilities. "
            "It does not constitute a guaranteed commercial credit sanction or legal underwriting contract."
        )

        return PredictionResponse(
            prediction=prediction_label,
            is_approved=is_approved,
            probability=round(prob_approved, 4),
            confidence_pct=confidence_pct,
            model_used=model_name,
            risk_level=risk_level,
            disclaimer=disclaimer,
            input_summary=record,
            factors=factors,
            total_income=total_income,
            loan_to_income_ratio=loan_to_income,
            emi_estimate=emi
        )

predictor = LoanPredictor()
