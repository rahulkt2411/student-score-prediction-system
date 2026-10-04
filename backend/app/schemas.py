from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class LoanPredictionInput(BaseModel):
    Gender: str = Field(..., description="Gender: 'Male' or 'Female'")
    Married: str = Field(..., description="Marital status: 'Yes' or 'No'")
    Dependents: str = Field(..., description="Number of dependents: '0', '1', '2', '3+'")
    Education: str = Field(..., description="Education: 'Graduate' or 'Not Graduate'")
    Self_Employed: str = Field(..., description="Self employed: 'Yes' or 'No'")
    ApplicantIncome: float = Field(..., ge=0, description="Monthly applicant income in INR/currency")
    CoapplicantIncome: float = Field(0.0, ge=0, description="Monthly coapplicant income in INR/currency")
    LoanAmount: float = Field(..., gt=0, description="Loan amount in thousands (e.g. 150 for 150,000)")
    Loan_Amount_Term: float = Field(360.0, gt=0, description="Term of loan in months (e.g. 360 for 30 yrs)")
    Credit_History: float = Field(..., description="1.0 for Good credit history, 0.0 for Bad credit history")
    Property_Area: str = Field(..., description="Property location: 'Urban', 'Semiurban', 'Rural'")
    model_name: Optional[str] = Field(None, description="Optional model to use for prediction (defaults to active best model)")

    @field_validator('Credit_History')
    @classmethod
    def validate_credit(cls, v):
        if v not in [0.0, 1.0, 0, 1]:
            raise ValueError("Credit_History must be 1.0 (Good) or 0.0 (Bad)")
        return float(v)

    @field_validator('Property_Area')
    @classmethod
    def validate_property(cls, v):
        valid = ["Urban", "Semiurban", "Rural"]
        if v not in valid:
            raise ValueError(f"Property_Area must be one of {valid}")
        return v

class FactorExplanation(BaseModel):
    factor: str
    impact: str # "Positive" or "Negative"
    weight: float
    description: str

class PredictionResponse(BaseModel):
    prediction: str # "APPROVED" or "REJECTED"
    is_approved: bool
    probability: float
    confidence_pct: float
    model_used: str
    risk_level: str # "Low", "Moderate", "High"
    disclaimer: str
    input_summary: Dict[str, Any]
    factors: List[FactorExplanation]
    total_income: float
    loan_to_income_ratio: float
    emi_estimate: float

class TrainingConfigRequest(BaseModel):
    dataset_name: Optional[str] = "loan_dataset.csv"
    target_column: Optional[str] = "Loan_Status"
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(5, ge=2, le=10)
    models_to_train: List[str] = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors"
    ]
    tune_hyperparameters: bool = True

class ModelComparisonItem(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_mean_accuracy: float
    cv_std_accuracy: float
    cv_mean_f1: float
    is_best_model: bool
    best_params: Optional[Dict[str, Any]] = None
    baseline_accuracy: Optional[float] = None
    baseline_f1: Optional[float] = None

class ModelComparisonResponse(BaseModel):
    models: List[ModelComparisonItem]
    best_model_name: str
    best_metric_used: str
    best_model_reason: str
    active_model_name: str
    dataset_name: str
    training_timestamp: str

class SetActiveModelRequest(BaseModel):
    model_name: str
