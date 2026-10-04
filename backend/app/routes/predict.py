from fastapi import APIRouter, HTTPException
from app.schemas import LoanPredictionInput, PredictionResponse
from app.ml.predictor import predictor

router = APIRouter(prefix="", tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict_loan_eligibility(application: LoanPredictionInput):
    try:
        # Additional safety validation
        if application.ApplicantIncome < 0:
            raise HTTPException(status_code=422, detail="Applicant Income cannot be negative.")
        if application.CoapplicantIncome < 0:
            raise HTTPException(status_code=422, detail="Coapplicant Income cannot be negative.")
        if application.LoanAmount <= 0:
            raise HTTPException(status_code=422, detail="Loan Amount must be greater than zero.")
        if application.Loan_Amount_Term <= 0:
            raise HTTPException(status_code=422, detail="Loan Amount Term must be positive.")

        result = predictor.predict(
            input_data=application.model_dump(),
            requested_model_name=application.model_name
        )
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
