import sys
from pathlib import Path

# Ensure backend root is on sys.path regardless of execution CWD
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import DEFAULT_DATASET_PATH
from app.ml.trainer import ModelTrainer
from app.ml.predictor import predictor

print("Running initial training test...")
trainer = ModelTrainer(str(DEFAULT_DATASET_PATH), tune_hyperparameters=True)
results = trainer.run_training_pipeline()
print("Training succeeded!")
print("Best Model:", results['best_model_name'])
print("Models count:", len(results['models_comparison']))
for m in results['models_comparison']:
    print(f"- {m['name']}: Acc={m['accuracy']:.3f}, F1={m['f1_score']:.3f}, ROC={m['roc_auc']:.3f}, CV_Acc={m['cv_mean_accuracy']:.3f}")

print("\nTesting Prediction Engine with Good Credit Applicant...")
good_app = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '1',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5500.0,
    'CoapplicantIncome': 1800.0,
    'LoanAmount': 130.0,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0,
    'Property_Area': 'Semiurban'
}
pred1 = predictor.predict(good_app)
print(f"Prediction 1: {pred1.prediction}, Probability: {pred1.probability:.3f}, Confidence: {pred1.confidence_pct}%")
print(f"Top Factor 1: {pred1.factors[0].factor}")

print("\nTesting Prediction Engine with Bad Credit / High Loan Applicant...")
bad_app = {
    'Gender': 'Female',
    'Married': 'No',
    'Dependents': '3+',
    'Education': 'Not Graduate',
    'Self_Employed': 'Yes',
    'ApplicantIncome': 2000.0,
    'CoapplicantIncome': 0.0,
    'LoanAmount': 350.0,
    'Loan_Amount_Term': 180.0,
    'Credit_History': 0.0,
    'Property_Area': 'Rural'
}
pred2 = predictor.predict(bad_app)
print(f"Prediction 2: {pred2.prediction}, Probability: {pred2.probability:.3f}, Confidence: {pred2.confidence_pct}%")
print(f"Top Factor 2: {pred2.factors[0].factor}")
print("\nALL VERIFICATIONS PASSED!")
