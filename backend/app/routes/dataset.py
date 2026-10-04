import os
import shutil
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.config import DATA_DIR, UPLOADS_DIR, DEFAULT_DATASET_PATH
from app.ml.preprocessor import DataPreprocessor

router = APIRouter(prefix="/dataset", tags=["Dataset"])

@router.get("/list")
def list_datasets():
    datasets = [{"name": "loan_dataset.csv", "is_default": True, "path": str(DEFAULT_DATASET_PATH)}]
    if os.path.exists(UPLOADS_DIR):
        for f in os.listdir(UPLOADS_DIR):
            if f.endswith(".csv"):
                datasets.append({
                    "name": f,
                    "is_default": False,
                    "path": str(UPLOADS_DIR / f)
                })
    return {"datasets": datasets}

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    safe_filename = "".join([c for c in file.filename if c.isalnum() or c in (".", "_", "-")])
    target_path = UPLOADS_DIR / safe_filename

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Quick validation check on CSV readability
        df = pd.read_csv(target_path)
        if len(df) < 10:
            os.remove(target_path)
            raise HTTPException(status_code=400, detail="Uploaded dataset contains fewer than 10 rows.")
            
        return {
            "message": f"Dataset '{safe_filename}' uploaded successfully.",
            "filename": safe_filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@router.get("/summary")
def get_dataset_summary(
    dataset_name: Optional[str] = Query("loan_dataset.csv"),
    target_column: Optional[str] = Query("Loan_Status")
):
    if dataset_name == "loan_dataset.csv" or not dataset_name:
        dataset_path = DEFAULT_DATASET_PATH
    else:
        dataset_path = UPLOADS_DIR / dataset_name
        if not os.path.exists(dataset_path):
            dataset_path = DEFAULT_DATASET_PATH

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = pd.read_csv(dataset_path)
    
    # Auto-detect target column if not found
    if target_column not in df.columns:
        possible_targets = [c for c in df.columns if c.lower() in ["loan_status", "status", "approved", "target", "loan_approval"]]
        if possible_targets:
            target_column = possible_targets[0]
        else:
            target_column = df.columns[-1]

    preprocessor = DataPreprocessor(target_column=target_column)
    preprocessor.detect_columns(df)
    inspect_data = preprocessor.inspect_dataset(df)

    # Preview records (convert NaN to None for JSON compliance)
    preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")

    # Generate analytical distributions for charts
    # 1. Target distribution
    target_chart_data = []
    if target_column in df.columns:
        counts = df[target_column].value_counts().to_dict()
        for k, v in counts.items():
            label = "Approved (Y)" if str(k).upper() in ["Y", "1", "YES", "APPROVED"] else "Rejected (N)"
            target_chart_data.append({"name": label, "count": int(v)})

    # 2. Income distribution bins
    income_bins = [0, 2500, 5000, 7500, 10000, 15000, 100000]
    income_labels = ["< 2.5k", "2.5k - 5k", "5k - 7.5k", "7.5k - 10k", "10k - 15k", "> 15k"]
    income_dist = []
    if "ApplicantIncome" in df.columns:
        cut_income = pd.cut(df["ApplicantIncome"].dropna(), bins=income_bins, labels=income_labels)
        counts_income = cut_income.value_counts(sort=False).to_dict()
        income_dist = [{"range": k, "count": int(v)} for k, v in counts_income.items()]

    # 3. Loan Amount distribution bins
    loan_bins = [0, 50, 100, 150, 200, 300, 1000]
    loan_labels = ["< 50k", "50k - 100k", "100k - 150k", "150k - 200k", "200k - 300k", "> 300k"]
    loan_dist = []
    if "LoanAmount" in df.columns:
        cut_loan = pd.cut(df["LoanAmount"].dropna(), bins=loan_bins, labels=loan_labels)
        counts_loan = cut_loan.value_counts(sort=False).to_dict()
        loan_dist = [{"range": k, "count": int(v)} for k, v in counts_loan.items()]

    # 4. Credit History vs Approval
    credit_vs_approval = []
    if "Credit_History" in df.columns and target_column in df.columns:
        for ch_val, ch_lbl in [(1.0, "Good Credit (1.0)"), (0.0, "Poor Credit (0.0)")]:
            sub = df[df["Credit_History"] == ch_val]
            if len(sub) > 0:
                appr_count = int(sub[target_column].astype(str).str.upper().isin(["Y", "1", "YES"]).sum())
                rej_count = len(sub) - appr_count
                credit_vs_approval.append({
                    "category": ch_lbl,
                    "approved": appr_count,
                    "rejected": rej_count,
                    "approval_rate": round((appr_count / len(sub)) * 100, 1)
                })

    # 5. Education vs Approval
    edu_vs_approval = []
    if "Education" in df.columns and target_column in df.columns:
        for edu in df["Education"].dropna().unique():
            sub = df[df["Education"] == edu]
            if len(sub) > 0:
                appr_count = int(sub[target_column].astype(str).str.upper().isin(["Y", "1", "YES"]).sum())
                rej_count = len(sub) - appr_count
                edu_vs_approval.append({
                    "category": str(edu),
                    "approved": appr_count,
                    "rejected": rej_count,
                    "approval_rate": round((appr_count / len(sub)) * 100, 1)
                })

    # 6. Property Area vs Approval
    area_vs_approval = []
    if "Property_Area" in df.columns and target_column in df.columns:
        for area in df["Property_Area"].dropna().unique():
            sub = df[df["Property_Area"] == area]
            if len(sub) > 0:
                appr_count = int(sub[target_column].astype(str).str.upper().isin(["Y", "1", "YES"]).sum())
                rej_count = len(sub) - appr_count
                area_vs_approval.append({
                    "category": str(area),
                    "approved": appr_count,
                    "rejected": rej_count,
                    "approval_rate": round((appr_count / len(sub)) * 100, 1)
                })

    # Overall key metrics
    total_apps = len(df)
    approved_total = 0
    if target_column in df.columns:
        approved_total = int(df[target_column].astype(str).str.upper().isin(["Y", "1", "YES"]).sum())
    rejected_total = total_apps - approved_total
    approval_rate = round((approved_total / total_apps) * 100, 1) if total_apps > 0 else 0.0
    avg_income = round(float(df["ApplicantIncome"].mean()), 2) if "ApplicantIncome" in df.columns else 0.0
    avg_loan = round(float(df["LoanAmount"].mean()), 2) if "LoanAmount" in df.columns else 0.0

    return {
        "dataset_name": dataset_name,
        "summary": inspect_data,
        "preview": preview,
        "metrics": {
            "total_applications": total_apps,
            "approved": approved_total,
            "rejected": rejected_total,
            "approval_rate": approval_rate,
            "average_applicant_income": avg_income,
            "average_loan_amount": avg_loan
        },
        "charts": {
            "target_distribution": target_chart_data,
            "income_distribution": income_dist,
            "loan_amount_distribution": loan_dist,
            "credit_vs_approval": credit_vs_approval,
            "education_vs_approval": edu_vs_approval,
            "property_area_vs_approval": area_vs_approval
        }
    }
