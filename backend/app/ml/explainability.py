import numpy as np
from typing import List, Dict, Any

FEATURE_DISPLAY_NAMES = {
    "Credit_History": "Credit History",
    "ApplicantIncome": "Applicant Monthly Income",
    "CoapplicantIncome": "Coapplicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Repayment Term",
    "Property_Area": "Property Location Area",
    "Education": "Applicant Education Level",
    "Married": "Marital Status",
    "Dependents": "Number of Dependents",
    "Self_Employed": "Self-Employment Status",
    "Gender": "Applicant Gender"
}

def get_model_feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts normalized feature importance from tree-based or linear models.
    Supports Random Forest, Decision Tree, Logistic Regression, etc.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        # Tree-based algorithms (Random Forest, Decision Tree)
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression, Linear SVM)
        try:
            coef = model.coef_
            if len(coef.shape) > 1:
                importances = np.abs(coef[0])
            else:
                importances = np.abs(coef)
        except Exception:
            n = len(feature_names)
            importances = np.ones(n) / n
    else:
        # Fallback uniform/heuristic for KNN or non-linear kernels
        n = len(feature_names)
        importances = np.ones(n) / n

    # Normalize to sum to 100%
    total = np.sum(importances)
    if total > 0:
        norm_importances = (importances / total) * 100
    else:
        norm_importances = np.zeros_like(importances)

    # Map back to high-level domain features if one-hot encoded
    # e.g., 'cat__Property_Area_Semiurban' -> 'Property_Area'
    grouped_importance: Dict[str, float] = {}
    for name, imp in zip(feature_names, norm_importances):
        clean_name = name
        # Strip transformer prefixes like 'num__', 'cat__'
        if "__" in clean_name:
            clean_name = clean_name.split("__")[-1]
            
        # Group one-hot components: e.g. Property_Area_Urban -> Property_Area
        matched_group = clean_name
        for raw_feat in FEATURE_DISPLAY_NAMES.keys():
            if clean_name.startswith(raw_feat):
                matched_group = raw_feat
                break
                
        grouped_importance[matched_group] = grouped_importance.get(matched_group, 0.0) + float(imp)

    # Sort descending
    sorted_features = sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True)

    result = []
    for rank, (feat_key, weight) in enumerate(sorted_features, 1):
        display_name = FEATURE_DISPLAY_NAMES.get(feat_key, feat_key)
        result.append({
            "rank": rank,
            "feature_key": feat_key,
            "feature_name": display_name,
            "importance_pct": round(weight, 2),
            "description": get_feature_impact_summary(feat_key)
        })

    return result

def get_feature_impact_summary(feature_key: str) -> str:
    summaries = {
        "Credit_History": "Indicates whether applicant past credit obligations were fulfilled without default.",
        "ApplicantIncome": "Primary revenue stream determining capacity to service regular monthly debt obligations.",
        "LoanAmount": "Principal sum requested; directly scales monthly repayment burden and default risk exposure.",
        "Property_Area": "Locality asset valuation tier (Semiurban/Urban/Rural) affecting liquidation collateral stability.",
        "CoapplicantIncome": "Secondary financial backing that augments household debt-to-income servicing capacity.",
        "Education": "Predictor of long-term earning potential, career trajectory, and financial literacy.",
        "Loan_Amount_Term": "Duration of debt amortization; affects monthly EMI installments and lifetime interest.",
        "Dependents": "Fixed household liabilities reducing disposable monthly income available for repayment.",
        "Married": "Household stability and dual income potential vs combined living expenditures.",
        "Self_Employed": "Income volatility indicator compared to regular salaried employment.",
        "Gender": "Demographic baseline feature evaluated in accordance with non-discriminatory compliance."
    }
    return summaries.get(feature_key, "Domain factor influencing automated underwriting algorithm.")

def explain_individual_prediction(
    input_data: Dict[str, Any],
    is_approved: bool,
    probability: float
) -> List[Dict[str, Any]]:
    """
    Generates tailored, transparent factor-by-factor explanations
    for a specific individual loan prediction request.
    """
    factors = []
    
    applicant_income = float(input_data.get("ApplicantIncome", 0))
    coapplicant_income = float(input_data.get("CoapplicantIncome", 0))
    total_income = applicant_income + coapplicant_income
    loan_amount_k = float(input_data.get("LoanAmount", 100)) # in thousands
    loan_amount_val = loan_amount_k * 1000 # in absolute currency
    credit_hist = float(input_data.get("Credit_History", 1.0))
    property_area = str(input_data.get("Property_Area", "Urban"))
    education = str(input_data.get("Education", "Graduate"))
    dependents = str(input_data.get("Dependents", "0"))
    term_months = float(input_data.get("Loan_Amount_Term", 360))

    # Calculate Debt-to-Annual-Income ratio
    annual_income = total_income * 12
    dti_ratio = round((loan_amount_val / annual_income), 2) if annual_income > 0 else 999.0

    # 1. Credit History Factor
    if credit_hist >= 1.0:
        factors.append({
            "factor": "Credit History: Verified & Clean (Score 1.0)",
            "impact": "Positive",
            "weight": 42.0,
            "description": "Consistent past credit repayment track record meets prime lending guidelines, substantially elevating approval probability."
        })
    else:
        factors.append({
            "factor": "Credit History: Defaults / Insufficient (Score 0.0)",
            "impact": "Negative",
            "weight": 48.0,
            "description": "Prior credit delinquency or absence of established credit score poses significant underwriting default risk."
        })

    # 2. Income & Debt-to-Income
    if dti_ratio <= 2.2:
        factors.append({
            "factor": f"Favorable Debt-to-Income (DTI: {dti_ratio}x)",
            "impact": "Positive",
            "weight": 22.0,
            "description": f"Total household income (₹{int(total_income):,}/mo) comfortably supports requested loan amount of ₹{int(loan_amount_val):,}."
        })
    elif dti_ratio <= 3.5:
        factors.append({
            "factor": f"Moderate Debt Burden (DTI: {dti_ratio}x)",
            "impact": "Neutral",
            "weight": 14.0,
            "description": f"Loan amount is manageable relative to household earnings, though close to prudent underwriting thresholds."
        })
    else:
        factors.append({
            "factor": f"High Debt-to-Income Exposure (DTI: {dti_ratio}x)",
            "impact": "Negative",
            "weight": 28.0,
            "description": f"Requested loan amount (₹{int(loan_amount_val):,}) is disproportionately high compared to household income (₹{int(total_income):,}/mo)."
        })

    # 3. Coapplicant Support
    if coapplicant_income > 0:
        factors.append({
            "factor": f"Coapplicant Financial Backing (₹{int(coapplicant_income):,}/mo)",
            "impact": "Positive",
            "weight": 12.0,
            "description": "Secondary earning stream reduces overall default probability and expands joint repayment capability."
        })

    # 4. Property Collateral Location
    if property_area == "Semiurban":
        factors.append({
            "factor": "Property Location: Semiurban Area",
            "impact": "Positive",
            "weight": 9.0,
            "description": "Semiurban residential assets exhibit high historical approval rates and stable real estate appreciation."
        })
    elif property_area == "Rural":
        factors.append({
            "factor": "Property Location: Rural Area",
            "impact": "Neutral" if is_approved else "Negative",
            "weight": 7.0,
            "description": "Rural property market liquidity is factored into collateral valuation and risk indexing."
        })

    # 5. Education Level
    if education == "Graduate":
        factors.append({
            "factor": "Educational Qualification: Graduate",
            "impact": "Positive",
            "weight": 8.0,
            "description": "Graduate status correlates statistically with income continuity and career advancement."
        })
    else:
        factors.append({
            "factor": "Educational Qualification: Non-Graduate",
            "impact": "Neutral",
            "weight": 5.0,
            "description": "Evaluated alongside verifiable earnings and occupational tenure."
        })

    # Sort factors by absolute weight descending
    factors.sort(key=lambda x: x["weight"], reverse=True)
    return factors
