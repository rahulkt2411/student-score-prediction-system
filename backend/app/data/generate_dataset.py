import os
import numpy as np
import pandas as pd

def generate_standard_loan_dataset(filepath: str, n_samples: int = 614, random_state: int = 42):
    np.random.seed(random_state)
    
    loan_ids = [f"LP00{1002 + i:04d}" for i in range(n_samples)]
    
    # Demographics
    gender = np.random.choice(["Male", "Female", None], size=n_samples, p=[0.79, 0.19, 0.02])
    married = np.random.choice(["Yes", "No", None], size=n_samples, p=[0.64, 0.35, 0.01])
    dependents = np.random.choice(["0", "1", "2", "3+", None], size=n_samples, p=[0.57, 0.17, 0.17, 0.07, 0.02])
    education = np.random.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = np.random.choice(["No", "Yes", None], size=n_samples, p=[0.82, 0.13, 0.05])
    
    # Financials (realistic log-normal / right-skewed distributions)
    applicant_income = np.round(np.random.lognormal(mean=8.2, sigma=0.6, size=n_samples)).astype(int)
    # Clip applicant income between 1500 and 80000
    applicant_income = np.clip(applicant_income, 1500, 81000)
    
    # Coapplicant income: ~45% have 0 coapplicant income, rest have some
    has_coapplicant = np.random.choice([0, 1], size=n_samples, p=[0.44, 0.56])
    coapplicant_income = np.where(
        has_coapplicant == 1,
        np.round(np.random.lognormal(mean=7.5, sigma=0.55, size=n_samples)).astype(int),
        0
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 42000)
    
    # Total Income
    total_income = applicant_income + coapplicant_income
    
    # Loan Amount (in thousands ₹ or $) - correlated with total income + noise
    base_loan = (total_income * 0.02) + np.random.normal(30, 25, size=n_samples)
    loan_amount = np.round(np.clip(base_loan, 15, 700)).astype(float)
    
    # Add some realistic missing values (~3.5% missing in LoanAmount)
    loan_amount_missing_mask = np.random.random(n_samples) < 0.035
    loan_amount[loan_amount_missing_mask] = np.nan
    
    # Loan Amount Term (in months): most are 360 (30 yrs), some 180, 240, 120, 480
    terms = [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0]
    term_probs = [0.002, 0.003, 0.004, 0.007, 0.01, 0.07, 0.02, 0.02, 0.84, 0.024]
    loan_term = np.random.choice(terms, size=n_samples, p=term_probs)
    # Add ~2% missing in Loan_Amount_Term
    loan_term_missing_mask = np.random.random(n_samples) < 0.023
    loan_term[loan_term_missing_mask] = np.nan
    
    # Credit History: 1.0 (meets guidelines) or 0.0 (does not meet)
    # ~84% 1.0, 16% 0.0, with ~8% missing
    credit_history = np.random.choice([1.0, 0.0, None], size=n_samples, p=[0.77, 0.15, 0.08])
    
    # Property Area
    property_area = np.random.choice(["Semiurban", "Urban", "Rural"], size=n_samples, p=[0.38, 0.33, 0.29])
    
    # Real ground-truth probability calculation for Loan_Status (Y / N)
    # Simulating the true lending dynamics:
    # 1. Credit History is dominant (80%+ importance)
    # 2. Debt-to-income (LoanAmount / TotalIncome)
    # 3. Education (Graduates have slight edge)
    # 4. Property Area (Semiurban has higher approval rate historically)
    # 5. Dependents (More dependents slightly increases financial burden)
    
    loan_status = []
    for i in range(n_samples):
        ch = credit_history[i]
        inc = total_income[i]
        la = loan_amount[i] if not np.isnan(loan_amount[i]) else 140.0
        d_to_i = (la * 1000) / (inc * 12) # Approximate debt to annual income
        
        score = 0.0
        
        if ch == 1.0 or ch == '1.0' or ch == 1:
            score += 2.5
        elif ch == 0.0 or ch == '0.0' or ch == 0:
            score -= 2.6
        else: # Missing credit history
            score += 0.5 if inc > 6000 else -0.5
            
        # Debt ratio influence
        if d_to_i < 1.5:
            score += 0.8
        elif d_to_i > 3.0:
            score -= 1.0
            
        # Property Area influence
        if property_area[i] == "Semiurban":
            score += 0.5
        elif property_area[i] == "Rural":
            score -= 0.3
            
        # Education influence
        if education[i] == "Graduate":
            score += 0.3
            
        # Married
        if married[i] == "Yes":
            score += 0.2
            
        # Dependents
        if dependents[i] in ["2", "3+"]:
            score -= 0.2
            
        # Random noise to prevent perfect deterministic separation
        noise = np.random.normal(0, 0.6)
        prob = 1.0 / (1.0 + np.exp(-(score + noise)))
        
        status = "Y" if prob >= 0.5 else "N"
        loan_status.append(status)
        
    df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated standard loan dataset with {len(df)} rows at {filepath}")
    print(f"Target distribution:\n{df['Loan_Status'].value_counts(normalize=True)}")
    print(f"Missing values count:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "loan_dataset.csv")
    generate_standard_loan_dataset(out_path)
