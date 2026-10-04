import json
import urllib.request
import urllib.parse

BASE_URL = "http://127.0.0.1:8000/api"

def make_req(path, data=None, method="GET"):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_suite():
    print("========================================")
    print("LoanIQ End-to-End System Test Suite")
    print("========================================")

    # 1. Health
    h = make_req("/health")
    assert h["status"] == "healthy", "Health failed"
    print("[PASS] Health Check Passed:", h)

    # 2. Datasets
    d = make_req("/dataset/list")
    assert len(d["datasets"]) >= 1, "Dataset list empty"
    print(f"[PASS] Dataset List Passed: Found {len(d['datasets'])} datasets ({d['datasets'][0]['name']})")

    # 3. Dataset Summary
    s = make_req("/dataset/summary?dataset_name=loan_dataset.csv")
    assert s["summary"]["total_rows"] == 614, "Unexpected row count"
    assert s["metrics"]["total_applications"] == 614
    print(f"[PASS] Dataset Summary Passed: {s['summary']['total_rows']} rows, {s['metrics']['approval_rate']}% approval rate")
    print(f"  Missing values handled: {list(s['summary']['missing_values'].keys())[:4]}...")

    # 4. Model Comparison
    comp = make_req("/models/comparison")
    assert len(comp["models"]) == 5, f"Expected 5 models, got {len(comp['models'])}"
    print(f"[PASS] Model Comparison Passed: 5 Models benchmarked:")
    for m in comp["models"]:
        print(f"  - {m['name']}: Accuracy={m['accuracy']*100:.1f}%, F1={m['f1_score']*100:.1f}%, 5-Fold CV={m['cv_mean_accuracy']*100:.1f}%")
    print(f"  Winner Selected by Metric: {comp['best_model_name']}")

    # 5. Model Evaluation
    ev = make_req("/evaluation")
    assert "confusion_matrix" in ev, "Confusion matrix missing"
    assert "classification_report" in ev, "Classification report missing"
    cm = ev["confusion_matrix"]
    print(f"[PASS] Model Evaluation Passed for '{ev['model_name']}':")
    print(f"  Confusion Matrix: TN={cm['true_negative']}, FP={cm['false_positive']}, FN={cm['false_negative']}, TP={cm['true_positive']}")
    print(f"  5-Fold CV Detail: {ev['cross_validation']['folds']}")

    # 6. Feature Importance
    fi = make_req("/feature-importance")
    assert len(fi["factors"]) > 0, "No factors returned"
    print(f"[PASS] Feature Importance Passed for '{fi['model_name']}':")
    for f in fi["factors"][:3]:
        print(f"  Rank {f['rank']}: {f['feature_name']} -> {f['importance_pct']}% weight")

    # 7. Prediction Test 1: Good Applicant
    good_app = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 75000.0,
        "CoapplicantIncome": 25000.0,
        "LoanAmount": 150.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
    p1 = make_req("/predict", data=good_app, method="POST")
    assert p1["is_approved"] == True, "Expected Good Applicant to be approved"
    print(f"[PASS] Prediction 1 (Prime Applicant) Passed: {p1['prediction']} ({p1['confidence_pct']}% confidence, Risk: {p1['risk_level']})")
    print(f"  Top Factor: {p1['factors'][0]['factor']} ({p1['factors'][0]['impact']})")

    # 8. Prediction Test 2: High Risk Applicant
    bad_app = {
        "Gender": "Female",
        "Married": "No",
        "Dependents": "3+",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 15000.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 400.0,
        "Loan_Amount_Term": 180.0,
        "Credit_History": 0.0,
        "Property_Area": "Rural"
    }
    p2 = make_req("/predict", data=bad_app, method="POST")
    assert p2["is_approved"] == False, "Expected High Risk Applicant to be rejected"
    print(f"[PASS] Prediction 2 (High Risk Applicant) Passed: {p2['prediction']} ({p2['confidence_pct']}% confidence, Risk: {p2['risk_level']})")
    print(f"  Top Factor: {p2['factors'][0]['factor']} ({p2['factors'][0]['impact']})")

    # 9. Set Active Model
    set_res = make_req("/models/set-active", data={"model_name": "Random Forest"}, method="POST")
    assert set_res["active_model_name"] == "Random Forest"
    print(f"[PASS] Set Active Model Passed: {set_res['message']}")

    print("========================================")
    print("ALL 9 INTEGRATION TESTS PASSED 100%!")
    print("========================================")

if __name__ == "__main__":
    test_suite()
