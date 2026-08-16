import pandas as pd
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift():
    # Load original training data
    reference_data = pd.read_csv("data/train_reference.csv")
    
    # Pull recent production requests from your database
    current_data = pd.read_csv("data/recent_production_logs.csv") 
    
    # Generate drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    drift_result = report.as_dict()
    is_drifted = drift_result["metrics"][0]["result"]["dataset_drift"]
    
    if is_drifted:
        print("Data drift detected! Triggering retraining pipeline...")
        trigger_retraining_webhook()

def trigger_retraining_webhook():
    # Trigger the GitHub Actions workflow using a Personal Access Token
    url = "https://api.github.com/repos/YourUser/fraud-detection-mlops/actions/workflows/retrain.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token YOUR_GITHUB_PAT"
    }
    requests.post(url, headers=headers, json={"ref": "main"})
