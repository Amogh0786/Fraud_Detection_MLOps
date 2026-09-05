from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from feast import FeatureStore

app = FastAPI(title="Fraud Detection API")

model = None
fs = None

@app.on_event("startup")
def load_resources():
    global model, fs
    try:
        model_uri = "models:/FraudDetectionModel/latest"
        model = mlflow.xgboost.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
    
    try:
        # Initialize Feast Feature Store
        fs = FeatureStore(repo_path="feature_repo")
        print("Feature Store loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load feature store: {e}")

class Transaction(BaseModel):
    user_id: int
    amount: float
    time: float

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature Store not loaded.")
        
    # Fetch historical features from Feast
    feature_vector = fs.get_online_features(
        features=[
            "user_transaction_stats:v1",
            "user_transaction_stats:v2"
        ],
        entity_rows=[{"user_id": transaction.user_id}]
    ).to_dict()
    
    # Combine request data with fetched features
    data_dict = {
        "amount": [transaction.amount],
        "time": [transaction.time],
        "v1": feature_vector["v1"],
        "v2": feature_vector["v2"]
    }
    
    data = pd.DataFrame(data_dict)
    dmatrix_data = xgb.DMatrix(data)
    
    prediction = model.predict(dmatrix_data)
    is_fraud = bool(prediction[0] > 0.8)
    
    return {"fraud_probability": float(prediction[0]), "is_fraud": is_fraud}
