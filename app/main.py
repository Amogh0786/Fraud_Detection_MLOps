from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd

app = FastAPI(title="Fraud Detection API")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = "models:/FraudDetectionModel/latest"
        model = mlflow.xgboost.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        # Not raising here so the API can still start for health checks, 
        # though predictions will fail until model is available.

class Transaction(BaseModel):
    amount: float
    time: float
    v1: float
    # ... other anonymized features

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Ensure MLflow tracking server is accessible and model exists.")
        
    data = pd.DataFrame([transaction.dict()])
    # In XGBoost, convert to DMatrix
    import xgboost as xgb
    dmatrix_data = xgb.DMatrix(data)
    
    prediction = model.predict(dmatrix_data)
    is_fraud = bool(prediction[0] > 0.8) # Confidence threshold
    
    # In production, log this request to a database for drift monitoring
    
    return {"fraud_probability": float(prediction[0]), "is_fraud": is_fraud}
