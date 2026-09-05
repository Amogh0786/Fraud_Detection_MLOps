from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from feast import FeatureStore
import logging
import shap
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API")

model = None
shadow_model = None
fs = None
explainer = None

@app.on_event("startup")
def load_resources():
    global model, shadow_model, fs, explainer
    try:
        model_uri = "models:/FraudDetectionModel/Production"
        model = mlflow.xgboost.load_model(model_uri)
        print("Primary model loaded successfully.")
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        print("SHAP explainer loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load primary model or explainer on startup: {e}")
        
    try:
        shadow_model_uri = "models:/FraudDetectionModel/Staging"
        shadow_model = mlflow.xgboost.load_model(shadow_model_uri)
        print("Shadow model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load shadow model on startup: {e}")
    
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
    
    # Run shadow model if available
    if shadow_model is not None:
        try:
            shadow_pred = shadow_model.predict(dmatrix_data)
            logger.info(f"Shadow Model Prediction: {float(shadow_pred[0])}, Primary: {float(prediction[0])}")
        except Exception as e:
            logger.error(f"Shadow model failed: {e}")
            
    # Calculate SHAP values
    top_feature = "unknown"
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(data)
            # Find the feature with the highest absolute SHAP value
            feature_names = data.columns
            max_idx = np.argmax(np.abs(shap_values[0]))
            top_feature = feature_names[max_idx]
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
    
    return {
        "fraud_probability": float(prediction[0]), 
        "is_fraud": is_fraud,
        "top_contributing_feature": top_feature
    }
