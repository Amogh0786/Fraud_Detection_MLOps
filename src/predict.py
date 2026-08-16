import mlflow.xgboost
import xgboost as xgb
import pandas as pd

def predict_standalone(model_uri: str, data: pd.DataFrame):
    """
    Utility function for standalone inference, independent of FastAPI.
    """
    model = mlflow.xgboost.load_model(model_uri)
    dmatrix_data = xgb.DMatrix(data)
    predictions = model.predict(dmatrix_data)
    return predictions
