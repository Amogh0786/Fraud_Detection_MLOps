import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import average_precision_score
import os

def train_model(X_train, y_train, X_test, y_test):
    # Set tracking server URI (could be a remote server like AWS EC2)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Fraud_Detection_Experiment")
    
    with mlflow.start_run():
        params = {
            "objective": "binary:logistic",
            "max_depth": 5,
            "scale_pos_weight": 100, # Handle heavy fraud imbalance
            "learning_rate": 0.1
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(params, dtrain, evals=[(dtest, "test")])
        
        # Evaluate
        preds = model.predict(dtest)
        auc_pr = average_precision_score(y_test, preds)
        
        # Log metrics
        mlflow.log_metric("auc_pr", auc_pr)
        
        # Log model and register it
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="fraud_model",
            registered_model_name="FraudDetectionModel"
        )
