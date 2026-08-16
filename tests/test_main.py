from fastapi.testclient import TestClient
from app.main import app
import pytest

# Simple mock for XGBoost model to bypass MLflow loading in tests
class MockModel:
    def predict(self, dmatrix):
        # Return a mock prediction of 0.9 (fraud)
        return [0.9]

@pytest.fixture(autouse=True)
def mock_mlflow_model(monkeypatch):
    import app.main
    app.main.model = MockModel()

client = TestClient(app)

def test_predict_fraud():
    payload = {
        "amount": 100.50,
        "time": 0.0,
        "v1": -1.3598071
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert data["fraud_probability"] == 0.9
    assert data["is_fraud"] is True
