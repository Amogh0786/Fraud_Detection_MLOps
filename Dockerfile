FROM python:3.9-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
# Also copy MLflow artifacts/DB if running locally, or configure env vars for remote MLflow
COPY mlflow.db /app/mlflow.db 
COPY mlruns /app/mlruns

ENV MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
