# Fraud Detection MLOps

An end-to-end MLOps pipeline for credit card fraud detection using MLflow for versioning, Docker / FastAPI for serving, GitHub Actions for CI/CD, and Evidently AI for monitoring data drift.

## Overview
This repository contains a modular structure for automatically training, tracking, serving, and monitoring a fraud detection model using XGBoost.

## Key Components
- **Model Training**: Uses XGBoost and tracks metrics, parameters, and model artifacts with MLflow.
- **Serving**: FastAPI application containerized with Docker, pulling the latest model from the MLflow registry dynamically.
- **Monitoring**: Uses Evidently AI to compare reference distributions with incoming production data to detect data drift.
- **CI/CD**: Fully automated pipeline with GitHub Actions to test, build, and deploy, alongside a specific action triggered by data drift detection to retrain the model.

## Quickstart
1. Set up the local Python environment.
2. Provide dummy data in the `data/` folder.
3. Run `python src/train.py` to train and register the model.
4. Start the server with `uvicorn app.main:app --reload`.
