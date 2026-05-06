# Titanic MLOps Training Pipeline

This repository serves as a workspace for my MLOps journey. This specific lab focuses on building a fully automated, reproducible training pipeline for the Titanic dataset using Lightning AI and Scikit-Learn.

---

## lab 1
### 🚀 Key Features
- **Automated Pipeline**: Includes data loading, preprocessing (imputation/scaling/encoding), training, and model saving in one command.
- **Configurable**: Switch between `RandomForest` and `LogisticRegression` via YAML configs without touching the code.
- **Code Quality**: Enforces standards using `Ruff`, `Black`, and `isort`.
- **Reproducibility**: Uses `uv` for dependency management and 64-bit Python 3.12 for pre-compiled binaries.

---
## lab 2
### 🚀 Key Features
- **Versioned Experiments:** Uses DVC to track datasets, model outputs, and pipeline stages, ensuring full reproducibility of machine learning experiments.
- **Multiple Model Runs:** Supports running and comparing different models (`RandomForest`, `LogisticRegression`) using Hydra configuration overrides or separate DVC pipeline stages.
- **Remote Storage:** Stores large artifacts (`datasets and trained models`) in DagsHub remote storage instead of Git, keeping the repository lightweight.
- **Reproducible Experiments:** Allows re-running experiments on any machine using dvc repro and restoring all data with dvc pull.
- **Pipeline Tracking:** Automatically tracks dependencies (`data`, `code`, `configs`) and outputs (`.joblib models`) using dvc.yaml and dvc.lock.

---

## Lab 3
### 🚀 Key Features
- **Experiment Tracking with MLflow:** Integrates MLflow via DagsHub to log hyperparameters, performance metrics `Accuracy` , and training artifacts in real-time.
- **Centralized Model Registry:** register the best-performing model  to the DagsHub Model Registry, enabling version control for trained weight.
- **Model Lifecycle Management:** Implements a professional workflow by transitioning models through stages (`Staging`, `Production`, `Archived`) directly from the UI .
- **Production-Ready Inference:** Features a decoupled prediction script that pulls the "Production" tagged model from the cloud, eliminating the need for local .joblib files during deployment.

---

## Lab 4
### 🚀 Key Features
- **Production API with FastAPI:** Wraps the Titanic prediction model in a high-performance REST API, making the model accessible over the web via HTTPS.
- **Scalable Batch Inference:** Designed the solution to be versatile; it accepts both single records and batch requests (multiple records in one JSON array), optimizing throughput for large datasets.
- **Cloud Deployment on Hugging Face:** Successfully deployed the serving solution to Hugging Face Spaces using a custom Docker environment and `uv` for lightning-fast dependency resolution.
- **External API Testing with Bruno:** Validated the live endpoint using the Bruno API client, ensuring the model handles real-world JSON inputs and returns accurate survival predictions.
- **Dynamic Model Loading:** The API does not store the model locally; it pulls the "Production" version directly from the DagsHub Model Registry at startup using secure environment secrets.
- **🔗 Live API Endpoint:** `https://ahmed-samir-abdel-fattah-titanic-prediction-api.hf.space/predict`

---

## Lab 5
### 🚀 Key Features
- **Cloud Data Warehouse with MotherDuck:** Loaded the entire test dataset into MotherDuck, a serverless DuckDB cloud database, enabling scalable and cost-effective data storage without infrastructure management.
- **Orchestrated Batch Scoring with Prefect:** Built a batch inference job using Prefect to orchestrate the complete prediction pipeline, including extraction, transformation, model loading, and results storage.
- **Secure Token Management:** Implemented secure authentication for both MotherDuck and DagsHub using `.env` files with `python-dotenv`, ensuring sensitive credentials are never hardcoded in version control.
- **Automated Data Pipeline:** The Prefect flow automatically extracts raw test data from MotherDuck, applies necessary transformations, loads the "Production" model from DagsHub Model Registry, generates survival predictions, and saves the results back to MotherDuck.
- **End-to-End Automation:** Combined cloud data warehousing, experiment tracking, model registry, and workflow orchestration into a single, reproducible batch scoring pipeline.
