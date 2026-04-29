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
