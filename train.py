import logging

import dagshub
import mlflow

import hydra
import joblib
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from data_utils import load_data
from processing import get_preprocessing_pipeline

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # 1. Load Data (Convert relative path to absolute so Hydra finds it)
    data_path = to_absolute_path(cfg.data.train_path)
    log.info(f"Loading data from {data_path}")

    X_train, X_test, y_train, y_test = load_data(
        data_path, cfg.data.test_size, cfg.data.random_state
    )

    log.info(f"Instantiating model: {cfg.model.name}")
    if cfg.model.name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(**cfg.model.params)
    elif cfg.model.name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(**cfg.model.params)
    else:
        raise ValueError(f"Model {cfg.model.name} is not supported")

    preprocessor = get_preprocessing_pipeline()
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    
    dagshub.init(repo_owner='ahmedsamir42003', repo_name='MLOps', mlflow=True)
    
    with mlflow.start_run():
        
        log.info("Training pipeline...")
        clf.fit(X_train, y_train)

        log.info("Evaluating...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        log.info(f"Model Accuracy: {acc:.4f}")

        joblib.dump(clf, cfg.model_save_path)
        log.info(f"Model saved locally to {cfg.model_save_path}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")


if __name__ == "__main__":
    main()
