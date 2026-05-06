import os
import duckdb
import mlflow
import dagshub
import pandas as pd
from prefect import flow, task
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()

@task
def extract_data_from_motherduck():
    
    token = os.getenv("MOTHERDUCK_TOKEN")

    con = duckdb.connect(f"md:?motherduck_token={token}")

    df = con.sql("SELECT * FROM my_db.main.test").df()
    
    return df

@task
def get_model_and_predict(df):
    
    dagshub.init(repo_owner='ahmedsamir42003', repo_name='MLOps', mlflow=True)

    model_name = "MyAwesomeModel" 
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    predictions = model.predict(df)
    df['Survived'] = predictions
    return df[['PassengerId', 'Survived']]

@task
def load_predictions_to_motherduck(df):
    
    token = os.getenv("MOTHERDUCK_TOKEN")
    
    con = duckdb.connect(f"md:?motherduck_token={token}")

    con.sql("CREATE OR REPLACE TABLE my_db.main.titanic_predictions AS SELECT * FROM df")
    print("✅ Batch Predictions saved to MotherDuck!")

@flow(name="Titanic Batch Scoring")
def titanic_pipeline():
    raw_data = extract_data_from_motherduck()
    results = get_model_and_predict(raw_data)
    load_predictions_to_motherduck(results)

if __name__ == "__main__":
    titanic_pipeline()