import logging

import mlflow
import dagshub
import pandas as pd


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dagshub.init(repo_owner='ahmedsamir42003', repo_name='MLOps', mlflow=True)

model_name = "MyAwesomeModel" 
model_uri = f"models:/{model_name}/Production"


try:
    model = mlflow.pyfunc.load_model(model_uri)
    log.info("Production model loaded successfully.")
    
    sample_data = pd.read_csv("./data/test.csv")
    sample_data = sample_data.head(5)
    
    prediction = model.predict(sample_data)
    print(f"Final Prediction: {prediction}")
except Exception as e:
    print(f"Error loading model: {e}. Check if the model is set to 'Production' stage.")
