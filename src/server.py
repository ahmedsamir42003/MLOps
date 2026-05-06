import logging
from typing import List, Union

import mlflow
import dagshub
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dagshub.init(repo_owner="ahmedsamir42003", repo_name="MLOps", mlflow=True)

model_name = "MyAwesomeModel"
model_uri = f"models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Titanic Prediction Service")


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get("/")
def root():
    return {"status": "ok", "service": "Titanic Prediction Service"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data_input: Union[Passenger, List[Passenger]]):
    if isinstance(data_input, Passenger):
        records = [data_input.model_dump()]
    else:
        records = [p.model_dump() for p in data_input]

    df = pd.DataFrame(records)
    predictions = model.predict(df)

    log.info("Generated %s prediction(s).", len(predictions))
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
