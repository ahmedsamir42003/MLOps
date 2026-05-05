import mlflow
import dagshub
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


dagshub.init(repo_owner='ahmedsamir42003', repo_name='MLOps', mlflow=True)

model_name = "MyAwesomeModel"
model_uri = f"models:/{model_name}/production"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Titanic Prediction Service")


class Passenger(BaseModel):
    Pclass: int
    Sex: int  
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

from typing import List, Union

@app.post("/predict")
def predict(data_input: Union[Passenger, List[Passenger]]):

    if isinstance(data_input, Passenger):
        records = [data_input.model_dump()]
    else:
        records = [p.model_dump() for p in data_input]
    
    df = pd.DataFrame(records)
    
    predictions = model.predict(df)
    
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)