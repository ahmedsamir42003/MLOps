import dagshub
import mlflow

dagshub.init(repo_owner="ahmedsamir42003", repo_name="MLOps", mlflow=True)

run_id = "7790720802114c74b090014b7ceb28c2"
model_id = "m-5b237af8eaa54a1ba785d5cffabe2095"
model_uri = f"models:/{model_id}"
mlflow.register_model(model_uri, "MyAwesomeModel")
print("done")
