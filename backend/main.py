from fastapi import FastAPI
import joblib
import os
import numpy as np
from sklearn.datasets import load_iris


iris = load_iris()
app = FastAPI()
model_path = os.path.join(os.getcwd(), "models", "model.joblib")
model = joblib.load(model_path)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}


@app.post("/predict/")
async def predict_species(data: dict):
    # Implement your prediction logic here using the loaded model
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    class_name = iris.target_names[prediction][0]
    return {"class": class_name}
