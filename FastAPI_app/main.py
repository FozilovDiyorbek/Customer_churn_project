from fastapi import FastAPI
from FastAPI_app.schemes import CustomerData
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("models/best_model.pkl")

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "probability_of_churn": round(float(proba), 3)
    }
