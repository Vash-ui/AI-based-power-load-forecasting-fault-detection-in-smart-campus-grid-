# src/api/fastapi_app.py
from fastapi import FastAPI
from joblib import load

app = FastAPI()
model = load('models/lstm_model.pkl')

@app.post("/predict")
async def predict(payload: dict):
    input_data = preprocess(payload)
    prediction = model.predict(input_data)
    return {"load_forecast": prediction.tolist()}