from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore

app = FastAPI(title="Bati Bank Credit Risk API")

class PredictionRequest(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_category: str

@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Dummy model for now - replace with real MLflow model later
    probability = 0.25  # placeholder
    return {
        "risk_probability": probability,
        "risk_category": "Low Risk" if probability < 0.5 else "High Risk"
    }