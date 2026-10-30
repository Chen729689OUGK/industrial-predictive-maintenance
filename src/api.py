
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.json"


# Load trained model
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

THRESHOLD = float(threshold_data["threshold"])


# Create API
app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="API for predicting industrial machine failures.",
    version="1.0.0"
)


# Input data schema
class MachineData(BaseModel):
    Type: str = Field(description="Machine type: L, M or H")
    air_temperature: float = Field(alias="Air temperature [K]")
    process_temperature: float = Field(alias="Process temperature [K]")
    rotational_speed: float = Field(alias="Rotational speed [rpm]")
    torque: float = Field(alias="Torque [Nm]")
    tool_wear: float = Field(alias="Tool wear [min]")

    model_config = {"populate_by_name": True}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict_failure(machine: MachineData):

    input_data = pd.DataFrame([{
        "Type": machine.Type,
        "Air temperature [K]": machine.air_temperature,
        "Process temperature [K]": machine.process_temperature,
        "Rotational speed [rpm]": machine.rotational_speed,
        "Torque [Nm]": machine.torque,
        "Tool wear [min]": machine.tool_wear
    }])

    probability = float(model.predict_proba(input_data)[0][1])

    prediction = int(probability >= THRESHOLD)

    return {
        "failure_probability": round(probability, 6),
        "threshold": round(THRESHOLD, 4),
        "prediction": prediction,
        "risk": "High risk" if prediction == 1 else "Low risk",
        "message": (
            "Potential machine failure detected. Inspection recommended."
            if prediction == 1
            else "No failure detected at the selected threshold."
        )
    }
