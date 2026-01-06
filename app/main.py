
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

app = FastAPI(title="Predictive Maintenance API")

model = mlflow.pyfunc.load_model(
    "models:/PredictiveMaintenanceRUL"
)
class MachineInput(BaseModel):
    machine_type: str
    age_years: int
    operating_hours: int
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float
    humidity_pct: float
    load_pct: float
    maintenance_count: int
    last_maintenance_hours: int
    failure_history: int

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: MachineInput):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df, columns=["machine_type"], drop_first=True)
    df = df.reindex(columns=model.metadata.get_input_schema().input_names(), fill_value=0)
    pred = model.predict(df)[0]
    return {"time_to_failure_hours": float(pred)}
