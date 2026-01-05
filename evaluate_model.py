import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

model = joblib.load("models/rul_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def evaluate(X, y):
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    return rmse
