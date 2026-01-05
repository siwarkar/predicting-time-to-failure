import joblib
import pandas as pd

model = joblib.load("models/rul_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_rul(input_df):
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]
