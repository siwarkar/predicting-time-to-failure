
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_PATH = "D:\\visual_project_1\\predicting-time-to-failure\\data\\raw\\industrial_machine_time_to_failure_176500.csv"
EXPERIMENT_NAME = "Predictive_Maintenance_RUL"
MODEL_NAME = "PredictiveMaintenanceRUL"

client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp and exp.lifecycle_stage == "deleted":
    client.restore_experiment(exp.experiment_id)

mlflow.set_experiment(EXPERIMENT_NAME)

def train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    df = df.drop(columns=["machine_id"])

    X = df.drop("time_to_failure_hours", axis=1)
    y = df["time_to_failure_hours"]

    X = pd.get_dummies(X, columns=["machine_type"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = HistGradientBoostingRegressor(max_depth=10,learning_rate=0.05,max_iter=200,random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_metric("MAE", mae)
        mlflow.sklearn.log_model(
            model,
            artifact_path="rul_model",
            registered_model_name=MODEL_NAME
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/rul_model.pkl")

        print("Model trained and registered | MAE:", mae)

if __name__ == "__main__":
    train()

