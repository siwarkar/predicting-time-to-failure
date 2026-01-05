import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    return df.dropna()


def split_features_target(df):
    X = df.drop("time_to_failure_hours", axis=1)
    y = df["time_to_failure_hours"]
    return X, y


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, scaler
