def add_engineered_features(df):
    df["stress_index"] = (
        df["temperature_c"] *
        df["vibration_mm_s"] *
        (df["load_pct"] / 100)
    )

    df["usage_intensity"] = df["operating_hours"] / df["age_years"]
    df["maintenance_gap_ratio"] = df["last_maintenance_hours"] / (df["operating_hours"] + 1)

    return df
