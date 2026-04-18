import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df):
    """
    Detect anomalies using Isolation Forest on numeric columns.
    Returns:
    - updated dataframe with anomaly flag
    - anomaly summary
    """
    df = df.copy()

    numeric_df = df.select_dtypes(include=["number"]).copy()

    if numeric_df.empty or numeric_df.shape[1] < 1:
        df["anomaly_flag"] = 0
        return df, {
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "message": "No numeric columns available for anomaly detection."
        }

    model = IsolationForest(
        n_estimators=50,
        contamination=0.05,
        random_state=42
    )

    preds = model.fit_predict(numeric_df)
    df["anomaly_flag"] = preds
    df["anomaly_flag"] = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

    anomaly_count = int(df["anomaly_flag"].sum())
    anomaly_percentage = float((anomaly_count / len(df)) * 100) if len(df) > 0 else 0.0

    return df, {
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(anomaly_percentage, 2),
        "message": "Anomaly detection completed successfully."
    }