from pathlib import Path
import pandas as pd

from app.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    CHART_DIR,
    REPORT_DIR,
    MODEL_DIR,
    SHAP_DIR,
    EXPORT_DIR,
)


def ensure_directories():
    """Create all required project directories if they do not exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        CHART_DIR,
        REPORT_DIR,
        MODEL_DIR,
        SHAP_DIR,
        EXPORT_DIR,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_text_report(file_path, content):
    """Save plain text content to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def save_dataframe(df, file_path):
    """Save a DataFrame to CSV."""
    df.to_csv(file_path, index=False)


def detect_column_types(df):
    """Return numeric, categorical, and datetime column lists."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def try_parse_dates(df):
    """
    Try converting object columns to datetime where possible.
    Only converts a column if at least 70% of non-null values parse successfully.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_datetime(df[col], errors="coerce")
            non_null_count = df[col].notna().sum()

            if non_null_count > 0:
                parsed_ratio = converted.notna().sum() / non_null_count
                if parsed_ratio >= 0.7:
                    df[col] = converted

    return df