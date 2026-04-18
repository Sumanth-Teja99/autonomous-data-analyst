import pandas as pd
from pathlib import Path


def load_data(file_path):
    """
    Load CSV or Excel file into DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix == ".csv":
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
    elif file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    return df


def get_basic_info(df):
    """
    Get basic dataset information
    """
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict()
    }

    return info


def validate_dataset(df):
    """
    Basic validation checks
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns.")

    return True