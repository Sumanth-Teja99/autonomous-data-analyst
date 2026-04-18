from pathlib import Path

# Base directory of project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Output folders
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"
SHAP_DIR = OUTPUT_DIR / "shap"
EXPORT_DIR = OUTPUT_DIR / "exports"

# Default file names
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
EDA_REPORT_FILE = REPORT_DIR / "eda_report.txt"
MODEL_REPORT_FILE = REPORT_DIR / "model_report.txt"
INSIGHT_REPORT_FILE = REPORT_DIR / "insights.txt"