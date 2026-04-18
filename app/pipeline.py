from pathlib import Path

from app.config import (
    CLEANED_DATA_FILE,
    MODEL_REPORT_FILE,
    INSIGHT_REPORT_FILE,
    CHART_DIR,
    REPORT_DIR,
)
from app.modules.data_ingestion import load_data, validate_dataset, get_basic_info
from app.modules.data_cleaning import clean_data
from app.modules.eda import (
    generate_summary,
    plot_distributions,
    plot_correlation_heatmap,
    plot_category_analysis,
    plot_trend_analysis,
)
from app.modules.modeling import train_and_evaluate_models
from app.modules.insight_generation import generate_final_summary
from app.modules.anomaly_detection import detect_anomalies
from app.modules.shap_explainer import generate_shap_summary
from app.utils import ensure_directories, save_dataframe, save_text_report


def clear_old_outputs():
    """
    Delete old chart and report files so each new dataset gets fresh outputs.
    """
    for folder in [CHART_DIR, REPORT_DIR]:
        folder_path = Path(folder)
        if folder_path.exists():
            for file in folder_path.iterdir():
                if file.is_file():
                    file.unlink()


def run_pipeline(file_path, target_column=None):
    ensure_directories()
    clear_old_outputs()

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading file: {file_path.name}")

    df = load_data(file_path)
    validate_dataset(df)
    basic_info = get_basic_info(df)

    cleaned_df, cleaning_report = clean_data(df)

    print("Running anomaly detection...")
    cleaned_df, anomaly_report = detect_anomalies(cleaned_df)
    print("Anomaly detection completed.")

    print("Running EDA...")
    summary = generate_summary(cleaned_df)
    plot_distributions(cleaned_df)
    plot_correlation_heatmap(cleaned_df)
    plot_category_analysis(cleaned_df)
    plot_trend_analysis(cleaned_df)
    print("EDA completed.")

    save_dataframe(cleaned_df, CLEANED_DATA_FILE)

    cleaning_report_text = f"""
DATASET BASIC INFO
------------------
Rows: {basic_info['rows']}
Columns: {basic_info['columns']}
Column Names: {basic_info['column_names']}

MISSING VALUES
--------------
Before: {cleaning_report['missing_values']['missing_before']}
After: {cleaning_report['missing_values']['missing_after']}

DUPLICATES
----------
Before: {cleaning_report['duplicates']['duplicates_before']}
After: {cleaning_report['duplicates']['duplicates_after']}
Removed: {cleaning_report['duplicates']['duplicates_removed']}

DATA TYPE CHANGES
-----------------
{cleaning_report['data_types']['changed_columns']}

OUTLIER SUMMARY
---------------
{cleaning_report['outliers']}

ANOMALY SUMMARY
---------------
Count: {anomaly_report['anomaly_count']}
Percentage: {anomaly_report['anomaly_percentage']}%
Message: {anomaly_report['message']}

EDA SUMMARY
-----------
Shape: {summary['shape']}
Columns: {summary['columns']}
"""

    save_text_report(REPORT_DIR / "cleaning_report.txt", cleaning_report_text)

    if target_column and target_column in cleaned_df.columns:
        print(f"Running Machine Learning using target column: {target_column}")

        ml_results = train_and_evaluate_models(cleaned_df, target_column)

        shap_path = generate_shap_summary(
            ml_results["best_model_pipeline"],
            cleaned_df.drop(columns=[target_column]).head(100)
        )

        model_report = f"""
MACHINE LEARNING REPORT
-----------------------
Problem Type: {ml_results['problem_type']}
Target Column: {ml_results['target_column']}
Best Model: {ml_results['best_model_name']}

Best Model Metrics
------------------
{ml_results['best_model_metrics']}

All Model Results
-----------------
"""

        for model_name, result in ml_results["all_results"].items():
            model_report += f"\n{model_name}: {result['metrics']}\n"

        model_report += f"\nSHAP Plot:\n{shap_path}\n"

        save_text_report(MODEL_REPORT_FILE, model_report)

        insight_text = generate_final_summary(cleaning_report, cleaned_df, ml_results)
        save_text_report(INSIGHT_REPORT_FILE, insight_text)

    else:
        print("No valid target column selected. ML step skipped.")

    print("Pipeline completed successfully.")
    print(f"Cleaned data saved at: {CLEANED_DATA_FILE}")

    return cleaned_df