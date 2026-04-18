import pandas as pd
import numpy as np

from app.utils import try_parse_dates


def handle_missing_values(df):
    """
    Fill missing values:
    - numeric columns -> median
    - categorical columns -> mode
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    missing_before = df.isnull().sum().sum()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna("Unknown")

    missing_after = df.isnull().sum().sum()

    return df, {
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
    }


def remove_duplicates(df):
    """
    Remove duplicate rows
    """
    df = df.copy()

    duplicates_before = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_after = int(df.duplicated().sum())

    return df, {
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "duplicates_removed": duplicates_before - duplicates_after,
    }


def fix_data_types(df):
    """
    Try converting object columns to datetime where possible
    """
    df = df.copy()

    types_before = df.dtypes.astype(str).to_dict()
    df = try_parse_dates(df)
    types_after = df.dtypes.astype(str).to_dict()

    changed_columns = []
    for col in df.columns:
        if types_before[col] != types_after[col]:
            changed_columns.append(
                {
                    "column": col,
                    "from": types_before[col],
                    "to": types_after[col],
                }
            )

    return df, {
        "changed_columns": changed_columns
    }


def detect_outliers_iqr(df):
    """
    Detect outliers using IQR method for numeric columns
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    outlier_summary = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        outlier_summary[col] = {
            "outlier_count": int(outliers.shape[0]),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return outlier_summary


def clean_data(df):
    """
    Full cleaning pipeline
    """
    report = {}

    df, missing_report = handle_missing_values(df)
    report["missing_values"] = missing_report

    df, duplicate_report = remove_duplicates(df)
    report["duplicates"] = duplicate_report

    df, dtype_report = fix_data_types(df)
    report["data_types"] = dtype_report

    report["outliers"] = detect_outliers_iqr(df)

    return df, report