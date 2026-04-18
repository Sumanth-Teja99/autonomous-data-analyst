import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from app.config import CHART_DIR


def generate_summary(df):
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe().to_string()
    }
    return summary


def plot_distributions(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(CHART_DIR / f"{col}_distribution.png")
        plt.close()


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(CHART_DIR / "correlation_heatmap.png")
    plt.close()


def plot_category_analysis(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        df[col].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top Categories in {col}")
        plt.savefig(CHART_DIR / f"{col}_categories.png")
        plt.close()


def plot_trend_analysis(df):
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df["YearMonth"] = df["Order Date"].dt.to_period("M")

        trend = df.groupby("YearMonth")["Sales"].sum()

        trend.plot(figsize=(10, 5))
        plt.title("Monthly Sales Trend")
        plt.xticks(rotation=45)
        plt.savefig(CHART_DIR / "sales_trend.png")
        plt.close()