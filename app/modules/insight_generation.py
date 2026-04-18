import pandas as pd


def generate_data_quality_insights(cleaning_report):
    insights = []

    missing_before = cleaning_report["missing_values"]["missing_before"]
    missing_after = cleaning_report["missing_values"]["missing_after"]
    duplicates_removed = cleaning_report["duplicates"]["duplicates_removed"]

    insights.append(
        f"Missing values were reduced from {missing_before} to {missing_after} during automated cleaning."
    )
    insights.append(
        f"A total of {duplicates_removed} duplicate rows were removed from the dataset."
    )

    dtype_changes = cleaning_report["data_types"]["changed_columns"]
    if dtype_changes:
        insights.append(
            f"{len(dtype_changes)} columns had their data types corrected automatically."
        )
    else:
        insights.append("No major data type corrections were required.")

    return insights


def generate_business_insights(df):
    insights = []

    if "Sales" in df.columns:
        total_sales = df["Sales"].sum()
        avg_sales = df["Sales"].mean()
        insights.append(f"Total sales in the dataset are {total_sales:.2f}.")
        insights.append(f"Average sales per record are {avg_sales:.2f}.")

    if "Profit" in df.columns:
        total_profit = df["Profit"].sum()
        avg_profit = df["Profit"].mean()
        insights.append(f"Total profit in the dataset is {total_profit:.2f}.")
        insights.append(f"Average profit per record is {avg_profit:.2f}.")

    if "Category" in df.columns and "Sales" in df.columns:
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        top_category_sales = df.groupby("Category")["Sales"].sum().max()
        insights.append(
            f"The top-performing category by sales is {top_category} with total sales of {top_category_sales:.2f}."
        )

    if "Region" in df.columns and "Sales" in df.columns:
        top_region = df.groupby("Region")["Sales"].sum().idxmax()
        top_region_sales = df.groupby("Region")["Sales"].sum().max()
        insights.append(
            f"The highest sales region is {top_region} with total sales of {top_region_sales:.2f}."
        )

    if "Discount" in df.columns and "Profit" in df.columns:
        correlation = df["Discount"].corr(df["Profit"])
        if pd.notna(correlation):
            insights.append(
                f"The correlation between discount and profit is {correlation:.2f}, which helps explain how discounting affects profitability."
            )

    return insights


def generate_model_insights(ml_results):
    insights = []

    insights.append(
        f"The problem type was identified as {ml_results['problem_type']}."
    )
    insights.append(
        f"The best performing model is {ml_results['best_model_name']}."
    )

    metrics = ml_results["best_model_metrics"]
    for metric_name, metric_value in metrics.items():
        insights.append(f"{metric_name} of the best model is {metric_value:.4f}.")

    feature_importance = ml_results.get("feature_importance")
    if feature_importance is not None and not feature_importance.empty:
        top_feature = feature_importance.iloc[0]["feature"]
        top_importance = feature_importance.iloc[0]["importance"]
        insights.append(
            f"The most influential feature is {top_feature} with importance score {top_importance:.4f}."
        )

    return insights


def generate_final_summary(cleaning_report, df, ml_results):
    all_insights = []

    all_insights.extend(generate_data_quality_insights(cleaning_report))
    all_insights.extend(generate_business_insights(df))
    all_insights.extend(generate_model_insights(ml_results))

    return "\n".join(f"- {insight}" for insight in all_insights)