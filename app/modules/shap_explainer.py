import shap
import matplotlib.pyplot as plt
import pandas as pd
from app.config import CHART_DIR


def generate_shap_summary(pipeline, X_sample):
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Only apply SHAP for tree models
        if not hasattr(model, "feature_importances_"):
            return "SHAP skipped (non-tree model)"

        # Transform features
        X_transformed = preprocessor.transform(X_sample)

        # Convert to DataFrame
        X_transformed = pd.DataFrame(X_transformed)

        # Use TreeExplainer (best for RandomForest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        plt.figure()

        # For classification, shap_values is list
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_transformed, show=False)
        else:
            shap.summary_plot(shap_values, X_transformed, show=False)

        file_path = CHART_DIR / "shap_summary.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        return str(file_path)

    except Exception as e:
        return f"SHAP failed: {str(e)}"