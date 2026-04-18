import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from app.config import CHART_DIR


def detect_problem_type(df, target_column):
    target = df[target_column]

    if pd.api.types.is_numeric_dtype(target):
        return "regression"
    return "classification"


def prepare_features(df, target_column):
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def get_models(problem_type):
    if problem_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=20,   # 🔥 reduced from 100
                max_depth=5,       # 🔥 limit complexity
                random_state=42
            ),
        }

    return {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=20,
            max_depth=5,
            random_state=42
        ),
    }


def evaluate_model(problem_type, y_test, y_pred):
    if problem_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        return {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(rmse),
            "R2": float(r2_score(y_test, y_pred)),
        }

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }


def train_and_evaluate_models(df, target_column):
    problem_type = detect_problem_type(df, target_column)

    X, y, numeric_features, categorical_features = prepare_features(df, target_column)

    if problem_type == "classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = get_models(problem_type)

    results = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(problem_type, y_test, y_pred)

        results[model_name] = {
            "pipeline": pipeline,
            "metrics": metrics,
        }

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"]["R2"] if problem_type == "regression"
        else results[name]["metrics"]["F1"]
    )

    best_model_info = results[best_model_name]

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "all_results": results,
        "best_model_name": best_model_name,
        "best_model_pipeline": best_model_info["pipeline"],
        "best_model_metrics": best_model_info["metrics"],
        "feature_importance": None,
    }