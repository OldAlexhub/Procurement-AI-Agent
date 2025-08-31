# src/train.py
import os
import json
from pathlib import Path
import random

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
import joblib
os.environ["MLFLOW_ENABLE_LOGGED_MODELS"] = "false"
import mlflow
from mlflow.sklearn import log_model

# Load .env
load_dotenv()

# Load params from params.yaml (train section)
root = os.path.dirname(os.path.dirname(__file__))
params_path = os.path.join(root, "params.yaml")
with open(params_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)["train"]

def main():
    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mluser = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlpass = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mluser:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mluser
    if mlpass:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlpass
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Experiment
    try:
        mlflow.set_experiment("procurement_compliance_rf")
    except Exception:
        pass

    # Load training data from Mongo
    mongo_uri = os.getenv(config["mongo_uri_env"])
    if not mongo_uri:
        raise ValueError(f"Env var {config['mongo_uri_env']} is not set")

    client = MongoClient(mongo_uri)
    db = client[config["db"]]
    collection = db[config["collection"]]
    data = pd.DataFrame(list(collection.find()))
    if "_id" in data.columns:
        data = data.drop(columns="_id")

    # Features and target
    if "encoded_Compliance" not in data.columns:
        raise KeyError("Target column 'encoded_Compliance' not found in data")

    X = data.drop(columns="encoded_Compliance")
    y = data["encoded_Compliance"]

    # Basic sanity to keep only numeric features
    X = X.select_dtypes(include=["number"])

    random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    params = config.get("model_params", {})
    model = RandomForestClassifier(**params)

    # Output paths
    model_out = Path(root) / config["model_out"]
    metrics_out = Path(root) / config["metrics_out"]
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # Train and evaluate with MLflow logging
    with mlflow.start_run(run_name="rf_train"):
        # Log params
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            **{f"rf_{k}": v for k, v in params.items()}
        })

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities for AUCs if classifier supports it
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # Add AUCs if possible
        if y_prob is not None and len(y.unique()) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            metrics["pr_auc"] = average_precision_score(y_test, y_prob)

        # Confusion matrix and report
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Save metrics JSON
        metrics_payload = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "features": list(X.columns)
        }
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        # Save model locally and log to MLflow
        joblib.dump(model, model_out)
        # Also log the metrics file as an artifact
        mlflow.log_artifact(str(model_out), artifact_path="model")

        print(f"[info] model saved to {model_out}")
        print(f"[info] metrics saved to {metrics_out}")

if __name__ == "__main__":
    main()
