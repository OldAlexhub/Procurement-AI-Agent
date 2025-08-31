# src/evaluate.py
import os
import json
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve
)
import joblib
import mlflow
import matplotlib.pyplot as plt

# Load .env
load_dotenv()

# Load params from params.yaml (evaluate section)
root = os.path.dirname(os.path.dirname(__file__))
params_path = os.path.join(root, "params.yaml")
with open(params_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)["evaluate"]

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _save_curve(x, y, xlabel, ylabel, title, out_path: Path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _ensure_parent(out_path)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    # MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mluser = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlpass = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mluser:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mluser
    if mlpass:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlpass
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment("procurement_compliance_rf")
    except Exception:
        pass

    # Data source
    mongo_uri = os.getenv(config["mongo_uri_env"])
    if not mongo_uri:
        raise ValueError(f"Env var {config['mongo_uri_env']} is not set")

    client = MongoClient(mongo_uri)
    db = client[config["db"]]
    collection = db[config["collection"]]
    data = pd.DataFrame(list(collection.find()))
    if "_id" in data.columns:
        data = data.drop(columns="_id")

    if "encoded_Compliance" not in data.columns:
        raise KeyError("Target column 'encoded_Compliance' not found in data")

    X = data.drop(columns="encoded_Compliance").select_dtypes(include=["number"])
    y = data["encoded_Compliance"]

    # Load model
    model_path = Path(root) / config["model_in"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X)
    y_prob = None
    if hasattr(model, "predict_proba"):
        # For multiclass, keep the positive class as class 1 if binary
        probs = model.predict_proba(X)
        y_prob = probs[:, 1] if probs.shape[1] == 2 else None

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
    }

    # Curves and AUCs if binary with probabilities
    curves_dir = Path(root) / config.get("curves_dir", "metrics/curves")
    roc_path = curves_dir / "roc_curve.png"
    pr_path = curves_dir / "pr_curve.png"
    if y_prob is not None and len(y.unique()) == 2:
        metrics["roc_auc"] = roc_auc_score(y, y_prob)
        metrics["pr_auc"] = average_precision_score(y, y_prob)

        fpr, tpr, _ = roc_curve(y, y_prob)
        _save_curve(fpr, tpr, "False Positive Rate", "True Positive Rate", "ROC Curve", roc_path)

        precision, recall, _ = precision_recall_curve(y, y_prob)
        _save_curve(recall, precision, "Recall", "Precision", "PR Curve", pr_path)

    # Confusion matrix and report
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, zero_division=0, output_dict=True)

    # Save metrics JSON
    metrics_out = Path(root) / config["metrics_out"]
    _ensure_parent(metrics_out)
    payload = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Log to MLflow
    with mlflow.start_run(run_name="rf_eval"):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(metrics_out))
        if (Path.exists(roc_path)):
            mlflow.log_artifact(str(roc_path), artifact_path="curves")
        if (Path.exists(pr_path)):
            mlflow.log_artifact(str(pr_path), artifact_path="curves")

    print(f"[info] evaluated on {X.shape[0]} rows, {X.shape[1]} features")
    print(f"[info] metrics saved to {metrics_out}")
    if (Path.exists(roc_path)) and (Path.exists(pr_path)):
        print(f"[info] curves saved to {curves_dir}")

if __name__ == "__main__":
    main()
