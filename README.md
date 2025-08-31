# Procurement AI Agent

## Overview
This project is an attempt to turn **procurement and contract compliance** into an AI/ML project that flags when a vendor is **not in compliance** with agreed procurement guidelines.  
While the dataset does not contain explicit compliance rules, the model leverages historical patterns to infer compliance vs. non-compliance.

---

## Project Setup
- ✅ Setup GitHub repository  
- ✅ Integrated with [DagsHub](https://dagshub.com) for experiment & artifact tracking  
- ✅ Created a project directory structure (`src/`, `data/`, `models/`, `metrics/`)  
- ✅ Downloaded dataset from Kaggle: **Procurement KPI Analysis Dataset**  
- ✅ Loaded the dataset into **MongoDB** using R scripts  
- ✅ Created a local Python environment (`venv`)  
- ✅ Developed exploratory notebooks (`notebooks/`) to:
  - Understand the data
  - Identify the target variable (`encoded_Compliance`)
  - Experiment with ML models
  - Document preprocessing decisions

---

## Data Ingestion with R

I initially loaded the dataset into MongoDB using R. Keep that workflow documented here.

- R script location: `R/datauploader.R`
- Example using `mongolite`:

```r
# R/datauploader.R
library(mongolite)
library(readr)

# Read CSV
df <- read_csv("R/Procurement KPI Analysis Dataset.csv")

# Use MONGO_URI from an .Renviron or Sys.setenv
mongo_uri <- Sys.getenv("MONGO_URI")
stopifnot(nchar(mongo_uri) > 0)

# Write to MongoDB
con <- mongo(
  collection = "Procurement-AI-Agent",
  db = "test",
  url = mongo_uri
)

con$insert(df)
cat("Inserted", nrow(df), "rows into test.Procurement-AI-Agent\n")

```

```bash
Rscript R/datauploader.R
```

## Data Glossary

| Field          | Encoded Column        | Values                                                                 |
|----------------|-----------------------|------------------------------------------------------------------------|
| **Supplier**   | `encoded_Supplier`    | Alpha_Inc=0, Beta_Supplies=1, Delta_Logistics=2, Epsilon_Group=3, Gamma_Co=4 |
| **Item_Category** | `encoded_Item_Category` | Electronics=0, MRO=1, Office Supplies=2, Packaging=3, Raw Materials=4 |
| **Order_Status**  | `encoded_Order_Status`  | Cancelled=0, Delivered=1, Partially Delivered=2, Pending=3 |
| **Compliance**    | `encoded_Compliance`    | No=0, Yes=1 |

---

## Modeling
- Split dataset into train/test sets
- Utilized **RandomForestClassifier** to predict compliance
- Final metrics (on test set):

          precision    recall  f1-score   support

       0       0.45      0.32      0.38        28
       1       0.86      0.91      0.89       128

accuracy                           0.81       156

- macro avg 0.66 0.62 0.63 156
- weighted avg 0.79 0.81 0.79 156

✅ Achieved ~**80% accuracy** with high recall on the “compliant” class.  
✅ Satisfactory performance for first iteration.

---

## Model Parameters

```python
params = {
    "n_estimators": 350,         # More trees for stability
    "criterion": "gini",         # "gini" is faster, "entropy" also possible
    "max_depth": None,           # Grow fully unless overfitting
    "min_samples_split": 5,      # Prevents overfitting
    "min_samples_leaf": 2,       # Each leaf must have ≥2 samples
    "max_features": "sqrt",      # Common for classification tasks
    "bootstrap": True,           # Bootstrapping improves generalization
    "class_weight": "balanced",  # Handles imbalanced classes
    "random_state": 42,          # Reproducibility
    "n_jobs": -1,                # Use all cores
    "verbose": 0                 # Set >0 to see training progress
}
```
---

## Pipeline

The pipeline is managed with DVC:

- preprocess.py
Loads raw MongoDB data, cleans & encodes, saves parquet + scaler.

- train.py
Trains the RandomForest model, logs metrics and artifacts to MLflow + DagsHub.

- evaluate.py
Evaluates model on the full processed dataset, generates JSON metrics & ROC/PR curves.

- dvc.yaml defines the full DAG:

    1- preprocess → data/processed/data.parquet, models/scaler.pkl

    2- train → models/model.pkl, metrics/train.json

    3- evaluate → metrics/eval.json, metrics/curves/

Run the pipeline end-to-end:
```bash
dvc repro
```

## CI/CD

- Configured GitHub Actions for CI/CD:

    - Runs pipeline automatically on pushes/PRs

    - Pushes DVC-tracked artifacts (models, metrics) to DagsHub

    - Logs experiments to MLflow

- Secrets stored in GitHub:

    - MONGO_URI

    - MLFLOW_TRACKING_URI

    - MLFLOW_TRACKING_USERNAME

    - MLFLOW_TRACKING_PASSWORD

    - DAGSHUB_USER, DAGSHUB_TOKEN, DAGSHUB_REPO

---

## References

- Dataset: Kaggle – Procurement KPI Analysis Dataset

- Tracking: DagsHub

- Frameworks: scikit-learn, DVC, MLflow, and MongoDB


## Created by

Mohamed Gad
Data Scientist • MERN/AI Engineer
***🌐 www.mohamedgad.com***