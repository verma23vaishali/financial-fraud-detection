# Financial Transaction Fraud Detection & Risk Analytics

> An end-to-end machine learning pipeline for detecting fraudulent financial transactions using a **5,493-record** working dataset. The project covers data validation, feature engineering, imbalanced classification, Random Forest modeling, fraud-risk scoring, evaluation, SQL analytics, Jupyter workflow, unit testing, and GitHub Actions CI.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Installation & Setup](#installation--setup)
5. [How to Run](#how-to-run)
6. [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
7. [Machine Learning Model](#machine-learning-model)
8. [Model Results](#model-results)
9. [Fraud Risk Scoring](#fraud-risk-scoring)
10. [Generated Outputs](#generated-outputs)
11. [SQL Analytics](#sql-analytics)
12. [Jupyter Notebook](#jupyter-notebook)
13. [Testing & CI](#testing--ci)
14. [Key Findings](#key-findings)
15. [Limitations](#limitations)
16. [Tech Stack](#tech-stack)
17. [Reproducibility](#reproducibility)

---

## Project Overview

This project demonstrates a practical **Financial Transaction Fraud Detection & Risk Analytics** workflow using Python and Scikit-learn.

**Objective:** classify transactions as legitimate (`Class = 0`) or fraudulent (`Class = 1`) and generate a probability-based risk score that can be used to prioritize transactions for fraud review.

The pipeline includes:

- Data validation and cleaning
- Exploratory analysis
- Feature engineering
- Stratified train/test splitting
- Class-weighted Random Forest classification
- ROC-AUC and PR-AUC evaluation
- Precision, Recall and F1 evaluation
- Confusion-matrix analysis
- Batch fraud probability scoring
- LOW / MEDIUM / HIGH risk classification
- SQL-based fraud analytics
- Jupyter notebook workflow
- Pytest unit tests
- GitHub Actions CI

---

## Dataset

The attached project uses a **5,493-record working dataset** derived from the supplied credit-card transaction sample.

> **Dataset note:** the original `data/sample/creditcard_sample.csv` supplied with the project contained 5,492 rows (5,000 legitimate + 492 fraud). To meet the requested 5,493-record working dataset, one deterministic legitimate demonstration transaction was added to `data/raw/creditcard.csv`. The addition is documented here for transparency.

| Attribute | Value |
|---|---:|
| Working records | **5,493** |
| Features | **30 input features** |
| Total columns | **31** |
| Fraud transactions | **492** |
| Legitimate transactions | **5,001** |
| Fraud rate | **8.96%** |
| Target | `Class` |
| Target values | `0 = Legitimate`, `1 = Fraud` |
| Train/test split | 80% / 20% |
| Random seed | 42 |

### Dataset Columns

| Column | Description |
|---|---|
| `Time` | Seconds elapsed between transactions |
| `V1`–`V28` | Anonymized transaction features |
| `Amount` | Transaction amount |
| `Class` | Target label: 0 = legitimate, 1 = fraud |

The raw dataset is located at:

```text
data/raw/creditcard.csv
```

A sample dataset is also retained at:

```text
data/sample/creditcard_sample.csv
```

---

## Repository Structure

```text
financial-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   ├── sample/
│   │   └── creditcard_sample.csv
│   └── processed/
│       └── fraud_predictions.csv
│
├── models/
│   ├── fraud_random_forest.joblib
│   └── README.md
│
├── notebooks/
│   └── 01_fraud_detection.ipynb
│
├── reports/
│   ├── metrics_runtime.json
│   ├── evaluation.json
│   └── fraud_detection_results.xlsx
│
├── sql/
│   └── fraud_analysis.sql
│
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── tests/
│   └── test_features.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip
- Git

### Install dependencies

```bash
pip install -r requirements.txt
```

Main packages include:

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical processing |
| `scikit-learn` | Machine learning and evaluation |
| `joblib` | Model serialization |
| `matplotlib` / `seaborn` | Visualization |
| `jupyter` | Notebook workflow |
| `pytest` | Unit testing |
| `duckdb` | SQL analytics |

---

## How to Run

### Step 1 — Train the model

From the project root:

```bash
python src/train_model.py
```

This trains a class-weighted Random Forest and creates:

```text
models/fraud_random_forest.joblib
reports/metrics_runtime.json
```

### Step 2 — Generate fraud predictions

```bash
python src/predict.py --input data/raw/creditcard.csv --output data/processed/fraud_predictions.csv
```

The output contains:

- Original transaction fields
- `fraud_probability`
- `risk_level`
- `predicted_fraud`

### Step 3 — Evaluate the model

```bash
python src/evaluate_model.py
```

This creates:

```text
reports/evaluation.json
```

### Step 4 — Run tests

```bash
python -m pytest -q
```

### Step 5 — Open the Excel results sheet

Open:

```text
reports/fraud_detection_results.xlsx
```

The workbook contains:

- `Project Summary`
- `Risk Summary`
- `Model Metrics`
- `Predictions`
- `Confusion Matrix`

---

## Data Preparation & Feature Engineering

The preprocessing module validates the required columns:

```text
Time
Amount
Class
```

Missing values are removed before modeling.

Two additional model features are generated:

### 1. Hour

`Time` is converted from seconds into a 24-hour transaction-time feature:

```python
Hour = (Time / 3600) % 24
```

### 2. AmountLog

Transaction amount is transformed using:

```python
AmountLog = log1p(Amount)
```

This reduces the impact of highly skewed transaction amounts.

The final model input therefore contains the original transaction features plus:

```text
Hour
AmountLog
```

---

## Machine Learning Model

### Random Forest Classifier

The project uses a **class-weighted Random Forest** because fraud is a minority class and the model needs to place greater emphasis on fraudulent transactions.

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

Random Forest is useful for this project because it:

- Captures nonlinear relationships
- Captures feature interactions
- Works well with mixed feature distributions
- Provides probability estimates
- Supports class weighting
- Is relatively easy to explain and deploy
- Provides a strong portfolio-level baseline for fraud detection

---

## Model Results

The model was trained using an **80/20 stratified train/test split** with `random_state=42`.

### Test-set performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9848** |
| PR-AUC | **0.9457** |
| Precision | **0.9655** |
| Recall | **0.8571** |
| F1 Score | **0.9081** |

### Test-set confusion matrix

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| Legitimate | 998 | 3 |
| Fraud | 14 | 84 |

The model therefore identified **84 of 98 fraud transactions** in the held-out test set at the default 0.50 probability threshold.

### Full-dataset evaluation

For reference, the saved model was also scored against the complete 5,493-record working dataset:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9978** |
| PR-AUC | **0.9917** |
| Accuracy | **0.9964** |
| Fraud Precision | **0.9896** |
| Fraud Recall | **0.9695** |
| Fraud F1 | **0.9795** |

> Full-dataset metrics are not a substitute for held-out test performance. The test-set metrics are the appropriate figures for assessing generalization.

---

## Fraud Risk Scoring

Each transaction receives a fraud probability between 0 and 1.

The project converts the probability into three demonstration risk bands:

| Probability | Risk Level |
|---:|---|
| `0.00 – <0.20` | LOW |
| `0.20 – <0.70` | MEDIUM |
| `0.70 – 1.00` | HIGH |

The final prediction uses:

```text
fraud_probability >= 0.50
```

to classify a transaction as fraud.

These thresholds are **demonstration thresholds only**. In a real financial institution, thresholds should be optimized using fraud investigation capacity, false-positive cost, false-negative cost, customer impact, and regulatory requirements.

---

## Generated Outputs

The project includes generated model and analysis outputs.

### Model

```text
models/fraud_random_forest.joblib
```

Serialized Random Forest model produced by the training pipeline.

### Runtime metrics

```text
reports/metrics_runtime.json
```

Contains test-set ROC-AUC, PR-AUC, precision, recall, F1 and confusion matrix.

### Evaluation report

```text
reports/evaluation.json
```

Contains full-dataset evaluation metrics and classification report.

### Batch predictions

```text
data/processed/fraud_predictions.csv
```

Contains transaction-level fraud probabilities and risk levels.

### Excel results workbook

```text
reports/fraud_detection_results.xlsx
```

The workbook is included with the project and contains the following sheets:

1. **Project Summary** — dataset and model KPIs
2. **Risk Summary** — LOW/MEDIUM/HIGH transaction distribution
3. **Model Metrics** — model performance metrics
4. **Predictions** — transaction-level fraud scores
5. **Confusion Matrix** — classification results

---

## SQL Analytics

The file:

```text
sql/fraud_analysis.sql
```

contains SQL analysis for:

- Overall transaction count
- Fraud transaction count
- Fraud rate
- Average transaction amount
- Maximum transaction amount
- Fraud by amount band
- Fraud by transaction hour

---

## Jupyter Notebook

The notebook:

```text
notebooks/01_fraud_detection.ipynb
```

provides an interactive workflow for:

- Loading the transaction dataset
- Inspecting data quality
- Exploring fraud distribution
- Examining transaction amounts
- Creating engineered features
- Training the Random Forest
- Evaluating model performance
- Reviewing fraud predictions

Run Jupyter with:

```bash
jupyter notebook
```

Then open:

```text
notebooks/01_fraud_detection.ipynb
```

---

## Testing & CI

The project includes Pytest tests under:

```text
tests/
```

Run locally:

```bash
python -m pytest -q
```

GitHub Actions is configured under:

```text
.github/workflows/tests.yml
```

The workflow automatically runs the test suite when changes are pushed to GitHub.

---

## Key Findings

1. **Fraud is a minority-class problem.** Accuracy alone is not sufficient for evaluating a fraud model.
2. **The class-weighted Random Forest performs strongly** on the held-out sample, achieving approximately **0.985 ROC-AUC** and **0.946 PR-AUC**.
3. **Recall is critical** because missing a fraudulent transaction can have a higher business cost than investigating a legitimate transaction.
4. **Precision is also important** because excessive false-positive alerts can overwhelm fraud-investigation teams and negatively affect customers.
5. **PR-AUC is especially useful** for this problem because it focuses on performance for the positive/fraud class.
6. **Probability-based risk scoring** allows transactions to be prioritized instead of treating every transaction equally.
7. **Threshold selection should be business-driven** in a real financial environment rather than relying on an arbitrary 0.50 threshold.

---

## Limitations

| Limitation | Details |
|---|---|
| Small working dataset | The attached working dataset contains 5,493 records and should not be treated as a production-scale fraud dataset. |
| Synthetic demonstration row | One legitimate demonstration record was added to reach the requested 5,493-record working dataset. |
| Anonymized features | `V1`–`V28` have limited direct business interpretability. |
| Class distribution | The sample's fraud ratio differs from many real-world transaction streams. |
| Threshold | The 0.50 fraud threshold is illustrative and not cost-optimized. |
| No temporal validation | The project uses a stratified random split rather than a time-based validation strategy. |
| No calibration | Fraud probabilities have not been calibrated for production decision-making. |
| No real-time streaming | Kafka/event-stream scoring is not included. |
| No drift monitoring | Production data/model drift monitoring is outside the current scope. |
| No external validation | Performance should be validated on an independent, representative dataset before deployment. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model | Random Forest Classifier |
| Model Serialization | Joblib |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |
| SQL Analytics | DuckDB / SQL |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Version Control | Git / GitHub |
| Output Reporting | CSV, JSON, Excel |

---

## Reproducibility

All model training uses:

```text
random_state = 42
```

To reproduce the main workflow:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python src/train_model.py
```

Generate predictions:

```bash
python src/predict.py --input data/raw/creditcard.csv --output data/processed/fraud_predictions.csv
```

Evaluate:

```bash
python src/evaluate_model.py
```

Run tests:

```bash
python -m pytest -q
```

Expected held-out test performance is approximately:

```text
ROC-AUC : 0.9848
PR-AUC  : 0.9457
Precision: 0.9655
Recall   : 0.8571
F1       : 0.9081
```

---
