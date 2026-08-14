import argparse, json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data
from features import split_features_target

def train(data_path, model_path, metrics_path):
    df = load_data(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= .50).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, probability),
        "pr_auc": average_precision_score(y_test, probability),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist()
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/creditcard.csv")
    p.add_argument("--model", default="models/fraud_random_forest.joblib")
    p.add_argument("--metrics", default="reports/metrics_runtime.json")
    a = p.parse_args()
    train(a.data, a.model, a.metrics)
