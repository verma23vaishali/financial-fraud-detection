import argparse, json, joblib
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from data_preprocessing import load_data
from features import split_features_target

def evaluate(data_path, model_path, output_path):
    df = load_data(data_path)
    X, y = split_features_target(df)
    model = joblib.load(model_path)
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= .50).astype(int)
    result = {
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
        "classification_report": classification_report(y, prediction, output_dict=True, zero_division=0)
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/creditcard.csv")
    p.add_argument("--model", default="models/fraud_random_forest.joblib")
    p.add_argument("--output", default="reports/evaluation.json")
    a = p.parse_args()
    evaluate(a.data, a.model, a.output)
