import argparse
import joblib
import pandas as pd
from features import engineer_features

def predict(input_csv, model_path, output_csv, threshold=.50):
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    model_input = df.drop(columns=["Class"], errors="ignore")
    probability = model.predict_proba(engineer_features(model_input))[:, 1]
    result = df.copy()
    result["fraud_probability"] = probability
    result["risk_level"] = pd.cut(
        probability, bins=[-.01,.20,.70,1.01],
        labels=["LOW","MEDIUM","HIGH"]
    )
    result["predicted_fraud"] = (probability >= threshold).astype(int)
    result.to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="models/fraud_random_forest.joblib")
    p.add_argument("--output", default="data/processed/predictions.csv")
    p.add_argument("--threshold", type=float, default=.50)
    a = p.parse_args()
    predict(a.input, a.model, a.output, a.threshold)
