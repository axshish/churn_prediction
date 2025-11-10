from __future__ import annotations
import argparse, json, sys, pandas as pd, joblib
from typing import List
def predict_rows(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(model_path)
    probs = model.predict_proba(df)[:,1]
    preds = (probs >= 0.5).astype(int)
    out = df.copy()
    out["churn_prob"] = probs
    out["churn_pred"] = preds
    return out
def main(args):
    if args.json:
        payload = json.loads(args.json)
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("JSON must be an object or list of objects")
    elif args.csv_path:
        df = pd.read_csv(args.csv_path)
    else:
        print("Provide --json or --csv_path")
        sys.exit(1)
    out = predict_rows(args.model_path, df)
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to {args.out_csv}")
    else:
        print(out.to_string(index=False))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/best_model.joblib")
    parser.add_argument("--json", type=str, help='JSON string of one or more records')
    parser.add_argument("--csv_path", type=str, help='Path to CSV file')
    parser.add_argument("--out_csv", type=str, help='Path to save CSV predictions')
    args = parser.parse_args()
    main(args)
