from __future__ import annotations
import argparse
import joblib
from sklearn.metrics import classification_report
from .data import load_data, split_X_y, TARGET
def main(args):
    df = load_data(args.data_path)
    X, y = split_X_y(df, target=TARGET)
    model = joblib.load(args.model_path)
    preds = model.predict(X)
    print(classification_report(y, preds, digits=4))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
