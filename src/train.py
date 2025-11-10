from __future__ import annotations
import argparse
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .data import load_data, split_X_y, TARGET
from .features import build_preprocess_pipe
def build_candidates() -> List[Tuple[Any, Dict[str, Any]]]:
    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    search_space = [
        (lr, {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["l2"],
        }),
        (rf, {
            "model__n_estimators": list(range(100, 601, 50)),
            "model__max_depth": [None] + list(range(3, 21, 3)),
            "model__min_samples_split": [2, 5, 10],
        }),
        (gb, {
            "model__n_estimators": list(range(50, 401, 25)),
            "model__learning_rate": np.linspace(0.01, 0.3, 20),
            "model__max_depth": list(range(2, 7)),
        }),
    ]
    return search_space
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
def main(args: argparse.Namespace) -> None:
    df = load_data(args.data_path)
    X, y = split_X_y(df, target=TARGET)
    preprocessor, num_cols, cat_cols = build_preprocess_pipe(X)
    best_overall = None
    best_metrics = None
    best_name = None
    for estimator, param_dist in build_candidates():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=min(args.n_iter, sum(len(v) if hasattr(v, '__len__') else 20 for v in param_dist.values())),
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        search.fit(X_train, y_train)
        preds = search.best_estimator_.predict(X_test)
        metrics = evaluate(y_test, preds)
        print(f"Candidate: {estimator.__class__.__name__} -> {metrics} (best params: {search.best_params_})")
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_overall = search.best_estimator_
            best_name = estimator.__class__.__name__
    print(f"\nSelected model: {best_name} with metrics: {best_metrics}\n")
    joblib.dump(best_overall, args.out)
    print(f"Saved best model to {args.out}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with target column 'churn'")
    parser.add_argument("--out", type=str, default="models/best_model.joblib")
    parser.add_argument("--n_iter", type=int, default=25, help="RandomizedSearchCV iterations per candidate")
    args = parser.parse_args()
    main(args)
