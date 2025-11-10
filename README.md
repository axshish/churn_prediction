# Customer Churn Prediction – Subscription Businesses

A complete, production-style project to predict churn risk for subscription customers.
It includes data handling, feature engineering, multiple models with hyperparameter search,
evaluation, a simple UI for predictions, and packaging for quick local deployment.

## Quickstart (Local)

```bash
# 1) Unzip the project, then open a terminal in the project folder
# 2) Create a virtual environment and install
bash install.sh    # Mac/Linux
# or on Windows PowerShell:
./install.bat

# 3) Train a model
python -m src.train --data_path data/customers.csv --out models/best_model.joblib

# 4) Evaluate (optional; train prints metrics too)
python -m src.evaluate --model_path models/best_model.joblib --data_path data/customers.csv

# 5) Run the UI
streamlit run src/app.py
```

## Project Structure

```
churn_project/
├── data/                 # raw & processed data
│   └── customers.csv     # example dummy dataset
├── models/               # trained models & artifacts
├── notebooks/            # EDA & modeling notebooks
├── src/                  # source code
│   ├── __init__.py
│   ├── data.py           # loading & basic preprocessing
│   ├── features.py       # feature engineering transforms
│   ├── train.py          # train + hyperparameter search + save model
│   ├── evaluate.py       # evaluation on a dataset
│   ├── predict.py        # CLI for single/CSV predictions
│   └── app.py            # Streamlit UI
├── requirements.txt
├── install.sh            # environment creation + pip install
├── install.bat           # Windows setup
└── README.md
```

## Data Fields (example)

- **customer_id** (string)
- **tenure_months** (int)
- **monthly_charges** (float, may be missing)
- **total_charges** (float, may be missing)
- **contract_type** (categorical)
- **payment_method** (categorical, may be missing)
- **internet_service** (categorical, may be missing)
- **num_support_tickets_90d** (int)
- **avg_session_length_mins** (float, may be missing)
- **last_login_days_ago** (int)
- **auto_renew** (0/1)
- **promo_eligible** (0/1)
- **churn** (0/1 target)

## Modeling

- Algorithms: Logistic Regression, Random Forest, Gradient Boosting (XGBoost alternative not used to keep deps lean).
- Hyperparameter Search: RandomizedSearchCV over model-specific grids.
- Metrics: accuracy, precision, recall, F1.

The pipeline uses a `ColumnTransformer` with imputation, scaling for numeric features,
and one-hot encoding for categoricals. The trained model is persisted with `joblib`.

## Deployment/UI

- `src/app.py` exposes a Streamlit app to collect inputs and display churn probability.
- `src/predict.py` supports both a JSON dict and CSV file predictions.

## Notes

- Replace the dummy dataset with your real subscription data to get meaningful results.
- Track experiments by saving different model files under `/models`.
- Extend hyperparameter grids or swap in your preferred libraries if needed.
