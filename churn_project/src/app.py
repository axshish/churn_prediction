import joblib
import pandas as pd
import streamlit as st
import io
from pathlib import Path

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction")
st.write("Enter customer details to estimate churn probability.")
@st.cache_resource
def load_model():
    # return joblib.load("churn_project/models/best_model.joblib")
    model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
    print("Model loaded from:", model_path)
    return joblib.load(model_path)

col1, col2 = st.columns(2)
with col1:
    tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=50.0)
    num_support_tickets_90d = st.number_input("Support tickets (90d)", min_value=0, value=1)
    avg_session_length_mins = st.number_input("Avg session length (mins)", min_value=0.0, value=10.0, step=0.1)
    last_login_days_ago = st.number_input("Last login (days ago)", min_value=0, value=10)
    total_charges = tenure_months * monthly_charges
    st.markdown(f"<h4>Total charges: {total_charges:.2f}</h4>", unsafe_allow_html=True)
with col2:
    contract_type = st.selectbox("Contract type", ["month-to-month", "one-year", "two-year"])
    payment_method = st.selectbox("Payment method", ["credit_card", "debit_card", "bank_transfer", "upi", "net banking"])
    internet_service = st.selectbox("Internet service", ["wifi", "mobile data", "none"])
    auto_renew = st.selectbox("Auto renew", ["No", "Yes"])
    promo_eligible = st.selectbox("Promo eligible", ["No", "Yes"])
    st.write("")
if st.button("Predict", use_container_width=True):
    model = load_model()
    record = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "num_support_tickets_90d": num_support_tickets_90d,
        "avg_session_length_mins": avg_session_length_mins,
        "last_login_days_ago": last_login_days_ago,
        "auto_renew": 1 if auto_renew == "Yes" else 0,
        "promo_eligible": 1 if promo_eligible == "Yes" else 0,
    }])
    try:
        proba = model.predict_proba(record)[:,1][0]
        pred = int(proba >= 0.5)
        st.metric("Churn probability", f"{proba:.2%}")
        if pred == 1:
            prediction_text = "This customer is likely to <span style='color: red;'>churn</span>."
        else:
            prediction_text = "This customer is likely to be <span style='color: green;'>retained</span>."
        st.markdown(f"<h3>{prediction_text}</h3>", unsafe_allow_html=True)
        st.subheader("Suggestions to Reduce Churn💡")
        st.markdown("Here are actionable strategies to lower the churn probability based on simulated changes:")
        suggestions = []
        base_record = record.copy()
        new_record = base_record.copy()
        new_record["tenure_months"] *= 1.1
        new_proba = model.predict_proba(new_record)[:,1][0]
        if new_proba < proba:
            suggestions.append(f"📈 **Offer loyalty incentives** to increase tenure by 10% (reduces probability by {(proba - new_proba):.1%})")
        new_record = base_record.copy()
        new_record["monthly_charges"] *= 0.9
        new_record["total_charges"] = new_record["tenure_months"] * new_record["monthly_charges"]
        new_proba = model.predict_proba(new_record)[:,1][0]
        if new_proba < proba:
            suggestions.append(f"💰 **Provide a discount** to decrease monthly charges by 10% (reduces probability by {(proba - new_proba):.1%})")
        if auto_renew == "No":
            new_record = base_record.copy()
            new_record["auto_renew"] = 1
            new_proba = model.predict_proba(new_record)[:,1][0]
            if new_proba < proba:
                suggestions.append(f"🔄 **Encourage auto-renewal** by enabling it (reduces probability by {(proba - new_proba):.1%})")
        if promo_eligible == "No":
            new_record = base_record.copy()
            new_record["promo_eligible"] = 1
            new_proba = model.predict_proba(new_record)[:,1][0]
            if new_proba < proba:
                suggestions.append(f"🎁 **Apply promotional offers** to make the customer eligible (reduces probability by {(proba - new_proba):.1%})")
        new_record = base_record.copy()
        new_record["num_support_tickets_90d"] *= 0.5
        new_proba = model.predict_proba(new_record)[:,1][0]
        if new_proba < proba:
            suggestions.append(f"🛠️ **Improve customer support** to reduce tickets by 50% (reduces probability by {(proba - new_proba):.1%})")
        if contract_type == "month-to-month":
            new_record = base_record.copy()
            new_record["contract_type"] = "one-year"
            new_proba = model.predict_proba(new_record)[:,1][0]
            if new_proba < proba:
                suggestions.append(f"📅 **Upgrade to a one-year contract** (reduces probability by {(proba - new_proba):.1%})")
        if internet_service == "none":
            new_record = base_record.copy()
            new_record["internet_service"] = "wifi"
            new_proba = model.predict_proba(new_record)[:,1][0]
            if new_proba < proba:
                suggestions.append(f"🌐 **Offer internet service upgrade** to wifi (reduces probability by {(proba - new_proba):.1%})")
        if suggestions:
            st.markdown("**Suggestions:**")
            for sug in suggestions:
                st.markdown(f"- {sug}")
        else:
            st.write("No adjustments found that would reduce churn probability.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
st.sidebar.header("📊 Bulk Churn Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        st.sidebar.write(f"Rows: {len(df)}")
        required_cols = [
            "tenure_months", "monthly_charges", "total_charges", "contract_type",
            "payment_method", "internet_service", "num_support_tickets_90d",
            "avg_session_length_mins", "last_login_days_ago", "auto_renew", "promo_eligible"
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.sidebar.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            df_copy = df.copy()
            df_copy["auto_renew"] = df_copy["auto_renew"].map({"Yes": 1, "No": 0}).fillna(df_copy["auto_renew"])
            df_copy["promo_eligible"] = df_copy["promo_eligible"].map({"Yes": 1, "No": 0}).fillna(df_copy["promo_eligible"])
            model = load_model()
            probas = model.predict_proba(df_copy[required_cols])[:, 1]
            preds = (probas >= 0.5).astype(int)
            df["churn_probability"] = probas
            df["churn_prediction"] = pd.Series(preds).map({1: "Churn", 0: "Retain"})
            st.sidebar.write("### Prediction Results")
            st.sidebar.dataframe(df[["churn_probability", "churn_prediction"]])
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")
