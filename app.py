import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==========================================
# 1. Page Configuration & UI Theme
# ==========================================
st.set_page_config(
    page_title="Credit Risk Decision Engine", page_icon="🏦", layout="wide"
)

st.markdown(
    """
    <style>
        .main-title {
            background-color: #1E3A8A;
            color: white;
            padding: 18px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 25px;
        }
        .stMetric {
            background-color: #F8FAFC;
            padding: 8px;
            border-radius: 6px;
            border: 1px solid #E2E8F0;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-title">
        <h1>🏦 Credit Risk Assessment & Decision System</h1>
        <p>Machine learning evaluation for default prediction and real-time loan decisioning</p>
    </div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. Data Loading & Feature Engineering
# ==========================================
@st.cache_data
def load_credit_data():
    # Dynamic path handling to avoid FileNotFoundError
    base_dir = os.path.dirname(os.path.abspath(__file__))

    loans_path = os.path.join(base_dir, "loans (2).csv")
    customers_path = os.path.join(base_dir, "customers (2).csv")
    risk_path = os.path.join(base_dir, "risk assessment.csv")

    loans = pd.read_csv(loans_path)
    customers = pd.read_csv(customers_path)
    risk = pd.read_csv(risk_path)

    df = loans.merge(customers, on="customer_id", how="left").merge(
        risk, on="application_id", how="left"
    )

    df["Income"] = df["monthly_income"]
    df["LoanAmount"] = df["funded_amount"]
    df["DTI"] = df["original_dti"]
    df["CreditScore"] = df["credit_score"]
    df["HasDefault"] = df["previous_default_flag"].astype(int)
    df["CoverageRatio"] = (
        df["collateral_value"].fillna(0) / df["funded_amount"]
    ).fillna(0)
    df["IsDefault"] = df["default_flag"].astype(int)

    return df


dataset = load_credit_data()

# ==========================================
# 3. Model Training & Evaluation Setup
# ==========================================
feature_cols = [
    "Income",
    "LoanAmount",
    "DTI",
    "CreditScore",
    "HasDefault",
    "CoverageRatio",
]

X = dataset[feature_cols]
y = dataset["IsDefault"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
lr_model = LogisticRegression(class_weight="balanced", random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ==========================================
# 4. Streamlit Application Tabs
# ==========================================
tab_eval, tab_predict = st.tabs(
    ["📊 Model Evaluation", "🎯 Real-Time Loan Assessment"]
)

# ------------------------------------------
# TAB 1: Model Evaluation
# ------------------------------------------
with tab_eval:
    st.subheader("Model Performance Comparison")

    comparison_data = {
        "Metric": ["Accuracy", "ROC-AUC", "Recall", "Precision", "F1-Score"],
        "Logistic Regression": [
            f"{accuracy_score(y_test, y_pred_lr):.2%}",
            f"{auc(*roc_curve(y_test, y_prob_lr)[:2]):.2%}",
            f"{recall_score(y_test, y_pred_lr, zero_division=0):.2%}",
            f"{precision_score(y_test, y_pred_lr, zero_division=0):.2%}",
            f"{f1_score(y_test, y_pred_lr, zero_division=0):.2%}",
        ],
        "Random Forest": [
            f"{accuracy_score(y_test, y_pred_rf):.2%}",
            f"{auc(*roc_curve(y_test, y_prob_rf)[:2]):.2%}",
            f"{recall_score(y_test, y_pred_rf, zero_division=0):.2%}",
            f"{precision_score(y_test, y_pred_rf, zero_division=0):.2%}",
            f"{f1_score(y_test, y_pred_rf, zero_division=0):.2%}",
        ],
    }

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.markdown("**Performance Summary Table:**")
        st.dataframe(pd.DataFrame(comparison_data), width="stretch")

        st.info(
            "📌 **Analytical Insight:** Random Forest handles complex and non-linear risk factors effectively, whereas Logistic Regression offers straightforward model interpretability for credit compliance."
        )

    with col_chart:
        st.markdown("**ROC Curve Comparison:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

        ax.plot(
            fpr_lr,
            tpr_lr,
            label=f"Logistic Regression (AUC = {auc(fpr_lr, tpr_lr):.2f})",
            color="navy",
        )
        ax.plot(
            fpr_rf,
            tpr_rf,
            label=f"Random Forest (AUC = {auc(fpr_rf, tpr_rf):.2f})",
            color="darkgreen",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

# ------------------------------------------
# TAB 2: Real-Time Assessment
# ------------------------------------------
with tab_predict:
    st.subheader("New Loan Application Assessment")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Customer Profile**")
        monthly_income = st.number_input(
            "Monthly Income ($)", min_value=1000, value=25000, step=1000
        )
        credit_score = st.slider(
            "Credit Score (I-Score)", min_value=300, max_value=850, value=740
        )
        prior_default = st.selectbox(
            "Prior Default / Restructuring?", ["No", "Yes"], index=0
        )

    with c2:
        st.markdown("**Loan Request Details**")
        loan_amount = st.number_input(
            "Requested Amount ($)", min_value=1000, value=100000, step=5000
        )
        loan_term = st.selectbox(
            "Term Length (Months)", [12, 24, 36, 48, 60], index=2
        )
        monthly_obligations = st.number_input(
            "Existing Monthly Obligations ($)", min_value=0, value=3000
        )

    with c3:
        st.markdown("**Collateral & Model Selection**")
        collateral_val = st.number_input(
            "Collateral Value ($)", min_value=0, value=150000, step=10000
        )
        selected_model = st.radio(
            "Scoring Engine:",
            ["Random Forest Engine", "Logistic Regression Engine"],
        )

    # Risk Metrics Calculations
    interest_rate = 0.18 / 12
    monthly_installment = (
        loan_amount
        * interest_rate
        * (1 + interest_rate) ** loan_term
    ) / (((1 + interest_rate) ** loan_term) - 1)

    calculated_dti = (monthly_installment + monthly_obligations) / monthly_income
    coverage_ratio = collateral_val / loan_amount
    has_default_flag = 1 if prior_default == "Yes" else 0

    st.markdown("---")
    if st.button("Evaluate Application"):
        input_data = pd.DataFrame(
            [
                {
                    "Income": monthly_income,
                    "LoanAmount": loan_amount,
                    "DTI": calculated_dti,
                    "CreditScore": credit_score,
                    "HasDefault": has_default_flag,
                    "CoverageRatio": coverage_ratio,
                }
            ]
        )

        if "Logistic" in selected_model:
            input_scaled = scaler.transform(input_data)
            pd_probability = lr_model.predict_proba(input_scaled)[0][1]
        else:
            pd_probability = rf_model.predict_proba(input_data)[0][1]

        pd_percentage = pd_probability * 100
        dti_percentage = calculated_dti * 100
        coverage_percentage = coverage_ratio * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probability of Default (PD)", f"{pd_percentage:.2f}%")
        m2.metric("Debt-to-Income (DTI)", f"{dti_percentage:.1f}%")
        m3.metric("Collateral Coverage", f"{coverage_percentage:.1f}%")

        if pd_percentage < 25:
            risk_tier = "Low Risk"
        elif pd_percentage < 45:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "High Risk"

        m4.metric("Risk Rating", risk_tier)

        st.write("")
        if pd_percentage <= 25 and dti_percentage <= 45:
            st.success(
                "✅ **Approved:** The applicant demonstrates low default risk and an acceptable debt-to-income ratio."
            )
        elif pd_percentage <= 45 and dti_percentage <= 50:
            st.warning(
                "⚠️ **Conditional Approval:** Requires additional guarantor or income verification before underwriting."
            )
        else:
            st.error(
                "❌ **Declined:** High default probability or excessive debt-to-income ratio beyond policy limits."
            )