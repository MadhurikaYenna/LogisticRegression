import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #9ec7d6;
}

.card {
    background-color: #f3a6e7;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 25px;
    color: white;
}

.center-box {
    display: flex;
    justify-content: center;
    margin-top: 25px;
}

.result-box {
    background-color: #d1fae5;
    color: #065f46;
    padding: 18px 30px;
    border-radius: 14px;
    text-align: center;
    font-size: 18px;
    font-weight: 700;
    width: fit-content;
    min-width: 320px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("""
<div class="card">
    <h2>Customer Churn Prediction</h2>
    <p>Using Logistic Regression to predict whether a customer is likely to churn or stay</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load dataset (ROBUST PATH)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(csv_path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -------------------------------------------------
# Dataset Preview
# -------------------------------------------------
st.markdown("<div class='section-title'>Dataset Preview</div>", unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)

# -------------------------------------------------
# Preprocessing (SAME LOGIC)
# -------------------------------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop("customerID", axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# Train model
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)
disp.plot(cmap="Blues", ax=ax, values_format="d")
st.pyplot(fig)

# -------------------------------------------------
# Model Performance
# -------------------------------------------------
tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_test, y_pred)

st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("True Positive (TP)", tp)
c3.metric("True Negative (TN)", tn)
c4.metric("False Positive (FP)", fp)

c5, c6 = st.columns(2)
c5.metric("False Negative (FN)", fn)
c6.metric("Total Predictions", len(y_test))

# -------------------------------------------------
# Confusion Matrix Meaning
# -------------------------------------------------
st.markdown("""
<div class="card">
<b>Confusion Matrix Meaning</b><br>
TP (True Positive): Correctly identified churn customers<br>
TN (True Negative): Correctly identified non-churn customers<br>
FP (False Positive): Non-churn predicted as churn<br>
FN (False Negative): Churn predicted as non-churn
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.markdown("<div class='section-title'>Predict Customer Churn</div>", unsafe_allow_html=True)

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 20.0, 120.0, 70.0)
total = st.slider("Total Charges", 20.0, 9000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract_" + contract: 1,
    "InternetService_" + internet: 1
}])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

# -------------------------------------------------
# Centered Prediction Result (NOT FULL WIDTH)
# -------------------------------------------------
st.markdown(f"""
<div class="center-box">
    <div class="result-box">
        Prediction: {"Likely to Churn" if pred==1 else "Likely to Stay"}<br>
        Churn Probability: {prob:.2f}
    </div>
</div>
""", unsafe_allow_html=True)
