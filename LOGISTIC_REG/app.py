import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# Title
# -----------------------------
st.title("📉 Customer Churn Prediction")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -----------------------------
# Encode target
# -----------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# Drop ID column
# -----------------------------
df.drop("customerID", axis=1, inplace=True)

# -----------------------------
# One-hot encoding (same logic)
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Split features & target
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# -----------------------------
# Sidebar input (simple)
# -----------------------------
st.sidebar.header("🧾 New Customer Input")

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 20.0, 120.0, 70.0)
total = st.sidebar.slider("Total Charges", 20.0, 9000.0, 1000.0)

input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# align columns
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.subheader("🔍 Prediction")

if pred == 1:
    st.error(f"⚠️ Likely to CHURN (Probability: {prob:.2%})")
else:
    st.success(f"✅ Likely to STAY (Probability: {prob:.2%})")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📊 Model Evaluation")

y_pred = model.predict(x_test)

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    values_format="d",
    ax=ax
)
st.pyplot(fig)

