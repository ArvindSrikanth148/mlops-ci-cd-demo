import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("MLOps CI/CD Demo")

st.write("Deployment using:")
st.write("- GitHub Actions")
st.write("- DVC")
st.write("- AWS S3")
st.write("- Docker")
st.write("- Streamlit Cloud")

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("models/model.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# -----------------------------
# SPLIT FEATURES + TARGET
# -----------------------------

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# -----------------------------
# MODEL ACCURACY
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# PREDICTION SECTION
# -----------------------------

st.subheader("Make Prediction")

input_dict = {}

for col in X_test.columns:
    input_dict[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):

    prediction = model.predict(input_df)

    st.success(f"Prediction: {prediction[0]}")