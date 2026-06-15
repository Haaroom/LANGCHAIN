import streamlit as st
import pandas as pd
import joblib
model = joblib.load("models/student_risk.pkl")
uploaded_file = st.file_uploader("Upload Student CSV",type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    predictions = model.predict(df)
    df["Prediction"] = predictions
    st.dataframe(df)
    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )