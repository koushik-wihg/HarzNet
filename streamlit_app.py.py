import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Ultramafic MLOps Demo", layout="wide")

st.title("Ultramafic MLOps – Streamlit App")
st.write("This is a simple placeholder UI. Replace with prediction UI later.")

st.subheader("Sample Input")
col1, col2 = st.columns(2)

with col1:
    MgO = st.number_input("MgO (%)", 0.0, 60.0, 30.0)
    FeO = st.number_input("FeO (%)", 0.0, 40.0, 10.0)

with col2:
    SiO2 = st.number_input("SiO₂ (%)", 0.0, 80.0, 45.0)
    Al2O3 = st.number_input("Al₂O₃ (%)", 0.0, 40.0, 5.0)

if st.button("Predict"):
    st.success("Mock prediction result (replace with real model):  **Harzburgite**")

st.write("Upload XRF/Geochemical CSV for bulk processing:")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    st.success("CSV loaded successfully.")
