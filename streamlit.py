# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading the trained model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and LabelEncoder
model = joblib.load("health_assistant_model.pkl")  # Ensure the model file is in the same directory
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit app interface
st.title("Health Assistant AI")
st.write("Predict possible diseases based on user symptoms.")

# Input fields for symptoms
symptoms = st.text_input("Enter your symptoms (comma-separated)", "e.g., headache, fever, cough")

if st.button("Predict"):
    if symptoms:
        # Preprocess symptoms input
        symptom_list = symptoms.split(",")
        symptom_array = np.array([symptom_list])  # Ensure it matches the model input shape
        
        # Predict using the trained model
        prediction = model.predict(symptom_array)
        predicted_disease = label_encoder.inverse_transform(prediction)
        
        st.success(f"Predicted Disease: {predicted_disease[0]}")
    else:
        st.error("Please enter symptoms.")

st.write("This app is powered by a machine learning model trained on health datasets.")
